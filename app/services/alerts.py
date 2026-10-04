import logging
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy.orm import Session

from app.exceptions import ConflictError, NotFoundError
from app.models.alerts import AlertType
from app.models.companies import Company
from app.models.ingestion import IngestionRun
from app.repositories import alerts as alert_repository
from app.repositories import audit as audit_repository
from app.repositories import companies as company_repository
from app.repositories import notifications as notification_repository
from app.repositories import prices as price_repository
from app.schemas.alerts import AlertCreate, AlertResponse, AlertUpdate
from app.services import ingestion as ingestion_service

logger = logging.getLogger(__name__)

JOB_TYPE = "alert_evaluation"
MAX_ALERTS_PER_USER = 50
# Evaluation looks at the bars of this many days. A run that was missed is caught up by the next
# one (the unique constraint on notifications prevents duplicates); an outage longer than this
# loses crossings
EVALUATION_WINDOW_DAYS = 30


def check_can_create(db: Session, org_id: int, user_id: int, data: AlertCreate) -> Company:
    # The checks of create_alert, without writing. Also used by the agent's create_alert tool, so
    # the user is never asked to approve an alert that could not be created anyway
    company = company_repository.get_by_ticker(db, data.ticker.upper())
    if company is None:
        raise NotFoundError("Company not found")

    if alert_repository.count_by_user(db, org_id, user_id) >= MAX_ALERTS_PER_USER:
        raise ConflictError("Alert limit reached")

    # The same user may not have the same alert twice (another user may)
    duplicate = alert_repository.get_duplicate(
        db, org_id, user_id, company.id, data.alert_type, data.threshold
    )
    if duplicate:
        raise ConflictError("You already have this alert")
    return company


def create_alert(db: Session, org_id: int, user_id: int, data: AlertCreate) -> AlertResponse:
    company = check_can_create(db, org_id, user_id, data)

    # The containers run in UTC, so date.today() is the UTC date. Bars before this date can
    # never fire the new alert
    alert = alert_repository.create(
        db, org_id, user_id, company.id, data.alert_type, data.threshold, watch_from=date.today()
    )

    audit_repository.create(db, org_id, user_id, action="alert.create", entity_id=alert.id)
    db.commit()
    return AlertResponse.model_validate(alert)


def list_alerts(db: Session, org_id: int, user_id: int, active: bool | None) -> list[AlertResponse]:
    alerts = alert_repository.list_by_user(db, org_id, user_id, active)
    return [AlertResponse.model_validate(alert) for alert in alerts]


def get_alert(db: Session, org_id: int, user_id: int, alert_id: int) -> AlertResponse:
    alert = alert_repository.get_by_id(db, org_id, user_id, alert_id)

    # Same message whether the alert does not exist or belongs to someone else (in this or
    # another organization), so the existence of other people's alerts is not leaked
    if alert is None:
        raise NotFoundError("Alert not found")
    return AlertResponse.model_validate(alert)


def update_alert(
    db: Session, org_id: int, user_id: int, alert_id: int, data: AlertUpdate
) -> AlertResponse:
    alert = alert_repository.get_by_id(db, org_id, user_id, alert_id)
    if alert is None:
        raise NotFoundError("Alert not found")

    # A changed threshold or a re-activation starts watching from today, so old market moves
    # never fire a changed alert. Setting a field to the value it already has changes nothing
    if data.threshold is not None and data.threshold != alert.threshold:
        alert.threshold = data.threshold
        alert.watch_from = date.today()
    if data.active is not None:
        if data.active and not alert.active:
            alert.watch_from = date.today()
        alert.active = data.active

    audit_repository.create(db, org_id, user_id, action="alert.update", entity_id=alert.id)
    db.commit()
    return AlertResponse.model_validate(alert)


def delete_alert(db: Session, org_id: int, user_id: int, alert_id: int) -> None:
    alert = alert_repository.get_by_id(db, org_id, user_id, alert_id)
    if alert is None:
        raise NotFoundError("Alert not found")

    # The database cascade removes the alert's notifications
    alert_repository.delete(db, alert)

    audit_repository.create(db, org_id, user_id, action="alert.delete", entity_id=alert_id)
    db.commit()


def evaluate_alerts(db: Session, today: date | None = None) -> IngestionRun:
    # `today` exists only so tests control the date
    # 1. Guard against an overlapping run and record this one
    run = ingestion_service.start_run(db, JOB_TYPE, "alert evaluation")

    # Same single broad except as the other jobs: the run row must not stay "running".
    # There is no per-alert try/except: this job calls no external service, so a database error
    # should fail the run loudly. Everything is committed by finish_run
    try:
        # 2. Only bars of the last EVALUATION_WINDOW_DAYS days are looked at
        today = today or date.today()
        since = today - timedelta(days=EVALUATION_WINDOW_DAYS)

        bars_by_company: dict[int, list] = {}
        alerts_evaluated = 0
        alerts_skipped = 0
        notifications_created = 0

        # 3. Every active alert of every organization
        for alert in alert_repository.list_active(db):
            # Each company's bars are loaded once per run, oldest first
            if alert.company_id not in bars_by_company:
                bars_by_company[alert.company_id] = price_repository.list_since(
                    db, alert.company_id, since
                )
            bars = bars_by_company[alert.company_id]

            # A change needs a previous bar
            if len(bars) < 2:
                alerts_skipped += 1
                continue
            alerts_evaluated += 1

            # Days that already have a notification are skipped (this makes a re-run harmless)
            notified_dates = notification_repository.get_trade_dates_by_alert(db, alert.id)
            ticker = alert.company.ticker

            # Walk the bars oldest first, comparing each close with the previous bar's close
            for position in range(1, len(bars)):
                current = bars[position]
                previous = bars[position - 1]
                if current.trade_date < alert.watch_from or current.trade_date in notified_dates:
                    continue

                # The alert fires only on the day the close CROSSES the threshold, so an alert
                # whose condition is already true when it is created stays silent until the next
                # crossing. All comparisons use the split-adjusted close, not adj_close
                if alert.alert_type == AlertType.price_above:
                    if not (current.close > alert.threshold and previous.close <= alert.threshold):
                        continue
                    trigger_value = current.close
                    message = (
                        f"{ticker} closed at {current.close:.4f} on {current.trade_date}, "
                        f"crossing above your threshold of {alert.threshold:.4f} "
                        f"(previous close {previous.close:.4f})"
                    )
                elif alert.alert_type == AlertType.price_below:
                    if not (current.close < alert.threshold and previous.close >= alert.threshold):
                        continue
                    trigger_value = current.close
                    message = (
                        f"{ticker} closed at {current.close:.4f} on {current.trade_date}, "
                        f"crossing below your threshold of {alert.threshold:.4f} "
                        f"(previous close {previous.close:.4f})"
                    )
                else:
                    # daily_change_pct: the move in percent, signed. A previous close of 0
                    # cannot be divided by
                    if previous.close == 0:
                        continue
                    change = ((current.close - previous.close) / previous.close * 100).quantize(
                        Decimal("0.0001")
                    )
                    if abs(change) < alert.threshold:
                        continue
                    trigger_value = change
                    message = (
                        f"{ticker} moved {change:+.4f}% on {current.trade_date} "
                        f"(close {current.close:.4f} vs previous close {previous.close:.4f}), "
                        f"at or beyond your threshold of {alert.threshold:.4f}%"
                    )

                # The notification belongs to the alert's owner
                notification_repository.create(
                    db,
                    alert.org_id,
                    alert.user_id,
                    alert.id,
                    alert.company_id,
                    current.trade_date,
                    trigger_value,
                    message,
                )
                notifications_created += 1

        # 4. Finish the run (this commits everything)
        message = (
            f"Alerts evaluated: {alerts_evaluated}, notifications created: "
            f"{notifications_created}, alerts skipped (fewer than 2 bars): {alerts_skipped}"
        )
        ingestion_service.finish_run(db, run, 0, message)
        logger.info("Alert evaluation finished (%s): %s", run.status, run.message)
        return run
    except Exception as error:
        # 5. Record the failure and re-raise it
        ingestion_service.fail_run(db, run, error)
        logger.exception("Alert evaluation failed")
        raise
