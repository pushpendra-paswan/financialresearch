from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.exceptions import ConflictError
from app.models.alerts import Alert
from app.models.companies import Company
from app.models.ingestion import IngestionRun
from app.models.notifications import Notification
from app.models.users import User
from app.repositories import alerts as alert_repository
from app.repositories import companies as company_repository
from app.repositories import ingestion as ingestion_repository
from app.repositories import notifications as notification_repository
from app.repositories import organizations as organization_repository
from app.repositories import prices as price_repository
from app.repositories import users as user_repository
from app.services import alerts as alert_service

# A fixed date, so the tests are deterministic and never go stale
TODAY = date(2026, 9, 30)


# --- Set-up helpers (everything is created through the repositories) ---


@pytest.fixture
def companies(db: Session) -> dict[str, Company]:
    apple = company_repository.create(db, "AAPL", "0000320193", "Apple Inc.", "Nasdaq")
    microsoft = company_repository.create(db, "MSFT", "0000789019", "Microsoft Corp", "Nasdaq")
    db.commit()
    return {"AAPL": apple, "MSFT": microsoft}


@pytest.fixture
def owners(db: Session) -> dict[str, User]:
    # Two users in two different organizations. The password hash is never used
    acme = organization_repository.create(db, "Acme")
    globex = organization_repository.create(db, "Globex")
    ann = user_repository.create(db, acme.id, "ann@acme.com", "not-a-real-hash", "viewer")
    bob = user_repository.create(db, globex.id, "bob@globex.com", "not-a-real-hash", "analyst")
    db.commit()
    return {"ann": ann, "bob": bob}


def add_bars(
    db: Session, company: Company, closes: list[str], last_day: date = TODAY
) -> list[date]:
    # One bar per calendar day, the last one on last_day. Returns the days, oldest first.
    # Only close matters to the evaluation; the other columns just repeat it
    days = [
        last_day - timedelta(days=len(closes) - 1 - position) for position in range(len(closes))
    ]
    for day, close in zip(days, closes, strict=True):
        price = Decimal(close)
        price_repository.create(db, company.id, day, price, price, price, price, price, 1000)
    db.commit()
    return days


def add_alert(
    db: Session,
    owner: User,
    company: Company,
    alert_type: str,
    threshold: str,
    watch_from: date,
    active: bool = True,
) -> Alert:
    alert = alert_repository.create(
        db, owner.org_id, owner.id, company.id, alert_type, Decimal(threshold), watch_from, active
    )
    db.commit()
    return alert


def notifications_of(db: Session, alert: Alert) -> list[Notification]:
    statement = (
        select(Notification)
        .where(Notification.alert_id == alert.id)
        .order_by(Notification.trade_date)
    )
    return list(db.execute(statement).scalars().all())


def count_notifications(db: Session) -> int:
    return db.execute(select(func.count()).select_from(Notification)).scalar_one()


# --- price_above and price_below ---


def test_price_above_fires_once_on_the_crossing_day(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    days = add_bars(db, companies["AAPL"], ["95", "98", "101", "103"])
    alert = add_alert(db, owners["ann"], companies["AAPL"], "price_above", "100", days[0])

    alert_service.evaluate_alerts(db, today=TODAY)

    found = notifications_of(db, alert)
    assert len(found) == 1
    assert found[0].trade_date == days[2]
    assert found[0].trigger_value == Decimal("101")
    assert "AAPL" in found[0].message
    assert str(days[2]) in found[0].message
    assert found[0].is_read is False
    assert found[0].org_id == owners["ann"].org_id
    assert found[0].user_id == owners["ann"].id
    assert found[0].company_id == companies["AAPL"].id


def test_price_above_needs_a_strict_crossing(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    # Touching the threshold is not above it. Going from exactly 100 to 101 is a crossing
    days = add_bars(db, companies["AAPL"], ["99", "100", "100", "101"])
    alert = add_alert(db, owners["ann"], companies["AAPL"], "price_above", "100", days[0])

    alert_service.evaluate_alerts(db, today=TODAY)

    assert [n.trade_date for n in notifications_of(db, alert)] == [days[3]]


def test_price_above_is_silent_when_the_close_is_already_above(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    days = add_bars(db, companies["AAPL"], ["105", "106", "107", "108"])
    alert = add_alert(db, owners["ann"], companies["AAPL"], "price_above", "100", days[0])

    run = alert_service.evaluate_alerts(db, today=TODAY)

    assert notifications_of(db, alert) == []
    assert "notifications created: 0" in run.message


def test_price_below_fires_once_on_the_crossing_day(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    days = add_bars(db, companies["AAPL"], ["105", "102", "99", "97"])
    alert = add_alert(db, owners["ann"], companies["AAPL"], "price_below", "100", days[0])

    alert_service.evaluate_alerts(db, today=TODAY)

    found = notifications_of(db, alert)
    assert [n.trade_date for n in found] == [days[2]]
    assert found[0].trigger_value == Decimal("99")
    assert "AAPL" in found[0].message


def test_price_below_is_silent_when_the_close_is_already_below(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    days = add_bars(db, companies["AAPL"], ["95", "94", "93", "92"])
    alert = add_alert(db, owners["ann"], companies["AAPL"], "price_below", "100", days[0])

    alert_service.evaluate_alerts(db, today=TODAY)

    assert notifications_of(db, alert) == []


# --- daily_change_pct ---


def test_daily_change_pct_fires_on_big_moves_with_the_signed_change(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    # +4.0000 (no), +5.7692 (yes), -9.0909 (yes)
    days = add_bars(db, companies["AAPL"], ["100", "104", "110", "100"])
    alert = add_alert(db, owners["ann"], companies["AAPL"], "daily_change_pct", "5", days[0])

    alert_service.evaluate_alerts(db, today=TODAY)

    found = notifications_of(db, alert)
    assert [n.trade_date for n in found] == [days[2], days[3]]
    assert [n.trigger_value for n in found] == [Decimal("5.7692"), Decimal("-9.0909")]
    assert "+5.7692%" in found[0].message
    assert "-9.0909%" in found[1].message


def test_daily_change_pct_skips_a_previous_close_of_zero(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    # 0 -> 10 cannot be divided; 10 -> 10.5 is exactly +5 percent, which is "at least 5"
    days = add_bars(db, companies["AAPL"], ["0", "10", "10.5"])
    alert = add_alert(db, owners["ann"], companies["AAPL"], "daily_change_pct", "5", days[0])

    run = alert_service.evaluate_alerts(db, today=TODAY)

    assert run.status == "success"
    found = notifications_of(db, alert)
    assert [n.trade_date for n in found] == [days[2]]
    assert found[0].trigger_value == Decimal("5.0000")


# --- watch_from, inactive alerts and the window ---


def test_a_crossing_before_watch_from_does_not_fire(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    # Crossings above on day 1 and day 3
    days = add_bars(db, companies["AAPL"], ["95", "101", "95", "101"])
    alert = add_alert(db, owners["ann"], companies["AAPL"], "price_above", "100", days[2])

    alert_service.evaluate_alerts(db, today=TODAY)

    assert [n.trade_date for n in notifications_of(db, alert)] == [days[3]]


def test_a_crossing_exactly_on_watch_from_fires(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    days = add_bars(db, companies["AAPL"], ["95", "101", "95", "101"])
    alert = add_alert(db, owners["ann"], companies["AAPL"], "price_above", "100", days[1])

    alert_service.evaluate_alerts(db, today=TODAY)

    assert [n.trade_date for n in notifications_of(db, alert)] == [days[1], days[3]]


def test_inactive_alerts_are_ignored(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    days = add_bars(db, companies["AAPL"], ["95", "101"])
    alert = add_alert(
        db, owners["ann"], companies["AAPL"], "price_above", "100", days[0], active=False
    )

    run = alert_service.evaluate_alerts(db, today=TODAY)

    assert notifications_of(db, alert) == []
    assert "Alerts evaluated: 0" in run.message


def test_bars_older_than_the_window_are_not_evaluated(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    # A crossing 38 days ago (outside the 30-day window), then quiet recent bars
    old_days = add_bars(db, companies["AAPL"], ["95", "101"], last_day=TODAY - timedelta(days=38))
    add_bars(db, companies["AAPL"], ["101", "102"], last_day=TODAY)
    alert = add_alert(db, owners["ann"], companies["AAPL"], "price_above", "100", old_days[0])

    alert_service.evaluate_alerts(db, today=TODAY)

    assert notifications_of(db, alert) == []


# --- Idempotency and catch-up ---


def test_a_second_run_creates_nothing(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    days = add_bars(db, companies["AAPL"], ["95", "101", "95", "101"])
    add_alert(db, owners["ann"], companies["AAPL"], "price_above", "100", days[0])

    first = alert_service.evaluate_alerts(db, today=TODAY)
    count_after_first = count_notifications(db)
    second = alert_service.evaluate_alerts(db, today=TODAY)

    assert "notifications created: 2" in first.message
    assert count_after_first == 2
    assert "notifications created: 0" in second.message
    assert count_notifications(db) == 2


def test_the_next_run_catches_up_on_missed_days(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    first_days = add_bars(
        db, companies["AAPL"], ["95", "98", "99"], last_day=TODAY - timedelta(days=3)
    )
    alert = add_alert(db, owners["ann"], companies["AAPL"], "price_above", "100", first_days[0])
    alert_service.evaluate_alerts(db, today=TODAY - timedelta(days=3))
    assert notifications_of(db, alert) == []

    # Three days pass without a run. Closes 99 -> 101 (crossing), 101 -> 99, 99 -> 102 (crossing)
    new_days = add_bars(db, companies["AAPL"], ["101", "99", "102"], last_day=TODAY)
    run = alert_service.evaluate_alerts(db, today=TODAY)

    assert "notifications created: 2" in run.message
    assert [n.trade_date for n in notifications_of(db, alert)] == [new_days[0], new_days[2]]


# --- Several users and several companies ---


def test_two_users_in_two_organizations_each_get_their_own_notification(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    days = add_bars(db, companies["AAPL"], ["95", "101"])
    ann_alert = add_alert(db, owners["ann"], companies["AAPL"], "price_above", "100", days[0])
    bob_alert = add_alert(db, owners["bob"], companies["AAPL"], "price_above", "100", days[0])

    alert_service.evaluate_alerts(db, today=TODAY)

    ann_found = notifications_of(db, ann_alert)
    bob_found = notifications_of(db, bob_alert)
    assert len(ann_found) == len(bob_found) == 1
    assert (ann_found[0].org_id, ann_found[0].user_id) == (owners["ann"].org_id, owners["ann"].id)
    assert (bob_found[0].org_id, bob_found[0].user_id) == (owners["bob"].org_id, owners["bob"].id)
    assert owners["ann"].org_id != owners["bob"].org_id


def test_a_company_with_fewer_than_two_bars_is_skipped(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    one_bar_days = add_bars(db, companies["MSFT"], ["101"])
    add_alert(db, owners["ann"], companies["MSFT"], "price_above", "100", one_bar_days[0])
    days = add_bars(db, companies["AAPL"], ["95", "101"])
    apple_alert = add_alert(db, owners["ann"], companies["AAPL"], "price_above", "100", days[0])
    # A company without any bars is skipped too
    nvidia = company_repository.create(db, "NVDA", "0001045810", "NVIDIA Corp", "Nasdaq")
    add_alert(db, owners["bob"], nvidia, "price_below", "50", TODAY)

    run = alert_service.evaluate_alerts(db, today=TODAY)

    assert run.status == "success"
    assert len(notifications_of(db, apple_alert)) == 1
    assert count_notifications(db) == 1
    assert "alerts skipped (fewer than 2 bars): 2" in run.message


# --- The run row ---


def test_the_run_row_records_both_counts(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    days = add_bars(db, companies["AAPL"], ["95", "101"])
    add_alert(db, owners["ann"], companies["AAPL"], "price_above", "100", days[0])
    add_alert(db, owners["ann"], companies["AAPL"], "price_below", "50", days[0])

    run = alert_service.evaluate_alerts(db, today=TODAY)

    assert run.job_type == "alert_evaluation"
    assert run.status == "success"
    assert run.finished_at is not None
    assert run.error is None
    assert "Alerts evaluated: 2" in run.message
    assert "notifications created: 1" in run.message


def start_running_run(db: Session, job_type: str, started_ago: timedelta) -> None:
    run = ingestion_repository.create_run(db, job_type)
    run.started_at = datetime.now(UTC) - started_ago
    db.commit()


def test_a_recent_running_run_blocks_a_new_one(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    days = add_bars(db, companies["AAPL"], ["95", "101"])
    add_alert(db, owners["ann"], companies["AAPL"], "price_above", "100", days[0])
    start_running_run(db, "alert_evaluation", timedelta(minutes=10))

    with pytest.raises(ConflictError) as error:
        alert_service.evaluate_alerts(db, today=TODAY)

    assert "already in progress" in error.value.message
    # Nothing was created: only the existing run row, and no notifications
    assert db.execute(select(func.count()).select_from(IngestionRun)).scalar_one() == 1
    assert count_notifications(db) == 0


def test_a_stale_running_run_does_not_block(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    start_running_run(db, "alert_evaluation", timedelta(hours=3))

    run = alert_service.evaluate_alerts(db, today=TODAY)

    assert run.status == "success"


@pytest.mark.parametrize("other_job_type", ["prices", "ingest_filings", "financial_facts"])
def test_a_running_run_of_another_job_does_not_block(
    db: Session, companies: dict[str, Company], owners: dict[str, User], other_job_type: str
) -> None:
    start_running_run(db, other_job_type, timedelta(minutes=10))

    run = alert_service.evaluate_alerts(db, today=TODAY)

    assert run.status == "success"


def test_an_unexpected_error_marks_the_run_failed_and_is_raised(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_list_active(db: Session) -> list[Alert]:
        raise RuntimeError("boom")

    monkeypatch.setattr(alert_repository, "list_active", broken_list_active)

    with pytest.raises(RuntimeError):
        alert_service.evaluate_alerts(db, today=TODAY)

    run = db.execute(select(IngestionRun)).scalar_one()
    assert run.job_type == "alert_evaluation"
    assert run.status == "failed"
    assert run.error == "RuntimeError: boom"
    assert run.finished_at is not None


# --- The database constraint ---


def test_a_second_notification_for_the_same_alert_and_day_is_rejected(
    db: Session, companies: dict[str, Company], owners: dict[str, User]
) -> None:
    alert = add_alert(db, owners["ann"], companies["AAPL"], "price_above", "100", TODAY)
    arguments = (alert.org_id, alert.user_id, alert.id, alert.company_id, TODAY, Decimal("101"))
    notification_repository.create(db, *arguments, "first")

    with pytest.raises(IntegrityError):
        notification_repository.create(db, *arguments, "second")
    db.rollback()


# --- Celery wiring ---


def test_celery_registration() -> None:
    # No database needed: only the Celery app is inspected
    from app.workers import tasks  # noqa: F401  (importing registers the tasks)
    from app.workers.celery_app import celery_app

    scheduled_tasks = [entry["task"] for entry in celery_app.conf.beat_schedule.values()]

    assert "evaluate_alerts" in celery_app.tasks
    # Chained after ingest_prices, so it has no schedule of its own
    assert "evaluate_alerts" not in scheduled_tasks
    assert "ingest_prices" in celery_app.tasks
    assert "ingest_prices" in scheduled_tasks


@pytest.fixture
def delay_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    from app.workers import tasks

    calls: list[int] = []
    monkeypatch.setattr(tasks.evaluate_alerts, "delay", lambda: calls.append(1))
    return calls


@pytest.mark.parametrize("status", ["success", "partial"])
def test_the_price_task_enqueues_the_evaluation_once(
    monkeypatch: pytest.MonkeyPatch, delay_calls: list[int], status: str
) -> None:
    from app.workers import tasks

    run = SimpleNamespace(id=1, status=status, message="done")
    monkeypatch.setattr(tasks.price_service, "ingest_prices", lambda db: run)

    result = tasks.ingest_prices()

    assert delay_calls == [1]
    assert result == f"{status}: done"


def test_the_price_task_does_not_enqueue_when_the_run_is_skipped(
    monkeypatch: pytest.MonkeyPatch, delay_calls: list[int]
) -> None:
    from app.workers import tasks

    def skipped(db: Session) -> None:
        raise ConflictError("A price ingestion run is already in progress")

    monkeypatch.setattr(tasks.price_service, "ingest_prices", skipped)

    assert tasks.ingest_prices() is None
    assert delay_calls == []


def test_the_price_task_does_not_enqueue_when_the_service_fails(
    monkeypatch: pytest.MonkeyPatch, delay_calls: list[int]
) -> None:
    from app.workers import tasks

    def broken(db: Session) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(tasks.price_service, "ingest_prices", broken)

    with pytest.raises(RuntimeError):
        tasks.ingest_prices()
    assert delay_calls == []
