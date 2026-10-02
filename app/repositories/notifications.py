# Notifications are private and personal: every function that reads them takes org_id AND
# user_id, except get_trade_dates_by_alert (see the comment there).
from datetime import date
from decimal import Decimal

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.models.notifications import Notification


def create(
    db: Session,
    org_id: int,
    user_id: int,
    alert_id: int,
    company_id: int,
    trade_date: date,
    trigger_value: Decimal,
    message: str,
) -> Notification:
    notification = Notification(
        org_id=org_id,
        user_id=user_id,
        alert_id=alert_id,
        company_id=company_id,
        trade_date=trade_date,
        trigger_value=trigger_value,
        message=message,
    )
    db.add(notification)
    db.flush()
    return notification


def get_trade_dates_by_alert(db: Session, alert_id: int) -> set[date]:
    # Used only by the evaluation job, which works across all organizations and already holds
    # the alert, so this takes no org_id
    statement = select(Notification.trade_date).where(Notification.alert_id == alert_id)
    return set(db.execute(statement).scalars().all())


def list_by_user(
    db: Session, org_id: int, user_id: int, unread_only: bool, limit: int, offset: int
) -> tuple[list[Notification], int]:
    filters = [Notification.org_id == org_id, Notification.user_id == user_id]
    if unread_only:
        filters.append(Notification.is_read.is_(False))

    total = db.execute(select(func.count()).select_from(Notification).where(*filters)).scalar_one()
    statement = (
        select(Notification)
        .where(*filters)
        # Newest first
        .order_by(Notification.created_at.desc(), Notification.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.execute(statement).scalars().all()), total


def count_unread(db: Session, org_id: int, user_id: int) -> int:
    statement = (
        select(func.count())
        .select_from(Notification)
        .where(
            Notification.org_id == org_id,
            Notification.user_id == user_id,
            Notification.is_read.is_(False),
        )
    )
    return db.execute(statement).scalar_one()


def get_by_id(db: Session, org_id: int, user_id: int, notification_id: int) -> Notification | None:
    statement = select(Notification).where(
        Notification.id == notification_id,
        Notification.org_id == org_id,
        Notification.user_id == user_id,
    )
    return db.execute(statement).scalar_one_or_none()


def mark_all_read(db: Session, org_id: int, user_id: int) -> int:
    # One UPDATE; the result says how many rows changed
    statement = (
        update(Notification)
        .where(
            Notification.org_id == org_id,
            Notification.user_id == user_id,
            Notification.is_read.is_(False),
        )
        .values(is_read=True)
    )
    result = db.execute(statement)
    db.flush()
    return result.rowcount
