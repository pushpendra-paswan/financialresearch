# Alerts are private and personal: every function that reads them takes org_id AND user_id,
# except list_active (see the comment there).
from datetime import date
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.alerts import Alert


def get_by_id(db: Session, org_id: int, user_id: int, alert_id: int) -> Alert | None:
    statement = select(Alert).where(
        Alert.id == alert_id, Alert.org_id == org_id, Alert.user_id == user_id
    )
    return db.execute(statement).scalar_one_or_none()


def list_by_user(db: Session, org_id: int, user_id: int, active: bool | None) -> list[Alert]:
    statement = select(Alert).where(Alert.org_id == org_id, Alert.user_id == user_id)
    if active is not None:
        statement = statement.where(Alert.active == active)
    # Newest first
    statement = statement.order_by(Alert.created_at.desc(), Alert.id.desc())
    return list(db.execute(statement).scalars().all())


def count_by_user(db: Session, org_id: int, user_id: int) -> int:
    statement = (
        select(func.count())
        .select_from(Alert)
        .where(Alert.org_id == org_id, Alert.user_id == user_id)
    )
    return db.execute(statement).scalar_one()


def get_duplicate(
    db: Session,
    org_id: int,
    user_id: int,
    company_id: int,
    alert_type: str,
    threshold: Decimal,
) -> Alert | None:
    statement = (
        select(Alert)
        .where(
            Alert.org_id == org_id,
            Alert.user_id == user_id,
            Alert.company_id == company_id,
            Alert.alert_type == alert_type,
            Alert.threshold == threshold,
        )
        .limit(1)
    )
    return db.execute(statement).scalar_one_or_none()


def create(
    db: Session,
    org_id: int,
    user_id: int,
    company_id: int,
    alert_type: str,
    threshold: Decimal,
    watch_from: date,
    active: bool = True,
) -> Alert:
    alert = Alert(
        org_id=org_id,
        user_id=user_id,
        company_id=company_id,
        alert_type=alert_type,
        threshold=threshold,
        watch_from=watch_from,
        active=active,
    )
    db.add(alert)
    db.flush()
    return alert


def delete(db: Session, alert: Alert) -> None:
    # The caller loaded the alert with get_by_id (org_id and user_id), so ownership is checked.
    # The database cascade removes the alert's notifications
    db.delete(alert)
    db.flush()


def list_active(db: Session) -> list[Alert]:
    # The third documented exception to the org_id rule (after the two user lookups in
    # repositories/users.py): the evaluation job is a system job that checks every active alert
    # of every organization. It creates each notification with the alert's own org_id and user_id.
    statement = select(Alert).where(Alert.active.is_(True)).order_by(Alert.company_id, Alert.id)
    return list(db.execute(statement).scalars().all())
