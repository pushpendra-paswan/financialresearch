from sqlalchemy.orm import Session

from app.models.audit import AuditLog


def create(
    db: Session, org_id: int, user_id: int, action: str, entity_id: int | None = None
) -> AuditLog:
    audit_log = AuditLog(org_id=org_id, user_id=user_id, action=action, entity_id=entity_id)
    db.add(audit_log)
    db.flush()
    return audit_log
