# Ingestion runs are system data, so no function in this file takes an org_id.
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.ingestion import IngestionRun, IngestionStatus


def create_run(db: Session, job_type: str) -> IngestionRun:
    run = IngestionRun(job_type=job_type, status=IngestionStatus.running)
    db.add(run)
    db.flush()
    return run


def get_recent_running_run(db: Session, job_type: str, since: datetime) -> IngestionRun | None:
    statement = (
        select(IngestionRun)
        .where(
            IngestionRun.job_type == job_type,
            IngestionRun.status == IngestionStatus.running,
            IngestionRun.started_at >= since,
        )
        .order_by(IngestionRun.started_at.desc())
        .limit(1)
    )
    return db.execute(statement).scalar_one_or_none()
