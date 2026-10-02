from datetime import datetime
from enum import StrEnum

from sqlalchemy import CheckConstraint, DateTime, Identity, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class IngestionStatus(StrEnum):
    running = "running"
    success = "success"
    partial = "partial"
    failed = "failed"


class IngestionRun(Base):
    # One row per run of a scheduled job. Shared system data: there is no org_id
    __tablename__ = "ingestion_runs"
    # Plain string plus CHECK, the same pattern as the user roles
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'success', 'partial', 'failed')", name="ck_ingestion_runs_status"
        ),
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    job_type: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Human-readable summary, e.g. the counts of what was stored
    message: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
