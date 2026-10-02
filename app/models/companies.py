from datetime import datetime

from sqlalchemy import DateTime, Identity, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Company(Base):
    # Shared public data: there is no org_id, every organization reads the same rows
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), unique=True)
    # 10-character zero-padded string, e.g. "0000320193". This is the format the
    # data.sec.gov endpoints need (milestones 1.4 and 1.5)
    cik: Mapped[str] = mapped_column(String(10), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    exchange: Mapped[str | None] = mapped_column(String(20))
    # The SEC's SIC industry description, e.g. "Electronic Computers". Filled by the filing
    # ingestion. The SEC does not provide GICS sectors
    industry: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
