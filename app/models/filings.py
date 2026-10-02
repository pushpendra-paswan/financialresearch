from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Identity, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Filing(Base):
    # Shared public data: there is no org_id, every organization reads the same rows
    __tablename__ = "filings"
    __table_args__ = (Index("ix_filings_company_id_filed_on", "company_id", "filed_on"),)

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))
    # The dashed form, e.g. "0000320193-24-000123". Unique, so a filing is never stored twice
    accession_number: Mapped[str] = mapped_column(String(20), unique=True)
    form_type: Mapped[str] = mapped_column(String(10))
    filed_on: Mapped[date] = mapped_column(Date)
    # The period the filing covers. The SEC leaves it empty for some filings
    report_date: Mapped[date | None] = mapped_column(Date)
    # The year of report_date, so null when report_date is null
    fiscal_year: Mapped[int | None]
    primary_document: Mapped[str] = mapped_column(String(255))
    # Path of the downloaded file RELATIVE to RAW_DATA_DIR. Null until it is downloaded
    raw_path: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
