from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, ForeignKey, Identity, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class FinancialFact(Base):
    # Shared public data: there is no org_id, every organization reads the same rows
    __tablename__ = "financial_facts"
    # One row per period. Plain Postgres treats two NULL period_start values as different, which
    # would allow duplicate balance-sheet rows, so NULLS NOT DISTINCT is needed (Postgres 15+).
    # This constraint is also the lookup index (company_id comes first), so there is no other index
    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "concept",
            "unit",
            "period_start",
            "period_end",
            name="uq_financial_facts_period",
            postgresql_nulls_not_distinct=True,
        ),
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))
    # The us-gaap tag, e.g. "Revenues"
    concept: Mapped[str] = mapped_column(String(200))
    # "USD" or "USD/shares"
    unit: Mapped[str] = mapped_column(String(20))
    # Null for point-in-time values (balance sheet)
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    value: Mapped[Decimal] = mapped_column(Numeric(28, 6))
    # The year of period_end. The XBRL fy/fp fields describe the filing, not the period
    fiscal_year: Mapped[int]
    # Always "10-K" for now
    form_type: Mapped[str] = mapped_column(String(10))
    # The filing the stored value came from. Not a foreign key to filings
    accession_number: Mapped[str] = mapped_column(String(20))
    filed_on: Mapped[date] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
