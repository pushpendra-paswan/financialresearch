from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Numeric,
    String,
    func,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.companies import Company


class AlertType(StrEnum):
    price_above = "price_above"
    price_below = "price_below"
    daily_change_pct = "daily_change_pct"


class Alert(Base):
    # Private AND personal: besides org_id, every query also filters by the owning user_id
    __tablename__ = "alerts"
    # The type is a plain string with a CHECK (same pattern as the user roles)
    __table_args__ = (
        CheckConstraint(
            "alert_type IN ('price_above', 'price_below', 'daily_change_pct')",
            name="ck_alerts_alert_type",
        ),
        CheckConstraint("threshold > 0", name="ck_alerts_threshold_positive"),
        Index("ix_alerts_org_id_user_id", "org_id", "user_id"),
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))
    alert_type: Mapped[str] = mapped_column(String(20))
    # A price level for price_above / price_below, a percent for daily_change_pct
    threshold: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    active: Mapped[bool] = mapped_column(Boolean, server_default=true())
    # The first day the alert may fire: bars before this date are never considered
    watch_from: Mapped[date] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Joined in the same query, so the ticker is always available without an extra query per row
    company: Mapped[Company] = relationship(lazy="joined")

    @property
    def ticker(self) -> str:
        return self.company.ticker

    @property
    def company_name(self) -> str:
        return self.company.name
