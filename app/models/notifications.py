from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Numeric,
    Text,
    UniqueConstraint,
    false,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.companies import Company


class Notification(Base):
    # Private and personal: it belongs to the owner of the alert (org_id and user_id are copied
    # from the alert when the evaluation job creates it)
    __tablename__ = "notifications"
    # At most one notification per alert per trading day
    __table_args__ = (
        UniqueConstraint("alert_id", "trade_date", name="uq_notifications_alert_id_trade_date"),
        Index("ix_notifications_org_id_user_id", "org_id", "user_id"),
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    # Deleting an alert deletes its notifications (done by the database)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alerts.id", ondelete="CASCADE"))
    # A snapshot: an alert can never change its company
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))
    trade_date: Mapped[date] = mapped_column(Date)
    # The close for price alerts, the SIGNED percent change for daily_change_pct
    trigger_value: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    message: Mapped[str] = mapped_column(Text)
    is_read: Mapped[bool] = mapped_column(Boolean, server_default=false())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Joined in the same query, so the ticker is always available without an extra query per row
    company: Mapped[Company] = relationship(lazy="joined")

    @property
    def ticker(self) -> str:
        return self.company.ticker
