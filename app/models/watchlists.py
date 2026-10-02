from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Identity, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Watchlist(Base):
    # Private data: every watchlist belongs to one organization
    __tablename__ = "watchlists"

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), index=True)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# Names are unique within an organization, ignoring letter case ("Tech" and "tech" clash).
# Two organizations can use the same name. This is declared after the class because it uses the
# mapped columns inside lower().
Index("uq_watchlists_org_id_lower_name", Watchlist.org_id, func.lower(Watchlist.name), unique=True)


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"

    # The primary key is the pair, so a company can be in a watchlist only once and lookups by
    # watchlist are covered without another index. Deleting a watchlist deletes its items
    # (ON DELETE CASCADE). The company FK has no cascade: the catalog is never touched.
    watchlist_id: Mapped[int] = mapped_column(
        ForeignKey("watchlists.id", ondelete="CASCADE"), primary_key=True
    )
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), primary_key=True)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
