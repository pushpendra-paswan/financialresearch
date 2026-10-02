# Watchlists are private data: every function that reads them takes org_id and filters by it.
# The companies catalog is shared and is not touched here (use repositories.companies).
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.companies import Company
from app.models.watchlists import Watchlist, WatchlistItem


def create(db: Session, org_id: int, user_id: int, name: str) -> Watchlist:
    watchlist = Watchlist(org_id=org_id, created_by=user_id, name=name)
    db.add(watchlist)
    db.flush()
    return watchlist


def get_by_id(db: Session, org_id: int, watchlist_id: int) -> Watchlist | None:
    statement = select(Watchlist).where(Watchlist.org_id == org_id, Watchlist.id == watchlist_id)
    return db.execute(statement).scalar_one_or_none()


def get_by_name(db: Session, org_id: int, name: str) -> Watchlist | None:
    # Case-insensitive, to match the unique index on (org_id, lower(name))
    statement = select(Watchlist).where(
        Watchlist.org_id == org_id, func.lower(Watchlist.name) == name.lower()
    )
    return db.execute(statement).scalar_one_or_none()


def list_with_counts(db: Session, org_id: int) -> list[tuple[Watchlist, int]]:
    # One query for all watchlists and their item counts (outer join, so empty ones count 0)
    statement = (
        select(Watchlist, func.count(WatchlistItem.company_id))
        .outerjoin(WatchlistItem, WatchlistItem.watchlist_id == Watchlist.id)
        .where(Watchlist.org_id == org_id)
        .group_by(Watchlist.id)
        .order_by(func.lower(Watchlist.name))
    )
    return [(watchlist, count) for watchlist, count in db.execute(statement).all()]


def list_companies(db: Session, org_id: int, watchlist_id: int) -> list[Company]:
    # The join goes through watchlists so the org_id filter applies to the items as well
    statement = (
        select(Company)
        .join(WatchlistItem, WatchlistItem.company_id == Company.id)
        .join(Watchlist, Watchlist.id == WatchlistItem.watchlist_id)
        .where(Watchlist.org_id == org_id, Watchlist.id == watchlist_id)
        .order_by(Company.ticker)
    )
    return list(db.execute(statement).scalars().all())


def get_item(db: Session, org_id: int, watchlist_id: int, company_id: int) -> WatchlistItem | None:
    statement = (
        select(WatchlistItem)
        .join(Watchlist, Watchlist.id == WatchlistItem.watchlist_id)
        .where(
            Watchlist.org_id == org_id,
            Watchlist.id == watchlist_id,
            WatchlistItem.company_id == company_id,
        )
    )
    return db.execute(statement).scalar_one_or_none()


# --- Writes ---
# The functions below do not check org_id themselves. The service must already have loaded the
# watchlist with get_by_id(org_id) before calling them; the tests prove that another
# organization's watchlist is never changed.


def delete_watchlist(db: Session, watchlist: Watchlist) -> None:
    # The database cascade (ON DELETE CASCADE) removes the items; companies are not touched
    db.delete(watchlist)
    db.flush()


def add_item(db: Session, watchlist_id: int, company_id: int) -> WatchlistItem:
    item = WatchlistItem(watchlist_id=watchlist_id, company_id=company_id)
    db.add(item)
    db.flush()
    return item


def remove_item(db: Session, item: WatchlistItem) -> None:
    db.delete(item)
    db.flush()
