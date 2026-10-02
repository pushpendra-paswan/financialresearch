from sqlalchemy.orm import Session

from app.exceptions import ConflictError, NotFoundError
from app.repositories import audit as audit_repository
from app.repositories import companies as company_repository
from app.repositories import watchlists as watchlist_repository
from app.schemas.companies import CompanyResponse
from app.schemas.watchlists import (
    AddCompanyRequest,
    WatchlistCreate,
    WatchlistDetail,
    WatchlistSummary,
    WatchlistUpdate,
)

# The audit entity_id is always the watchlist id, so an audit row does not say which company
# was added or removed.


def create_watchlist(
    db: Session, org_id: int, user_id: int, data: WatchlistCreate
) -> WatchlistSummary:
    # Names are unique within an organization, ignoring letter case
    existing = watchlist_repository.get_by_name(db, org_id, data.name)
    if existing:
        raise ConflictError("A watchlist with this name already exists")

    watchlist = watchlist_repository.create(db, org_id, user_id, data.name)

    audit_repository.create(db, org_id, user_id, action="watchlist.create", entity_id=watchlist.id)
    db.commit()

    # A new watchlist has no companies yet
    return WatchlistSummary(
        id=watchlist.id,
        name=watchlist.name,
        created_by=watchlist.created_by,
        created_at=watchlist.created_at,
        item_count=0,
    )


def list_watchlists(db: Session, org_id: int) -> list[WatchlistSummary]:
    rows = watchlist_repository.list_with_counts(db, org_id)
    return [
        WatchlistSummary(
            id=watchlist.id,
            name=watchlist.name,
            created_by=watchlist.created_by,
            created_at=watchlist.created_at,
            item_count=item_count,
        )
        for watchlist, item_count in rows
    ]


def get_watchlist(db: Session, org_id: int, watchlist_id: int) -> WatchlistDetail:
    watchlist = watchlist_repository.get_by_id(db, org_id, watchlist_id)

    # Same message whether the watchlist does not exist or belongs to another organization,
    # so the existence of other organizations' watchlists is not leaked
    if watchlist is None:
        raise NotFoundError("Watchlist not found")

    companies = watchlist_repository.list_companies(db, org_id, watchlist.id)
    return WatchlistDetail(
        id=watchlist.id,
        name=watchlist.name,
        created_by=watchlist.created_by,
        created_at=watchlist.created_at,
        companies=[CompanyResponse.model_validate(company) for company in companies],
    )


def rename_watchlist(
    db: Session, org_id: int, user_id: int, watchlist_id: int, data: WatchlistUpdate
) -> WatchlistSummary:
    watchlist = watchlist_repository.get_by_id(db, org_id, watchlist_id)
    if watchlist is None:
        raise NotFoundError("Watchlist not found")

    # Only ANOTHER watchlist holding the name is a conflict. Renaming to the same name, or
    # changing only the letter case of its own name ("tech" -> "Tech"), is allowed.
    existing = watchlist_repository.get_by_name(db, org_id, data.name)
    if existing and existing.id != watchlist.id:
        raise ConflictError("A watchlist with this name already exists")

    watchlist.name = data.name

    audit_repository.create(db, org_id, user_id, action="watchlist.rename", entity_id=watchlist.id)
    db.commit()

    # The item count is not changed by a rename, but the summary needs it
    companies = watchlist_repository.list_companies(db, org_id, watchlist.id)
    return WatchlistSummary(
        id=watchlist.id,
        name=watchlist.name,
        created_by=watchlist.created_by,
        created_at=watchlist.created_at,
        item_count=len(companies),
    )


def delete_watchlist(db: Session, org_id: int, user_id: int, watchlist_id: int) -> None:
    watchlist = watchlist_repository.get_by_id(db, org_id, watchlist_id)
    if watchlist is None:
        raise NotFoundError("Watchlist not found")

    # The database cascade removes the items; the companies stay in the catalog
    watchlist_repository.delete_watchlist(db, watchlist)

    audit_repository.create(db, org_id, user_id, action="watchlist.delete", entity_id=watchlist_id)
    db.commit()


def add_company(
    db: Session, org_id: int, user_id: int, watchlist_id: int, data: AddCompanyRequest
) -> WatchlistDetail:
    # Check the watchlist first, so another organization's watchlist never reveals anything
    # about companies
    watchlist = watchlist_repository.get_by_id(db, org_id, watchlist_id)
    if watchlist is None:
        raise NotFoundError("Watchlist not found")

    # Tickers are stored in uppercase, and the request may use any case
    company = company_repository.get_by_ticker(db, data.ticker.upper())
    if company is None:
        raise NotFoundError("Company not found")

    existing_item = watchlist_repository.get_item(db, org_id, watchlist.id, company.id)
    if existing_item:
        raise ConflictError("Company is already in this watchlist")

    watchlist_repository.add_item(db, watchlist.id, company.id)

    audit_repository.create(
        db, org_id, user_id, action="watchlist.add_company", entity_id=watchlist.id
    )
    db.commit()

    # Return the full updated watchlist
    companies = watchlist_repository.list_companies(db, org_id, watchlist.id)
    return WatchlistDetail(
        id=watchlist.id,
        name=watchlist.name,
        created_by=watchlist.created_by,
        created_at=watchlist.created_at,
        companies=[CompanyResponse.model_validate(company) for company in companies],
    )


def remove_company(db: Session, org_id: int, user_id: int, watchlist_id: int, ticker: str) -> None:
    watchlist = watchlist_repository.get_by_id(db, org_id, watchlist_id)
    if watchlist is None:
        raise NotFoundError("Watchlist not found")

    company = company_repository.get_by_ticker(db, ticker.strip().upper())
    if company is None:
        raise NotFoundError("Company not found")

    item = watchlist_repository.get_item(db, org_id, watchlist.id, company.id)
    if item is None:
        raise NotFoundError("Company is not in this watchlist")

    watchlist_repository.remove_item(db, item)

    audit_repository.create(
        db, org_id, user_id, action="watchlist.remove_company", entity_id=watchlist.id
    )
    db.commit()
