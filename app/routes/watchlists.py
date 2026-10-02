from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app.dependencies import get_current_user, get_db, require_editor
from app.models.users import User
from app.schemas.watchlists import (
    AddCompanyRequest,
    WatchlistCreate,
    WatchlistDetail,
    WatchlistSummary,
    WatchlistUpdate,
)
from app.services import watchlists as watchlist_service

router = APIRouter(prefix="/watchlists", tags=["watchlists"])


@router.post("", response_model=WatchlistSummary, status_code=201)
def create_watchlist(
    data: WatchlistCreate,
    current_user: User = Depends(require_editor),
    db: Session = Depends(get_db),
) -> WatchlistSummary:
    return watchlist_service.create_watchlist(db, current_user.org_id, current_user.id, data)


@router.get("", response_model=list[WatchlistSummary])
def list_watchlists(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> list[WatchlistSummary]:
    return watchlist_service.list_watchlists(db, current_user.org_id)


@router.get("/{watchlist_id}", response_model=WatchlistDetail)
def get_watchlist(
    watchlist_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WatchlistDetail:
    return watchlist_service.get_watchlist(db, current_user.org_id, watchlist_id)


@router.patch("/{watchlist_id}", response_model=WatchlistSummary)
def rename_watchlist(
    watchlist_id: int,
    data: WatchlistUpdate,
    current_user: User = Depends(require_editor),
    db: Session = Depends(get_db),
) -> WatchlistSummary:
    return watchlist_service.rename_watchlist(
        db, current_user.org_id, current_user.id, watchlist_id, data
    )


@router.delete("/{watchlist_id}", status_code=204)
def delete_watchlist(
    watchlist_id: int,
    current_user: User = Depends(require_editor),
    db: Session = Depends(get_db),
) -> Response:
    watchlist_service.delete_watchlist(db, current_user.org_id, current_user.id, watchlist_id)
    return Response(status_code=204)


@router.post("/{watchlist_id}/companies", response_model=WatchlistDetail, status_code=201)
def add_company(
    watchlist_id: int,
    data: AddCompanyRequest,
    current_user: User = Depends(require_editor),
    db: Session = Depends(get_db),
) -> WatchlistDetail:
    return watchlist_service.add_company(
        db, current_user.org_id, current_user.id, watchlist_id, data
    )


@router.delete("/{watchlist_id}/companies/{ticker}", status_code=204)
def remove_company(
    watchlist_id: int,
    ticker: str,
    current_user: User = Depends(require_editor),
    db: Session = Depends(get_db),
) -> Response:
    watchlist_service.remove_company(db, current_user.org_id, current_user.id, watchlist_id, ticker)
    return Response(status_code=204)
