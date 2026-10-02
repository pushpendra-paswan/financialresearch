from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.dependencies import get_current_user, get_db
from app.models.users import User
from app.schemas.filings import FilingListResponse
from app.services import filings as filing_service

router = APIRouter(prefix="/companies", tags=["filings"])


@router.get("/{ticker}/filings", response_model=FilingListResponse)
def list_filings(
    ticker: str,
    form_type: str | None = Query(default=None, pattern="^(10-K|10-Q)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FilingListResponse:
    return filing_service.list_filings(db, ticker, form_type, page, page_size)
