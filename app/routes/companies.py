from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.dependencies import get_current_user, get_db
from app.models.companies import Company
from app.models.users import User
from app.schemas.companies import CompanyListResponse, CompanyResponse
from app.services import companies as company_service

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get("", response_model=CompanyListResponse)
def list_companies(
    search: str | None = Query(default=None, max_length=50),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CompanyListResponse:
    return company_service.list_companies(db, search, page, page_size)


@router.get("/{ticker}", response_model=CompanyResponse)
def get_company(
    ticker: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Company:
    return company_service.get_company(db, ticker)
