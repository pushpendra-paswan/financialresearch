from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.dependencies import get_current_user, get_db
from app.models.users import User
from app.schemas.prices import PriceHistoryResponse
from app.services import prices as price_service

router = APIRouter(prefix="/companies", tags=["prices"])


@router.get("/{ticker}/prices", response_model=PriceHistoryResponse)
def get_prices(
    ticker: str,
    days: int = Query(default=365, ge=1, le=1825),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PriceHistoryResponse:
    return price_service.get_prices(db, ticker, days)
