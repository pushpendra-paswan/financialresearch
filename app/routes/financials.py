from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.dependencies import get_current_user, get_db
from app.models.users import User
from app.schemas.financials import FinancialsResponse, MetricName
from app.services import financials as financial_service

router = APIRouter(prefix="/companies", tags=["financials"])


@router.get("/{ticker}/financials", response_model=FinancialsResponse)
def get_financials(
    ticker: str,
    metric: MetricName | None = None,
    years: int = Query(default=5, ge=1, le=10),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FinancialsResponse:
    return financial_service.get_financials(db, ticker, metric, years)
