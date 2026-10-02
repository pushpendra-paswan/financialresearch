# Prices are shared public data, so no function in this file takes an org_id.
from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.prices import PriceBar


def get_by_company(db: Session, company_id: int) -> list[PriceBar]:
    statement = select(PriceBar).where(PriceBar.company_id == company_id)
    return list(db.execute(statement).scalars().all())


def create(
    db: Session,
    company_id: int,
    trade_date: date,
    open: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    adj_close: Decimal,
    volume: int,
) -> PriceBar:
    bar = PriceBar(
        company_id=company_id,
        trade_date=trade_date,
        open=open,
        high=high,
        low=low,
        close=close,
        adj_close=adj_close,
        volume=volume,
    )
    db.add(bar)
    db.flush()
    return bar


def list_since(db: Session, company_id: int, since: date) -> list[PriceBar]:
    # Oldest first
    statement = (
        select(PriceBar)
        .where(PriceBar.company_id == company_id, PriceBar.trade_date >= since)
        .order_by(PriceBar.trade_date)
    )
    return list(db.execute(statement).scalars().all())
