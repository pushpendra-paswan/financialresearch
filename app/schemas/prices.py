from datetime import date

from pydantic import BaseModel


class PriceBarResponse(BaseModel):
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int


class PriceHistoryResponse(BaseModel):
    ticker: str
    name: str
    bars: list[PriceBarResponse]
