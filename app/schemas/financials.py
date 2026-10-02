from datetime import date
from enum import StrEnum

from pydantic import BaseModel


class MetricName(StrEnum):
    revenue = "revenue"
    net_income = "net_income"
    operating_income = "operating_income"
    gross_profit = "gross_profit"
    total_assets = "total_assets"
    total_liabilities = "total_liabilities"
    shareholders_equity = "shareholders_equity"
    eps_diluted = "eps_diluted"
    operating_cash_flow = "operating_cash_flow"


class FinancialPoint(BaseModel):
    fiscal_year: int
    period_start: date | None
    period_end: date
    value: float
    # The us-gaap tag the value came from (a company can switch tags between years)
    concept: str
    accession_number: str
    filed_on: date


class MetricSeries(BaseModel):
    metric: MetricName
    label: str
    unit: str
    points: list[FinancialPoint]


class FinancialsResponse(BaseModel):
    ticker: str
    name: str
    metrics: list[MetricSeries]
