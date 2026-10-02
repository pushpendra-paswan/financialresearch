from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, StringConstraints

from app.schemas.companies import CompanyResponse

# Whitespace is stripped first, then the length is checked, so "   " is rejected as empty
WatchlistName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]


class WatchlistCreate(BaseModel):
    name: WatchlistName


class WatchlistUpdate(BaseModel):
    name: WatchlistName


class AddCompanyRequest(BaseModel):
    ticker: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=15)]


class WatchlistSummary(BaseModel):
    id: int
    name: str
    created_by: int
    created_at: datetime
    item_count: int


class WatchlistDetail(BaseModel):
    id: int
    name: str
    created_by: int
    created_at: datetime
    companies: list[CompanyResponse]
