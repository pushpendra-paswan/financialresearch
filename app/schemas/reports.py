from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ReportSummary(BaseModel):
    id: int
    title: str
    tickers: list[str]
    # The creator's email
    created_by: str
    created_at: datetime


class ReportListResponse(BaseModel):
    items: list[ReportSummary]
    total: int
    page: int
    page_size: int


class ReportCompanyResponse(BaseModel):
    ticker: str
    name: str


class ReportCitationResponse(BaseModel):
    # The stored snapshot of what the report cited. chunk_id is null when the chunk was replaced
    # by a re-embed. org_id is not part of a citation row
    model_config = ConfigDict(from_attributes=True)

    number: int
    chunk_id: int | None
    ticker: str
    fiscal_year: int | None
    section: str
    score: float
    content: str


class ReportDetail(BaseModel):
    id: int
    title: str
    content: str
    companies: list[ReportCompanyResponse]
    citations: list[ReportCitationResponse]
    # [{"tool": ..., "args": {...}}]: the data tool calls of the run that wrote the report
    data_sources: list[dict]
    created_by: str
    agent_run_id: int | None
    created_at: datetime
