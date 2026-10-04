from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from app.config import settings


class SessionResponse(BaseModel):
    # org_id and user_id are deliberately not included
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str | None
    created_at: datetime
    updated_at: datetime


class CitationResponse(BaseModel):
    # The snapshot stored when the answer was saved, so it stays readable after a re-embed
    # (chunk_id is then null)
    model_config = ConfigDict(from_attributes=True)

    number: int
    chunk_id: int | None
    ticker: str
    fiscal_year: int | None
    section: str
    score: float
    content: str


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
    # Empty for user messages and for answers without citations
    citations: list[CitationResponse]
    # The agent run that wrote this answer (3.2); null for user messages and RAG answers
    run_id: int | None = None


class SessionDetailResponse(SessionResponse):
    messages: list[MessageResponse]


class MessageCreate(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
    ]
    # Optional filter; null means all of RAG_TICKERS
    ticker: str | None = None
    # How the question is answered (3.2): "rag" is the 2.4 chat (the API default, so older clients
    # are unchanged), "agent" always runs the research agent, "auto" lets a router choose
    mode: Literal["rag", "agent", "auto"] = "rag"

    @field_validator("ticker")
    @classmethod
    def ticker_in_scope(cls, value: str | None) -> str | None:
        if value is None:
            return None
        allowed = [t.strip().upper() for t in settings.RAG_TICKERS.split(",") if t.strip()]
        ticker = value.strip().upper()
        if ticker not in allowed:
            raise ValueError(f"must be one of {', '.join(allowed)}")
        return ticker
