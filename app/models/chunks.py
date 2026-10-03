from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

# The size of every embedding vector. It is a constant, not a setting: changing it needs a
# migration (a new column type), and every stored vector would have to be re-embedded
EMBEDDING_DIMENSIONS = 1536


class DocumentChunk(Base):
    # Shared public data (derived from public filings): there is no org_id.
    # There is no vector (HNSW) index on purpose: with a few hundred rows an exact scan is fast
    # and has perfect recall. Add one if the table grows to many thousands of rows
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    filing_id: Mapped[int] = mapped_column(ForeignKey("filings.id"))
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))
    # A key of SECTIONS in app/rag/parsing.py, e.g. "risk_factors"
    section: Mapped[str] = mapped_column(String(30))
    fiscal_year: Mapped[int | None]
    # Position of the chunk inside its (filing, section), counted from 0
    chunk_index: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    # Filled by Postgres itself from content, so it can never get out of sync
    search_vector = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )
    # Which model made the vector, so mixed models are visible after a config change
    embedding_model: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # The unique constraint also serves lookups by filing_id (its first column), so there is no
    # separate filing_id index
    __table_args__ = (
        UniqueConstraint("filing_id", "section", "chunk_index", name="uq_document_chunks_chunk"),
        Index("ix_document_chunks_search_vector", "search_vector", postgresql_using="gin"),
    )
