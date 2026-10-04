from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Report(Base):
    # Private to the ORGANIZATION (shared by the team, like watchlists), not personal. Created only
    # by the agent after the user approved the text. The report outlives the chat that made it:
    # agent_run_id becomes null when the run is deleted
    __tablename__ = "reports"
    __table_args__ = (Index("ix_reports_org_id_created_at", "org_id", "created_at"),)

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    # The creator
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    agent_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="SET NULL")
    )
    title: Mapped[str] = mapped_column(String(200))
    # Restricted Markdown with the citation markers [1]..[k]
    content: Mapped[str] = mapped_column(Text)
    # Snapshot of the successful non-search tool calls of the run: [{"tool", "args"}]. The run is
    # private to its owner, the report is shared, so teammates see which data was used
    data_sources: Mapped[list] = mapped_column(
        JSONB, default=list, server_default=text("'[]'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReportCompany(Base):
    # No org_id: only read through a report loaded with org_id (the watchlist_items pattern).
    # The company FK has no cascade: the catalog is never touched
    __tablename__ = "report_companies"

    report_id: Mapped[int] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE"), primary_key=True
    )
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), primary_key=True)


class ReportCitation(Base):
    # What a report cited. The 2.4 citations table is bound to chat messages, so reports have their
    # own. Same rule: chunk_id is nullable with ON DELETE SET NULL because --reembed replaces chunk
    # rows, and the snapshot columns keep the exact passage that was cited
    __tablename__ = "report_citations"
    __table_args__ = (
        UniqueConstraint("report_id", "number", name="uq_report_citations_report_number"),
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("reports.id", ondelete="CASCADE"))
    # The [n] in the report text
    number: Mapped[int]
    chunk_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="SET NULL")
    )
    # The best vector similarity the run's searches gave this chunk
    score: Mapped[float] = mapped_column(Float)
    # Snapshot of the cited chunk. filing_id is a plain number, not a foreign key
    filing_id: Mapped[int]
    ticker: Mapped[str] = mapped_column(String(15))
    fiscal_year: Mapped[int | None]
    section: Mapped[str] = mapped_column(String(30))
    content: Mapped[str] = mapped_column(Text)
