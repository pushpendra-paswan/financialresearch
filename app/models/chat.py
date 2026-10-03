from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ChatSession(Base):
    # Private AND personal (like alerts): every query filters by org_id and user_id
    __tablename__ = "chat_sessions"
    __table_args__ = (Index("ix_chat_sessions_org_id_user_id", "org_id", "user_id"),)

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    # The first question, cut to 100 characters; null until the first answer is saved
    title: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChatMessage(Base):
    # No org_id: messages are only read through their org-and-user-scoped session (the same
    # pattern as watchlist_items). Deleting a session deletes its messages in the database
    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="ck_chat_messages_role"),
        Index("ix_chat_messages_session_id_id", "session_id", "id"),
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(10))
    content: Mapped[str] = mapped_column(Text)
    # User rows only: the standalone question used for retrieval (null when there was no history)
    # and the ticker filter that was chosen (null means all)
    rewritten_question: Mapped[str | None] = mapped_column(Text)
    ticker: Mapped[str | None] = mapped_column(String(15))
    # Assistant rows only: the chat model that wrote the answer (null for the fixed "I don't know")
    model: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Citation(Base):
    # What an answer cited. chunk_id is nullable with ON DELETE SET NULL because --reembed
    # replaces chunk rows; the snapshot columns keep the exact passage that was cited
    __tablename__ = "citations"
    __table_args__ = (UniqueConstraint("message_id", "number", name="uq_citations_message_number"),)

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("chat_messages.id", ondelete="CASCADE"))
    # The [n] in the answer text
    number: Mapped[int]
    chunk_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="SET NULL")
    )
    # The vector similarity of the chunk to the question
    score: Mapped[float] = mapped_column(Float)
    # Snapshot of the cited chunk. filing_id is a plain number, not a foreign key
    filing_id: Mapped[int]
    ticker: Mapped[str] = mapped_column(String(15))
    fiscal_year: Mapped[int | None]
    section: Mapped[str] = mapped_column(String(30))
    content: Mapped[str] = mapped_column(Text)
