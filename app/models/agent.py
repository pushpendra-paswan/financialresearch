from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AgentRunStatus(StrEnum):
    running = "running"
    completed = "completed"
    failed = "failed"
    step_limit = "step_limit"
    timeout = "timeout"
    cancelled = "cancelled"
    # 3.3: a write tool paused the run until the user decides; expired = nobody decided in time
    waiting_approval = "waiting_approval"
    expired = "expired"


class ApprovalStatus(StrEnum):
    not_required = "not_required"
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    expired = "expired"


class AgentRun(Base):
    # Private AND personal (like chats): every query filters by org_id and user_id
    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'step_limit', 'timeout', 'cancelled', "
            "'waiting_approval', 'expired')",
            name="ck_agent_runs_status",
        ),
        Index("ix_agent_runs_org_id_user_id", "org_id", "user_id"),
        Index("ix_agent_runs_session_id", "session_id"),
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    # The USER message that started the run, and the assistant message written when it ends
    message_id: Mapped[int] = mapped_column(ForeignKey("chat_messages.id", ondelete="CASCADE"))
    answer_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="CASCADE")
    )
    status: Mapped[str] = mapped_column(String(20))
    # Model calls so far
    step_count: Mapped[int] = mapped_column(server_default="0")
    # The exception CLASS NAME only (an OpenAI message can contain part of the key)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ToolCall(Base):
    # No org_id: tool calls are only read through their org-and-user-scoped run (the same pattern
    # as chat_messages and watchlist_items)
    __tablename__ = "tool_calls"
    __table_args__ = (
        CheckConstraint(
            "approval_status IN ('not_required', 'pending', 'approved', 'rejected', 'expired')",
            name="ck_tool_calls_approval_status",
        ),
        Index("ix_tool_calls_run_id_id", "run_id", "id"),
        # A pending row is later UPDATED with the result, never duplicated
        UniqueConstraint("run_id", "tool_call_id", name="uq_tool_calls_run_id_tool_call_id"),
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"))
    # The number of the model call that requested this tool call, from 1
    step: Mapped[int]
    # LangChain's id of the call (matches the ToolMessage)
    tool_call_id: Mapped[str] = mapped_column(String(100))
    tool_name: Mapped[str] = mapped_column(String(100))
    input: Mapped[dict] = mapped_column(JSONB)
    # What the tool returned (at most the 16,000-character tool bound)
    output: Mapped[str] = mapped_column(Text)
    is_error: Mapped[bool] = mapped_column(server_default=false())
    approval_status: Mapped[str] = mapped_column(String(20), server_default="not_required")
    # Wall time of the whole tools node of that step; parallel calls share it
    duration_ms: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
