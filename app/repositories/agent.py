# Agent runs are private and personal (like chat sessions): the run reads take org_id AND user_id.
# Tool-call reads take a run_id that the service has ALREADY loaded through get_run, and the
# answer-message lookup takes message ids of a session the service already loaded through the
# scoped chat query (the same pattern as the chat message queries).
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models.agent import AgentRun, AgentRunStatus, ApprovalStatus, ToolCall


def create_run(
    db: Session, org_id: int, user_id: int, session_id: int, message_id: int
) -> AgentRun:
    run = AgentRun(
        org_id=org_id,
        user_id=user_id,
        session_id=session_id,
        message_id=message_id,
        status=AgentRunStatus.running,
        step_count=0,
    )
    db.add(run)
    db.flush()
    return run


def get_run(db: Session, org_id: int, user_id: int, run_id: int) -> AgentRun | None:
    statement = select(AgentRun).where(
        AgentRun.id == run_id, AgentRun.org_id == org_id, AgentRun.user_id == user_id
    )
    return db.execute(statement).scalar_one_or_none()


def get_active_run(db: Session, org_id: int, user_id: int, since: datetime) -> AgentRun | None:
    # A run still marked running that started at or after `since`. Older running rows are stale
    # (a crashed process never finished them) and are ignored
    statement = (
        select(AgentRun)
        .where(
            AgentRun.org_id == org_id,
            AgentRun.user_id == user_id,
            AgentRun.status == AgentRunStatus.running,
            AgentRun.started_at >= since,
        )
        .limit(1)
    )
    return db.execute(statement).scalar_one_or_none()


def update_run(db: Session, run: AgentRun, **changes: object) -> None:
    # Sets the given columns, for example update_run(db, run, status="failed", error="ValueError")
    for column, value in changes.items():
        setattr(run, column, value)
    db.flush()


def change_run_status(
    db: Session,
    org_id: int,
    user_id: int,
    run_id: int,
    from_status: str,
    to_status: str,
    **changes: object,
) -> bool:
    # An atomic compare-and-set: the row changes only if it still has `from_status`, so of two
    # simultaneous decisions exactly one gets True. The owner filter is part of the statement
    statement = (
        update(AgentRun)
        .where(
            AgentRun.id == run_id,
            AgentRun.org_id == org_id,
            AgentRun.user_id == user_id,
            AgentRun.status == from_status,
        )
        .values(status=to_status, **changes)
    )
    return db.execute(statement).rowcount == 1


def change_tool_call_approval(
    db: Session, run_id: int, row_id: int, from_status: str, to_status: str
) -> bool:
    # The same compare-and-set for the approval status of one tool call row of a run the service
    # already loaded through get_run
    statement = (
        update(ToolCall)
        .where(
            ToolCall.id == row_id,
            ToolCall.run_id == run_id,
            ToolCall.approval_status == from_status,
        )
        .values(approval_status=to_status)
    )
    return db.execute(statement).rowcount == 1


def get_tool_call(db: Session, run_id: int, row_id: int) -> ToolCall | None:
    statement = select(ToolCall).where(ToolCall.id == row_id, ToolCall.run_id == run_id)
    return db.execute(statement).scalar_one_or_none()


def get_pending_tool_call(db: Session, run_id: int) -> ToolCall | None:
    # A run has at most one pending call (one write action per approval)
    statement = select(ToolCall).where(
        ToolCall.run_id == run_id, ToolCall.approval_status == ApprovalStatus.pending
    )
    return db.execute(statement).scalars().first()


def list_tool_calls_by_call_ids(db: Session, run_id: int, call_ids: list[str]) -> list[ToolCall]:
    # The rows that already exist for these LangChain tool call ids (the pending one)
    statement = select(ToolCall).where(
        ToolCall.run_id == run_id, ToolCall.tool_call_id.in_(call_ids)
    )
    return list(db.execute(statement).scalars().all())


def save_tool_calls(db: Session, run_id: int, rows: list[ToolCall]) -> None:
    # Inserts the rows. A row whose tool_call_id already exists for this run (the pending write
    # call of a paused run) is UPDATED with the result instead: the unique constraint on
    # (run_id, tool_call_id) forbids a second row, and its approval status is left as it is
    existing = {
        row.tool_call_id: row
        for row in list_tool_calls_by_call_ids(db, run_id, [row.tool_call_id for row in rows])
    }
    for row in rows:
        stored = existing.get(row.tool_call_id)
        if stored is None:
            db.add(row)
        else:
            stored.output = row.output
            stored.is_error = row.is_error
            stored.duration_ms = row.duration_ms
    db.flush()


def list_tool_calls(db: Session, run_id: int) -> list[ToolCall]:
    # Oldest first
    statement = select(ToolCall).where(ToolCall.run_id == run_id).order_by(ToolCall.id)
    return list(db.execute(statement).scalars().all())


def list_run_ids_by_answer_messages(db: Session, message_ids: list[int]) -> dict[int, int]:
    # {answer message id: run id} for the assistant messages that an agent run wrote
    if not message_ids:
        return {}
    statement = select(AgentRun.answer_message_id, AgentRun.id).where(
        AgentRun.answer_message_id.in_(message_ids)
    )
    return {message_id: run_id for message_id, run_id in db.execute(statement).all()}
