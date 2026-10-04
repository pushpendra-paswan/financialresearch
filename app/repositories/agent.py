# Agent runs are private and personal (like chat sessions): the run reads take org_id AND user_id.
# Tool-call reads take a run_id that the service has ALREADY loaded through get_run, and the
# answer-message lookup takes message ids of a session the service already loaded through the
# scoped chat query (the same pattern as the chat message queries).
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.agent import AgentRun, AgentRunStatus, ToolCall


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


def create_tool_calls(db: Session, rows: list[ToolCall]) -> None:
    db.add_all(rows)
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
