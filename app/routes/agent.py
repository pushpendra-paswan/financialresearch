from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent import run as agent_run
from app.dependencies import get_current_user, get_db
from app.models.users import User
from app.routes.chat import stream_lines
from app.schemas.agent import AgentRunResponse, DecisionRequest

# Agent runs are personal (like chats), so every role can read their own: get_current_user
router = APIRouter(prefix="/agent", tags=["agent"])


@router.get(
    "/runs/{run_id}",
    response_model=AgentRunResponse,
    description=(
        "One of the caller's own agent runs with its trace: the status, the number of model "
        "calls and every tool call (step, tool, input, output, error flag, approval status, "
        "duration). While the run waits for the user, pending_approval holds the action to approve "
        "(with its summary and expiry). 404 for a missing run, a colleague's run and another "
        "organization's run (identical bodies)."
    ),
)
def get_run(
    run_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> AgentRunResponse:
    return agent_run.get_run_detail(db, current_user.org_id, current_user.id, run_id)


@router.post(
    "/runs/{run_id}/decision",
    description=(
        "Approves or rejects the action a paused agent run is waiting for, naming the pending "
        "tool call (the tool_call_id of pending_approval). Answers with the same NDJSON stream as "
        "POST /chat/sessions/{id}/messages, starting with a decision event; the run resumes "
        "and may pause again. Only the run's owner can decide (404 otherwise, like a missing run). "
        "409 when the run is not waiting for this call, when the approval has expired, when "
        "another decision won, or when the user has another active run. 503 without "
        "OPENAI_API_KEY. Only an approval executes the action."
    ),
)
def decide(
    run_id: int,
    data: DecisionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    events = agent_run.decide_run(
        db, current_user.org_id, current_user.id, run_id, data.tool_call_id, data.decision
    )
    return StreamingResponse(stream_lines(events), media_type="application/x-ndjson")
