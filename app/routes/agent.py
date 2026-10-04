from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.agent import run as agent_run
from app.dependencies import get_current_user, get_db
from app.models.users import User
from app.schemas.agent import AgentRunResponse

# Agent runs are personal (like chats), so every role can read their own: get_current_user
router = APIRouter(prefix="/agent", tags=["agent"])


@router.get(
    "/runs/{run_id}",
    response_model=AgentRunResponse,
    description=(
        "One of the caller's own agent runs with its trace: the status, the number of model "
        "calls and every tool call (step, tool, input, output, error flag, duration). 404 for a "
        "missing run, a colleague's run and another organization's run (identical bodies)."
    ),
)
def get_run(
    run_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> AgentRunResponse:
    return agent_run.get_run_detail(db, current_user.org_id, current_user.id, run_id)
