import asyncio
import json
from collections.abc import AsyncIterator, Iterator

from fastapi import APIRouter, Depends, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent import run as agent_run
from app.dependencies import get_current_user, get_db
from app.models.users import User
from app.rag import chat as chat_service
from app.schemas.chat import MessageCreate, SessionDetailResponse, SessionResponse

# Chats are personal (like alerts) and change no shared organization data, so every role,
# viewers included, can use them: get_current_user, not require_editor
router = APIRouter(prefix="/chat", tags=["chat"])


# Writes the events as NDJSON lines. The sync generator runs in a worker thread, one event at a
# time.
# When the client leaves, Starlette stops reading but never closes a sync generator (only the
# garbage collector would, much later), so an agent run would stay "running" and block the user. The
# finally block closes it at once: the agent's own cleanup then marks the run cancelled
async def stream_lines(events: Iterator[dict]) -> AsyncIterator[str]:
    try:
        while True:
            event = await run_in_threadpool(next, events, None)
            if event is None:
                break
            yield json.dumps(event) + "\n"
    finally:
        # shield: the close must finish even if the request task is being cancelled
        await asyncio.shield(run_in_threadpool(events.close))


@router.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> SessionResponse:
    return chat_service.create_session(db, current_user.org_id, current_user.id)


@router.get("/sessions", response_model=list[SessionResponse])
def list_sessions(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> list[SessionResponse]:
    return chat_service.list_sessions(db, current_user.org_id, current_user.id)


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(
    session_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> SessionDetailResponse:
    return chat_service.get_session_detail(db, current_user.org_id, current_user.id, session_id)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> Response:
    chat_service.delete_session(db, current_user.org_id, current_user.id, session_id)
    return Response(status_code=204)


@router.post(
    "/sessions/{session_id}/messages",
    response_class=StreamingResponse,
    description=(
        "Asks a question and streams the answer as NDJSON (one JSON object per line, "
        "application/x-ndjson): first {type: sources, sources: [...]}, then {type: token, text} "
        "pieces, then {type: done, message_id, cited_numbers}. If something fails after the "
        "stream started, the last line is {type: error, detail} and nothing is saved. The body "
        "field mode is rag (default, the filings chat), agent (the research agent) or auto (a "
        "router chooses); agent runs add the events route, step, step_result and a done event "
        "with run_id and status. A missing session (404), a missing OpenAI key (503), an agent "
        "run already in progress (409), validation (422), 401 and 429 are normal JSON errors "
        "sent before the stream."
    ),
)
def ask_question(
    session_id: int,
    data: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    # ask_question() checks the session and the key right now (errors become JSON responses) and
    # calls the 2.4 chat for mode "rag"; the events it returns are written as they are produced,
    # one JSON line each
    events = agent_run.ask_question(
        db, current_user.org_id, current_user.id, session_id, data.question, data.ticker, data.mode
    )
    return StreamingResponse(stream_lines(events), media_type="application/x-ndjson")
