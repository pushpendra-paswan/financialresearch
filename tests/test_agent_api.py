import asyncio
import json
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent import run as agent_run
from app.agent.run import RouteDecision
from app.config import settings
from app.models.agent import AgentRun
from app.rag import llm
from app.rag.llm import get_chat_model as real_get_chat_model  # the real one, not the test fake
from app.repositories import agent as agent_repository
from app.repositories import chat as chat_repository
from app.routes.chat import stream_lines
from tests.conftest import CHAT_CHUNK_DATA, ScriptedChatModel, add_bars, tool_calls_message

EXPORT_TEXT = CHAT_CHUNK_DATA["nvda_export"][2]
NVDA_PRICE_CALL = ("get_price_history", {"ticker": "NVDA", "days": 30})
QUESTION = "How did the NVIDIA stock move recently?"
RUN_NOT_FOUND = {"detail": "Agent run not found"}
RUN_KEYS = {
    "id",
    "status",
    "step_count",
    "error",
    "started_at",
    "finished_at",
    "message_id",
    "answer_message_id",
    "tool_calls",
}
TOOL_CALL_KEYS = {
    "step",
    "tool_name",
    "input",
    "output",
    "is_error",
    "approval_status",
    "duration_ms",
}


@pytest.fixture(autouse=True)
def default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RELEVANCE_THRESHOLD", 0.30)
    monkeypatch.setattr(settings, "CHAT_HISTORY_MESSAGES", 6)
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA")
    monkeypatch.setattr(settings, "RAG_LOOKBACK_YEARS", 2)
    monkeypatch.setattr(settings, "AGENT_MAX_STEPS", 8)
    monkeypatch.setattr(settings, "AGENT_TIMEOUT_SECONDS", 120)


@pytest.fixture
def world(agent_environment: None, market: dict, chat_chunks: dict, db: Session) -> None:
    add_bars(db, market["NVDA"], [10, 11, 12])


def create_session(client: TestClient, person: dict) -> int:
    response = client.post("/chat/sessions", headers=person["headers"])
    assert response.status_code == 201
    return response.json()["id"]


def ask(client: TestClient, person: dict, session_id: int, body: dict):
    return client.post(
        f"/chat/sessions/{session_id}/messages", json=body, headers=person["headers"]
    )


def parse_events(response) -> list[dict]:
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = response.text.split("\n")
    assert lines[-1] == ""
    return [json.loads(line) for line in lines[:-1]]


def run_agent(
    client: TestClient, person: dict, script_chat: Callable[..., ScriptedChatModel]
) -> tuple[int, int, list[dict]]:
    # One finished agent run through HTTP: returns (session id, run id, events)
    script_chat(tool_calls_message(NVDA_PRICE_CALL), "NVDA rose from 10 to 12.")
    session_id = create_session(client, person)
    events = parse_events(ask(client, person, session_id, {"question": QUESTION, "mode": "agent"}))
    return session_id, events[-1]["run_id"], events


def test_a_viewer_runs_the_agent_and_reads_the_trace_of_their_own_run(
    client: TestClient,
    people: dict,
    world: None,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    viewer = people["viewer"]

    _session_id, run_id, events = run_agent(client, viewer, script_chat)

    assert [event["type"] for event in events if event["type"] != "token"] == [
        "route",
        "step",
        "step_result",
        "done",
    ]
    assert events[-1]["status"] == "completed"
    response = client.get(f"/agent/runs/{run_id}", headers=viewer["headers"])
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RUN_KEYS
    assert (body["status"], body["step_count"], body["error"]) == ("completed", 2, None)
    assert body["finished_at"] is not None
    assert body["answer_message_id"] == events[-1]["message_id"]
    (call,) = body["tool_calls"]
    assert set(call) == TOOL_CALL_KEYS
    assert (call["step"], call["tool_name"], call["input"]) == (
        1,
        "get_price_history",
        NVDA_PRICE_CALL[1],
    )
    assert (call["is_error"], call["approval_status"]) == (False, "not_required")
    assert json.loads(call["output"])["last_close"] == 12.0


@pytest.mark.parametrize("other", ["colleague", "outsider"])
def test_someone_elses_run_is_a_404_with_the_same_body_as_a_missing_one(
    client: TestClient,
    people: dict,
    world: None,
    script_chat: Callable[..., ScriptedChatModel],
    other: str,
) -> None:
    _session_id, run_id, _events = run_agent(client, people["analyst"], script_chat)

    someone_else = client.get(f"/agent/runs/{run_id}", headers=people[other]["headers"])
    missing = client.get("/agent/runs/999999", headers=people["analyst"]["headers"])

    assert someone_else.status_code == 404
    assert someone_else.json() == RUN_NOT_FOUND
    assert missing.status_code == 404
    assert missing.json() == RUN_NOT_FOUND
    # The owner still sees it
    assert (
        client.get(f"/agent/runs/{run_id}", headers=people["analyst"]["headers"]).status_code == 200
    )


def test_the_run_endpoint_needs_a_token(client: TestClient) -> None:
    assert client.get("/agent/runs/1").status_code == 401


def test_the_conversation_shows_run_id_on_agent_answers_and_null_on_rag_answers(
    client: TestClient,
    people: dict,
    world: None,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    viewer = people["viewer"]
    script_chat("Export controls matter [1].", tool_calls_message(NVDA_PRICE_CALL), "NVDA rose.")
    session_id = create_session(client, viewer)
    # No mode field: the API default is "rag", the 2.4 chat
    rag_events = parse_events(ask(client, viewer, session_id, {"question": EXPORT_TEXT}))
    agent_events = parse_events(
        ask(client, viewer, session_id, {"question": QUESTION, "mode": "agent"})
    )

    detail = client.get(f"/chat/sessions/{session_id}", headers=viewer["headers"]).json()

    assert "route" not in [event["type"] for event in rag_events]
    assert [message["role"] for message in detail["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message["run_id"] for message in detail["messages"]] == [
        None,
        None,
        None,
        agent_events[-1]["run_id"],
    ]
    assert detail["messages"][3]["content"] == "NVDA rose."


def test_auto_mode_through_the_api_sends_the_route_first(
    client: TestClient,
    people: dict,
    world: None,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    viewer = people["viewer"]
    script_chat("answer", structured=[RouteDecision(route="agent")])
    session_id = create_session(client, viewer)

    events = parse_events(ask(client, viewer, session_id, {"question": QUESTION, "mode": "auto"}))

    assert events[0] == {"type": "route", "route": "agent", "mode": "auto"}
    assert events[-1]["type"] == "done"


@pytest.mark.parametrize(
    "body",
    [
        {"question": QUESTION, "mode": "robot"},
        {"question": QUESTION, "mode": None},
        {"question": "   ", "mode": "agent"},
        {"mode": "agent"},
    ],
)
def test_invalid_input_is_a_422_before_any_stream(
    client: TestClient, people: dict, world: None, body: dict
) -> None:
    session_id = create_session(client, people["viewer"])

    response = ask(client, people["viewer"], session_id, body)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


def test_a_run_in_progress_is_a_409_json_error(
    client: TestClient, people: dict, world: None, db: Session
) -> None:
    viewer = people["viewer"]
    session_id = create_session(client, viewer)
    message = chat_repository.create_message(db, session_id, "user", "earlier")
    agent_repository.create_run(db, viewer["org_id"], viewer["user_id"], session_id, message.id)

    response = ask(client, viewer, session_id, {"question": QUESTION, "mode": "agent"})

    assert response.status_code == 409
    assert response.json() == {"detail": "An agent run is already in progress"}
    assert db.execute(select(func.count()).select_from(AgentRun)).scalar_one() == 1


@pytest.mark.parametrize("other", ["colleague", "outsider"])
def test_asking_in_someone_elses_session_is_the_chat_404(
    client: TestClient, people: dict, world: None, other: str
) -> None:
    session_id = create_session(client, people["analyst"])

    response = ask(client, people[other], session_id, {"question": QUESTION, "mode": "agent"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Chat session not found"}


def test_a_missing_key_is_a_503_json_error_for_the_agent_modes(
    client: TestClient, people: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = create_session(client, people["viewer"])
    monkeypatch.setattr(llm, "get_chat_model", real_get_chat_model)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")

    for mode in ("agent", "auto"):
        response = ask(client, people["viewer"], session_id, {"question": QUESTION, "mode": mode})
        assert response.status_code == 503
        assert response.json() == {"detail": "Chat is disabled: OPENAI_API_KEY not set"}


def test_a_failed_run_ends_the_stream_with_an_error_line_and_keeps_the_trace(
    client: TestClient,
    people: dict,
    world: None,
    script_chat: Callable[..., ScriptedChatModel],
    db: Session,
) -> None:
    viewer = people["viewer"]
    script_chat("   ")
    session_id = create_session(client, viewer)

    events = parse_events(ask(client, viewer, session_id, {"question": QUESTION, "mode": "agent"}))

    assert events[-1] == {
        "type": "error",
        "detail": "The research could not be completed. Try again.",
    }
    run = db.execute(select(AgentRun)).scalar_one()
    body = client.get(f"/agent/runs/{run.id}", headers=viewer["headers"]).json()
    assert (body["status"], body["error"]) == ("failed", "EmptyAnswer")


def test_when_the_client_leaves_the_stream_wrapper_closes_the_run_as_cancelled(
    db: Session,
    people: dict,
    world: None,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    # Starlette never closes a sync generator when the client disconnects. stream_lines does, so
    # the run does not stay "running" (and block the user with a 409) until the garbage collector
    # runs. aclose() is what happens to the wrapper when the response is dropped
    viewer = people["viewer"]
    script_chat(tool_calls_message(NVDA_PRICE_CALL), "never")
    chat_session = chat_repository.create_session(db, viewer["org_id"], viewer["user_id"])
    events = agent_run.ask_question(
        db, viewer["org_id"], viewer["user_id"], chat_session.id, QUESTION, None, "agent"
    )

    async def read_two_lines_then_leave() -> list[str]:
        lines = stream_lines(events)
        first_lines = [await anext(lines), await anext(lines)]
        await lines.aclose()
        return first_lines

    first_lines = asyncio.run(read_two_lines_then_leave())

    assert [json.loads(line)["type"] for line in first_lines] == ["route", "step"]
    run = db.execute(select(AgentRun)).scalar_one()
    assert (run.status, run.finished_at is not None) == ("cancelled", True)


def test_the_stream_wrapper_writes_one_json_line_per_event() -> None:
    def events():
        yield {"type": "a"}
        yield {"type": "b", "n": 1}

    async def collect() -> list[str]:
        return [line async for line in stream_lines(events())]

    assert asyncio.run(collect()) == ['{"type": "a"}\n', '{"type": "b", "n": 1}\n']
