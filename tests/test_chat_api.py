import json
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.chat import ChatMessage, ChatSession, Citation
from app.models.chunks import DocumentChunk
from app.rag import llm
from app.repositories import chunks as chunk_repository
from tests.conftest import ScriptedChatModel

EXPORT_QUESTION = "Export controls restrict sales of data center products to China."
NOT_FOUND = {"detail": "Chat session not found"}


@pytest.fixture(autouse=True)
def default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RELEVANCE_THRESHOLD", 0.30)
    monkeypatch.setattr(settings, "CHAT_HISTORY_MESSAGES", 6)
    monkeypatch.setattr(settings, "RETRIEVAL_TOP_K", 5)
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA")
    monkeypatch.setattr(settings, "RAG_LOOKBACK_YEARS", 2)


def create_session(client: TestClient, person: dict) -> int:
    response = client.post("/chat/sessions", headers=person["headers"])
    assert response.status_code == 201
    return response.json()["id"]


def ask(client: TestClient, person: dict, session_id: int, body: dict):
    return client.post(
        f"/chat/sessions/{session_id}/messages", json=body, headers=person["headers"]
    )


def parse_events(response) -> list[dict]:
    # NDJSON: one JSON object per line
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = response.text.split("\n")
    assert lines[-1] == ""  # every event, the last one too, ends with a newline
    return [json.loads(line) for line in lines[:-1]]


def count_rows(db: Session, model: type) -> int:
    return db.execute(select(func.count()).select_from(model)).scalar_one()


# ---------- sessions ----------


def test_viewer_creates_a_session_with_exactly_the_documented_fields(
    client: TestClient, people: dict
) -> None:
    response = client.post("/chat/sessions", headers=people["viewer"]["headers"])

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"id", "title", "created_at", "updated_at"}
    assert body["title"] is None


def test_endpoints_need_a_token(client: TestClient) -> None:
    assert client.post("/chat/sessions").status_code == 401
    assert client.get("/chat/sessions").status_code == 401
    assert client.get("/chat/sessions/1").status_code == 401
    assert client.delete("/chat/sessions/1").status_code == 401
    assert client.post("/chat/sessions/1/messages", json={"question": "hi"}).status_code == 401


def test_list_shows_only_own_sessions_newest_used_first(client: TestClient, people: dict) -> None:
    first = create_session(client, people["analyst"])
    second = create_session(client, people["analyst"])
    create_session(client, people["colleague"])
    create_session(client, people["outsider"])

    response = client.get("/chat/sessions", headers=people["analyst"]["headers"])

    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [second, first]
    assert client.get("/chat/sessions", headers=people["viewer"]["headers"]).json() == []


def test_delete_returns_204_and_the_session_is_gone(client: TestClient, people: dict) -> None:
    session_id = create_session(client, people["viewer"])

    response = client.delete(f"/chat/sessions/{session_id}", headers=people["viewer"]["headers"])

    assert response.status_code == 204
    gone = client.get(f"/chat/sessions/{session_id}", headers=people["viewer"]["headers"])
    assert gone.status_code == 404


# ---------- asking ----------


def test_viewer_asks_and_gets_a_stream_then_reads_the_stored_conversation(
    client: TestClient,
    people: dict,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    answer = "Export controls limit data center sales to China [1]."
    script_chat(answer)
    viewer = people["viewer"]
    session_id = create_session(client, viewer)

    events = parse_events(ask(client, viewer, session_id, {"question": EXPORT_QUESTION}))

    assert events[0]["type"] == "sources"
    assert events[0]["sources"][0]["chunk_id"] == chat_chunks["nvda_export"].id
    assert set(events[0]["sources"][0]) == {
        "number",
        "chunk_id",
        "ticker",
        "fiscal_year",
        "section",
        "score",
        "content",
    }
    assert "".join(e["text"] for e in events if e["type"] == "token") == answer
    assert events[-1]["type"] == "done"
    assert events[-1]["cited_numbers"] == [1]

    detail = client.get(f"/chat/sessions/{session_id}", headers=viewer["headers"])
    assert detail.status_code == 200
    body = detail.json()
    assert set(body) == {"id", "title", "created_at", "updated_at", "messages"}
    assert body["title"] == EXPORT_QUESTION
    user_message, assistant_message = body["messages"]
    assert set(user_message) == {"id", "role", "content", "created_at", "citations", "run_id"}
    assert (user_message["role"], user_message["citations"]) == ("user", [])
    assert assistant_message["content"] == answer
    assert assistant_message["id"] == events[-1]["message_id"]
    (citation,) = assistant_message["citations"]
    assert set(citation) == {
        "number",
        "chunk_id",
        "ticker",
        "fiscal_year",
        "section",
        "score",
        "content",
    }
    assert citation["content"] == chat_chunks["nvda_export"].content


def test_nothing_relevant_gives_the_fixed_answer_through_the_api(
    client: TestClient, people: dict, chat_chunks: dict[str, DocumentChunk]
) -> None:
    # The autouse fake chat model fails the test if it is called
    session_id = create_session(client, people["viewer"])

    events = parse_events(
        ask(client, people["viewer"], session_id, {"question": "How do I bake banana bread?"})
    )

    assert events[0] == {"type": "sources", "sources": []}
    assert events[1]["text"].startswith("I don't know.")
    assert events[-1]["cited_numbers"] == []


def test_ticker_is_stored_uppercase_and_filters(
    client: TestClient,
    people: dict,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
    db: Session,
) -> None:
    monkeypatch.setattr(settings, "RELEVANCE_THRESHOLD", -1.0)
    script_chat("Answer.")
    session_id = create_session(client, people["analyst"])

    events = parse_events(
        ask(client, people["analyst"], session_id, {"question": EXPORT_QUESTION, "ticker": "aapl"})
    )

    assert {source["ticker"] for source in events[0]["sources"]} == {"AAPL"}
    stored = db.execute(select(ChatMessage.ticker).order_by(ChatMessage.id)).scalars().first()
    assert stored == "AAPL"


@pytest.mark.parametrize(
    ("body", "message_part"),
    [
        ({"question": ""}, "question"),
        ({"question": "   "}, "question"),
        ({"question": "x" * 1001}, "question"),
        ({}, "question"),
        ({"question": "hello", "ticker": "MSFT"}, "ticker: Value error, must be one of AAPL, NVDA"),
        ({"question": "hello", "ticker": ""}, "ticker"),
    ],
)
def test_invalid_input_is_a_422_before_any_stream(
    client: TestClient, people: dict, body: dict, message_part: str
) -> None:
    session_id = create_session(client, people["viewer"])

    response = ask(client, people["viewer"], session_id, body)

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert message_part in response.json()["detail"]


def test_question_of_exactly_1000_characters_is_accepted_and_stripped(
    client: TestClient, people: dict, chat_chunks: dict[str, DocumentChunk], db: Session
) -> None:
    session_id = create_session(client, people["viewer"])

    response = ask(client, people["viewer"], session_id, {"question": "  " + "q" * 1000 + "  "})

    events = parse_events(response)
    assert events[-1]["type"] == "done"
    first_message = db.execute(select(ChatMessage).order_by(ChatMessage.id)).scalars().first()
    assert first_message.content == "q" * 1000


def test_missing_key_is_a_503_json_error_before_the_stream(
    client: TestClient, people: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_key() -> None:
        raise ValueError("OPENAI_API_KEY is not set")

    monkeypatch.setattr(llm, "get_chat_model", no_key)
    session_id = create_session(client, people["viewer"])

    response = ask(client, people["viewer"], session_id, {"question": "hello"})

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Chat is disabled: OPENAI_API_KEY not set"}


def test_mid_stream_failure_ends_with_an_error_line_and_saves_nothing(
    client: TestClient,
    people: dict,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
    db: Session,
) -> None:
    script_chat("one two three four five [1]", fail_after_pieces=2)
    viewer = people["viewer"]
    session_id = create_session(client, viewer)

    response = ask(client, viewer, session_id, {"question": EXPORT_QUESTION})

    # The status is already 200 when the failure happens: the error is the last event
    events = parse_events(response)
    assert [event["type"] for event in events] == ["sources", "token", "token", "error"]
    assert "example.invalid" not in response.text
    assert count_rows(db, ChatMessage) == 0
    detail = client.get(f"/chat/sessions/{session_id}", headers=viewer["headers"]).json()
    assert detail["messages"] == []
    assert detail["title"] is None


def test_follow_up_through_the_api_stores_the_rewritten_question(
    client: TestClient,
    people: dict,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
    db: Session,
) -> None:
    model = script_chat(
        "First [1].", "Data Center revenue grew on demand for accelerated computing.", "Second [1]."
    )
    viewer = people["viewer"]
    session_id = create_session(client, viewer)
    parse_events(ask(client, viewer, session_id, {"question": EXPORT_QUESTION}))

    events = parse_events(ask(client, viewer, session_id, {"question": "What about growth?"}))

    assert len(model.received) == 3
    assert events[0]["sources"][0]["content"].startswith("Data Center revenue grew")
    rewritten = db.execute(
        select(ChatMessage.rewritten_question).where(ChatMessage.content == "What about growth?")
    ).scalar_one()
    assert rewritten == "Data Center revenue grew on demand for accelerated computing."


# ---------- personal access ----------


@pytest.mark.parametrize("other", ["colleague", "outsider"])
@pytest.mark.parametrize("action", ["get", "delete", "ask"])
def test_someone_elses_session_is_a_404_with_the_same_body_as_a_missing_one(
    client: TestClient,
    people: dict,
    chat_chunks: dict[str, DocumentChunk],
    db: Session,
    other: str,
    action: str,
) -> None:
    # The owner has a session with a conversation in it
    owner = people["analyst"]
    session_id = create_session(client, owner)

    def attempt(person: dict, target_id: int):
        headers = person["headers"]
        if action == "get":
            return client.get(f"/chat/sessions/{target_id}", headers=headers)
        if action == "delete":
            return client.delete(f"/chat/sessions/{target_id}", headers=headers)
        return ask(client, person, target_id, {"question": EXPORT_QUESTION})

    response = attempt(people[other], session_id)
    missing = attempt(people[other], 999999)

    assert response.status_code == 404
    assert response.json() == NOT_FOUND
    assert missing.status_code == 404
    assert missing.json() == NOT_FOUND
    # The owner's session is untouched: still there, and nothing was added to it
    assert db.get(ChatSession, session_id) is not None
    assert count_rows(db, ChatMessage) == 0


def test_owner_still_works_after_the_failed_attempts(
    client: TestClient, people: dict, chat_chunks: dict[str, DocumentChunk]
) -> None:
    session_id = create_session(client, people["analyst"])
    assert (
        client.get(
            f"/chat/sessions/{session_id}", headers=people["colleague"]["headers"]
        ).status_code
        == 404
    )

    own = client.get(f"/chat/sessions/{session_id}", headers=people["analyst"]["headers"])

    assert own.status_code == 200


# ---------- database behaviour seen through the API ----------


def test_delete_cascades_to_messages_and_citations(
    client: TestClient,
    people: dict,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
    db: Session,
) -> None:
    script_chat("Answer [1].")
    viewer = people["viewer"]
    session_id = create_session(client, viewer)
    parse_events(ask(client, viewer, session_id, {"question": EXPORT_QUESTION}))
    assert (count_rows(db, ChatMessage), count_rows(db, Citation)) == (2, 1)

    assert (
        client.delete(f"/chat/sessions/{session_id}", headers=viewer["headers"]).status_code == 204
    )

    assert (count_rows(db, ChatMessage), count_rows(db, Citation)) == (0, 0)


def test_cited_passage_is_still_returned_after_its_chunk_is_replaced(
    client: TestClient,
    people: dict,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
    db: Session,
) -> None:
    script_chat("Answer [1].")
    viewer = people["viewer"]
    session_id = create_session(client, viewer)
    parse_events(ask(client, viewer, session_id, {"question": EXPORT_QUESTION}))
    target = chat_chunks["nvda_export"]

    # What `--reembed` does first: delete the filing's chunk rows
    chunk_repository.delete_by_filing(db, target.filing_id)
    db.expire_all()

    detail = client.get(f"/chat/sessions/{session_id}", headers=viewer["headers"]).json()
    (citation,) = detail["messages"][1]["citations"]
    assert citation["chunk_id"] is None
    assert citation["content"] == EXPORT_QUESTION
    assert citation["ticker"] == "NVDA"
    assert citation["section"] == "risk_factors"
