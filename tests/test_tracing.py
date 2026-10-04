import json
import logging
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langsmith.utils import tracing_is_enabled
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent import run as agent_run
from app.agent.run import RouteDecision
from app.config import settings
from app.models.agent import AgentRun
from app.models.chat import ChatSession
from app.models.users import User
from app.rag import chat, llm
from app.rag.llm import get_langsmith_client as real_get_langsmith_client  # the real one
from app.repositories import organizations as organization_repository
from app.repositories import users as user_repository
from tests.conftest import CHAT_CHUNK_DATA, ScriptedChatModel, add_bars, tool_calls_message

EXPORT_TEXT = CHAT_CHUNK_DATA["nvda_export"][2]
MDNA_TEXT = CHAT_CHUNK_DATA["nvda_mdna"][2]
QUESTION = "How did the NVIDIA stock move recently?"
SECRET_KEY = "lsv2_pt_this_secret_must_never_appear"
CREATE_CALL = ("create_alert", {"ticker": "NVDA", "alert_type": "price_above", "threshold": 250})
PRICE_CALL = ("get_price_history", {"ticker": "NVDA", "days": 30})


@pytest.fixture(autouse=True)
def default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Do not let a local .env change the numbers the tests expect
    monkeypatch.setattr(settings, "RELEVANCE_THRESHOLD", 0.30)
    monkeypatch.setattr(settings, "CHAT_HISTORY_MESSAGES", 6)
    monkeypatch.setattr(settings, "CHAT_MODEL", "gpt-5.4-mini")
    monkeypatch.setattr(settings, "RETRIEVAL_TOP_K", 5)
    monkeypatch.setattr(settings, "RETRIEVAL_CANDIDATES_K", 20)
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA")
    monkeypatch.setattr(settings, "RAG_LOOKBACK_YEARS", 2)
    monkeypatch.setattr(settings, "AGENT_MAX_STEPS", 8)
    monkeypatch.setattr(settings, "AGENT_TIMEOUT_SECONDS", 120)
    monkeypatch.setattr(settings, "APPROVAL_TTL_MINUTES", 60)
    monkeypatch.setattr(settings, "LANGSMITH_PROJECT", "fin-copilot")
    for name in ("app.agent.run", "app.agent.tools", "app.rag.chat", "app.rag.llm"):
        logging.getLogger(name).disabled = False


# Every start event a FakeTracer saw, in order (reset for each test by the fixture below)
STARTS: list[dict] = []
FLUSHES: list[float | None] = []


class FakeTracer(BaseCallbackHandler):
    # Stands in for LangChainTracer: a real LangChain callback handler, so LangChain calls it with
    # the run name, tags and metadata exactly as it would call the real tracer. It sends nothing
    def __init__(self, client: object, project_name: str) -> None:
        self.client = client
        self.project_name = project_name

    def record(self, kind: str, name: str | None, parent_run_id, tags, metadata) -> None:
        STARTS.append(
            {
                "kind": kind,
                "name": name,
                "root": parent_run_id is None,
                "tags": tags or [],
                "metadata": metadata or {},
                "project": self.project_name,
                "client": self.client,
            }
        )

    def on_chain_start(
        self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs
    ):
        self.record("chain", kwargs.get("name"), parent_run_id, tags, metadata)

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kw
    ):
        self.record("llm", kw.get("name"), parent_run_id, tags, metadata)


class ExplodingTracer(FakeTracer):
    # A tracer that raises inside every callback. LangChain must log it and carry on
    def on_chain_start(self, *args, **kwargs):
        raise RuntimeError("tracer exploded")

    def on_chat_model_start(self, *args, **kwargs):
        raise RuntimeError("tracer exploded")


@pytest.fixture
def fake_client() -> SimpleNamespace:
    return SimpleNamespace(flush=lambda timeout=None: FLUSHES.append(timeout))


@pytest.fixture
def tracing(monkeypatch: pytest.MonkeyPatch, fake_client: SimpleNamespace) -> Callable[..., None]:
    # tracing() turns tracing ON for the rest of the test (a fake key, the fake tracer and the
    # fake client). Until it is called the test runs with tracing off, like every other test
    STARTS.clear()
    FLUSHES.clear()

    def turn_on(tracer_class: type = FakeTracer) -> None:
        monkeypatch.setattr(settings, "LANGSMITH_API_KEY", SECRET_KEY)
        monkeypatch.setattr(llm, "LangChainTracer", tracer_class)
        monkeypatch.setattr(llm, "get_langsmith_client", lambda: fake_client)

    return turn_on


@pytest.fixture
def world(agent_environment: None, market: dict, chat_chunks: dict, db: Session) -> dict:
    add_bars(db, market["NVDA"], [10, 11, 12])
    return {"chunks": chat_chunks}


@pytest.fixture
def owner(db: Session) -> User:
    acme = organization_repository.create(db, "Acme")
    return user_repository.create(db, acme.id, "owner@acme.com", "hash", "viewer")


@pytest.fixture
def chat_session(db: Session, owner: User) -> ChatSession:
    return chat.create_session(db, owner.org_id, owner.id)


def ask(
    db: Session,
    user: User,
    chat_session: ChatSession,
    question: str = QUESTION,
    mode: str = "agent",
    trace_tags: list[str] | None = None,
) -> list[dict]:
    events = agent_run.ask_question(
        db, user.org_id, user.id, chat_session.id, question, None, mode, trace_tags
    )
    return list(events)


def roots() -> list[dict]:
    return [start for start in STARTS if start["root"]]


def without_ids(events: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in event.items() if key not in ("run_id", "message_id")}
        for event in events
    ]


# ---------- tracing is off unless a call asks for it ----------


def test_global_tracing_is_off_in_every_test() -> None:
    # The autouse fixture removed every LangSmith / LangChain tracing variable and emptied the key,
    # so a developer's shell or .env can never switch tracing on
    import os

    assert tracing_is_enabled() is False
    assert not [name for name in os.environ if name.startswith(("LANGSMITH_", "LANGCHAIN_"))]
    assert settings.LANGSMITH_API_KEY == ""


def test_with_no_key_the_trace_config_is_empty_and_nothing_is_built() -> None:
    # get_langsmith_client is the "forbidden" stub here: building a client would fail the test
    assert llm.get_trace_config("agent_run", 1, 2, ["agent"]) == {}
    llm.flush_traces()  # a no-op, it must not build the client either


def test_a_run_without_a_key_is_not_traced(
    db: Session,
    world: dict,
    owner: User,
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat(tool_calls_message(PRICE_CALL), "NVDA rose.")

    events = ask(db, owner, chat_session)

    assert events[-1]["type"] == "done" and events[-1]["status"] == "completed"
    assert STARTS == []


def test_a_traced_run_gives_the_same_events_rows_and_answer_as_an_untraced_run(
    db: Session,
    world: dict,
    owner: User,
    script_chat: Callable[..., ScriptedChatModel],
    tracing: Callable[..., None],
) -> None:
    first = chat.create_session(db, owner.org_id, owner.id)
    second = chat.create_session(db, owner.org_id, owner.id)

    script_chat(tool_calls_message(PRICE_CALL), "NVDA rose.")
    untraced = ask(db, owner, first)
    assert STARTS == []

    tracing()
    script_chat(tool_calls_message(PRICE_CALL), "NVDA rose.")
    traced = ask(db, owner, second)

    assert STARTS  # this one really was traced
    assert without_ids(traced) == without_ids(untraced)
    runs = db.execute(select(AgentRun).order_by(AgentRun.id)).scalars().all()
    assert [(run.status, run.step_count, run.error) for run in runs] == [
        ("completed", 2, None),
        ("completed", 2, None),
    ]


# ---------- what a traced agent run carries ----------


def test_an_agent_run_trace_has_the_name_tags_thread_user_and_nested_model_calls(
    db: Session,
    world: dict,
    owner: User,
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    tracing: Callable[..., None],
    fake_client: SimpleNamespace,
) -> None:
    tracing()
    script_chat(tool_calls_message(PRICE_CALL), "NVDA rose.")

    ask(db, owner, chat_session, trace_tags=["eval:agent_run_1", "eval_task:7"])

    run = db.execute(select(AgentRun)).scalar_one()
    (root,) = roots()
    assert root["name"] == "agent_run"
    assert root["project"] == "fin-copilot"
    assert root["client"] is fake_client
    assert root["tags"] == [
        "agent",
        "mode:agent",
        f"run:{run.id}",
        "eval:agent_run_1",
        "eval_task:7",
    ]
    # user_id is a numeric string, thread_id is the CHAT SESSION (it replaces the checkpoint
    # thread id, which is the run id, that LangChain copies from configurable)
    assert root["metadata"]["user_id"] == str(owner.id)
    assert root["metadata"]["thread_id"] == f"chat-{chat_session.id}"
    # The model calls are children of the run, in the same trace
    model_calls = [start for start in STARTS if start["kind"] == "llm"]
    assert len(model_calls) == 2
    assert all(not call["root"] for call in model_calls)


def test_no_email_key_or_question_text_reaches_a_tag_or_metadata(
    db: Session,
    world: dict,
    owner: User,
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    tracing: Callable[..., None],
) -> None:
    tracing()
    script_chat(tool_calls_message(PRICE_CALL), "NVDA rose.")

    ask(db, owner, chat_session, question=QUESTION, trace_tags=["eval:run", "eval_task:3"])

    for start in STARTS:
        text = json.dumps({"tags": start["tags"], "metadata": start["metadata"]}, default=str)
        assert owner.email not in text
        assert "acme.com" not in text
        assert SECRET_KEY not in text
        assert QUESTION not in text
        assert "NVIDIA stock" not in text


def test_a_resumed_run_is_its_own_trace_in_the_same_thread_with_the_same_run_tag(
    db: Session,
    world: dict,
    owner: User,
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    tracing: Callable[..., None],
) -> None:
    tracing()
    script_chat(tool_calls_message(CREATE_CALL), "Alert created.")

    events = ask(db, owner, chat_session)
    approval = next(event for event in events if event["type"] == "approval_required")
    list(
        agent_run.decide_run(
            db, owner.org_id, owner.id, approval["run_id"], approval["tool_call_id"], "approve"
        )
    )

    first, resumed = roots()
    run_tag = f"run:{approval['run_id']}"
    assert (first["name"], resumed["name"]) == ("agent_run", "agent_run_resumed")
    assert run_tag in first["tags"] and run_tag in resumed["tags"]
    assert "resumed" in resumed["tags"] and "resumed" not in first["tags"]
    assert first["metadata"]["thread_id"] == resumed["metadata"]["thread_id"]
    assert first["metadata"]["thread_id"] == f"chat-{chat_session.id}"
    assert resumed["metadata"]["user_id"] == str(owner.id)


def test_auto_mode_traces_the_router_as_its_own_trace_and_then_the_run(
    db: Session,
    world: dict,
    owner: User,
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    tracing: Callable[..., None],
) -> None:
    tracing()
    script_chat("answer", structured=[RouteDecision(route="agent")])

    ask(db, owner, chat_session, mode="auto")

    router, agent = roots()
    assert router["name"] == "agent_router"
    assert router["tags"] == ["router", "mode:auto"]
    assert router["metadata"]["thread_id"] == f"chat-{chat_session.id}"
    assert agent["name"] == "agent_run" and "mode:auto" in agent["tags"]


def test_the_router_without_a_trace_config_is_not_traced(
    script_chat: Callable[..., ScriptedChatModel], tracing: Callable[..., None]
) -> None:
    # --route-samples calls route_question without a config
    tracing()
    script_chat(structured=[RouteDecision(route="rag")])

    assert agent_run.route_question("anything", []) == "rag"
    assert STARTS == []


def test_rag_chat_traces_the_rewrite_and_the_answer_as_two_traces(
    db: Session,
    owner: User,
    chat_session: ChatSession,
    chat_chunks: dict,
    script_chat: Callable[..., ScriptedChatModel],
    tracing: Callable[..., None],
) -> None:
    script_chat("First answer [1].", MDNA_TEXT, "Growth came from demand [1].")
    list(chat.ask(db, owner.org_id, owner.id, chat_session.id, EXPORT_TEXT, "NVDA"))
    assert STARTS == []  # tracing was still off

    tracing()
    list(chat.ask(db, owner.org_id, owner.id, chat_session.id, "And its growth?", "NVDA"))

    rewrite, answer = roots()
    assert (rewrite["name"], answer["name"]) == ("rag_chat", "rag_chat")
    assert rewrite["tags"] == ["rag", "mode:rag", "step:rewrite"]
    assert answer["tags"] == ["rag", "mode:rag", "step:answer"]
    for start in (rewrite, answer):
        assert start["metadata"]["user_id"] == str(owner.id)
        assert start["metadata"]["thread_id"] == f"chat-{chat_session.id}"


def test_auto_mode_that_routes_to_rag_tags_the_chat_with_the_mode(
    db: Session,
    world: dict,
    owner: User,
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    tracing: Callable[..., None],
) -> None:
    tracing()
    script_chat("Export controls matter [1].", structured=[RouteDecision(route="rag")])

    ask(db, owner, chat_session, question=EXPORT_TEXT, mode="auto")

    router, answer = roots()
    assert router["name"] == "agent_router"
    assert answer["name"] == "rag_chat"
    assert answer["tags"] == ["rag", "mode:auto", "step:answer"]


# ---------- tracing fails open ----------


def test_a_tracer_that_cannot_be_built_gives_an_empty_config_and_a_class_name_warning(
    monkeypatch: pytest.MonkeyPatch,
    tracing: Callable[..., None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenTracer:
        def __init__(self, client: object, project_name: str) -> None:
            raise RuntimeError(f"bad configuration for key {SECRET_KEY}")

    tracing(BrokenTracer)
    caplog.set_level(logging.WARNING, logger="app.rag.llm")

    assert llm.get_trace_config("agent_run", 1, 2, ["agent"]) == {}

    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["tracing is off for this call: RuntimeError"]
    assert SECRET_KEY not in caplog.text


def test_a_client_that_cannot_be_built_also_gives_an_empty_config(
    monkeypatch: pytest.MonkeyPatch, tracing: Callable[..., None]
) -> None:
    tracing()

    def broken_client() -> None:
        raise ValueError("no client")

    monkeypatch.setattr(llm, "get_langsmith_client", broken_client)

    assert llm.get_trace_config("agent_run", 1, 2, ["agent"]) == {}


def test_a_tracer_that_raises_in_every_callback_does_not_change_the_run(
    db: Session,
    world: dict,
    owner: User,
    script_chat: Callable[..., ScriptedChatModel],
    tracing: Callable[..., None],
) -> None:
    first = chat.create_session(db, owner.org_id, owner.id)
    second = chat.create_session(db, owner.org_id, owner.id)
    script_chat(tool_calls_message(PRICE_CALL), "NVDA rose.")
    untraced = ask(db, owner, first)

    tracing(ExplodingTracer)
    script_chat(tool_calls_message(PRICE_CALL), "NVDA rose.")
    exploded = ask(db, owner, second)

    assert without_ids(exploded) == without_ids(untraced)
    assert exploded[-1]["status"] == "completed"


def test_a_failing_tracer_does_not_break_the_rag_chat(
    db: Session,
    owner: User,
    chat_session: ChatSession,
    chat_chunks: dict,
    script_chat: Callable[..., ScriptedChatModel],
    tracing: Callable[..., None],
) -> None:
    tracing(ExplodingTracer)
    script_chat("Export controls matter [1].")

    events = list(chat.ask(db, owner.org_id, owner.id, chat_session.id, EXPORT_TEXT, "NVDA"))

    assert events[-1]["type"] == "done" and events[-1]["cited_numbers"] == [1]


def test_log_tracing_error_logs_only_the_class_name(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="app.rag.llm")

    llm.log_tracing_error(ConnectionError(f"POST failed, API Key: {SECRET_KEY}"))

    assert [record.getMessage() for record in caplog.records] == [
        "LangSmith tracing failed: ConnectionError"
    ]
    assert SECRET_KEY not in caplog.text


# ---------- flush and the real client's settings ----------


def test_flush_traces_is_a_no_op_when_off_and_flushes_with_a_limit_when_on(
    tracing: Callable[..., None],
) -> None:
    llm.flush_traces()
    assert FLUSHES == []

    tracing()
    llm.flush_traces()
    assert FLUSHES == [10]


def test_a_flush_that_fails_is_only_a_warning(
    monkeypatch: pytest.MonkeyPatch,
    tracing: Callable[..., None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    tracing()

    def broken_flush(timeout: float | None = None) -> None:
        raise OSError(f"cannot flush {SECRET_KEY}")

    monkeypatch.setattr(llm, "get_langsmith_client", lambda: SimpleNamespace(flush=broken_flush))
    caplog.set_level(logging.WARNING, logger="app.rag.llm")

    llm.flush_traces()  # must not raise

    assert [record.getMessage() for record in caplog.records] == ["flushing traces failed: OSError"]


def test_the_real_client_is_built_once_with_the_endpoint_the_limits_and_a_silenced_library_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Building a client makes no network call (nothing is traced here)
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", SECRET_KEY)
    monkeypatch.setattr(settings, "LANGSMITH_ENDPOINT", "")
    real_get_langsmith_client.cache_clear()
    try:
        client = real_get_langsmith_client()
        assert real_get_langsmith_client() is client  # one client, so one background thread
        assert client.api_url == "https://api.smith.langchain.com"  # empty endpoint: SDK default
        assert client.timeout_ms == (2000, 10000)
        assert logging.getLogger("langsmith").level == logging.CRITICAL

        real_get_langsmith_client.cache_clear()
        monkeypatch.setattr(settings, "LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")
        assert real_get_langsmith_client().api_url == "https://eu.api.smith.langchain.com"
    finally:
        real_get_langsmith_client.cache_clear()
