import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agent import graph as agent_graph
from app.agent import run as agent_run
from app.agent import tools as agent_tools
from app.agent.graph import build_graph
from app.agent.run import RouteDecision, route_question
from app.agent.run import (
    open_checkpointer as real_open_checkpointer,  # not the test's in-memory one
)
from app.config import settings
from app.exceptions import ConflictError, NotFoundError, ServiceUnavailableError
from app.models.agent import AgentRun, ToolCall
from app.models.chat import ChatMessage, ChatSession, Citation
from app.models.chunks import DocumentChunk
from app.models.users import User
from app.rag import chat, llm
from app.rag.llm import get_chat_model as real_get_chat_model  # the real one, not the test fake
from app.repositories import agent as agent_repository
from app.repositories import chat as chat_repository
from app.repositories import organizations as organization_repository
from app.repositories import users as user_repository
from tests.conftest import CHAT_CHUNK_DATA, ScriptedChatModel, add_bars, tool_calls_message

EXPORT_TEXT = CHAT_CHUNK_DATA["nvda_export"][2]
APPLE_TEXT = CHAT_CHUNK_DATA["aapl_risk"][2]
NVDA_PRICE_CALL = ("get_price_history", {"ticker": "NVDA", "days": 30})
AAPL_PRICE_CALL = ("get_price_history", {"ticker": "AAPL", "days": 30})
QUESTION = "How did the NVIDIA stock move recently?"


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
    # alembic's fileConfig can disable existing loggers in the test process
    for name in ("app.agent.run", "app.agent.tools", "app.rag.chat"):
        logging.getLogger(name).disabled = False


@pytest.fixture
def world(agent_environment: None, market: dict, chat_chunks: dict, db: Session) -> dict:
    # Companies, financial facts, price bars and filing chunks, the agent patched to use the test
    # session and an in-memory checkpointer
    add_bars(db, market["NVDA"], [10, 11, 12])
    add_bars(db, market["AAPL"], [20, 21, 22])
    return {"chunks": chat_chunks}


@pytest.fixture
def users(db: Session) -> dict[str, User]:
    acme = organization_repository.create(db, "Acme")
    globex = organization_repository.create(db, "Globex")
    return {
        "owner": user_repository.create(db, acme.id, "owner@acme.com", "hash", "viewer"),
        "colleague": user_repository.create(db, acme.id, "colleague@acme.com", "hash", "analyst"),
        "outsider": user_repository.create(db, globex.id, "admin@globex.com", "hash", "admin"),
    }


@pytest.fixture
def chat_session(db: Session, users: dict[str, User]) -> ChatSession:
    return chat.create_session(db, users["owner"].org_id, users["owner"].id)


def ask(
    db: Session,
    user: User,
    chat_session: ChatSession,
    question: str = QUESTION,
    ticker: str | None = None,
    mode: str = "agent",
) -> list[dict]:
    events = agent_run.ask_question(
        db, user.org_id, user.id, chat_session.id, question, ticker, mode
    )
    return list(events)


def only_run(db: Session) -> AgentRun:
    return db.execute(select(AgentRun)).scalar_one()


def tool_calls_of(db: Session, run: AgentRun) -> list[ToolCall]:
    return agent_repository.list_tool_calls(db, run.id)


def messages_of(db: Session, chat_session: ChatSession) -> list[ChatMessage]:
    statement = select(ChatMessage).where(ChatMessage.session_id == chat_session.id)
    return list(db.execute(statement.order_by(ChatMessage.id)).scalars().all())


def citations_of(db: Session, message_id: int) -> list[Citation]:
    statement = select(Citation).where(Citation.message_id == message_id)
    return list(db.execute(statement.order_by(Citation.number)).scalars().all())


def count_rows(db: Session, model: type) -> int:
    return db.execute(select(func.count()).select_from(model)).scalar_one()


def during_price_calls(monkeypatch: pytest.MonkeyPatch, hook: Callable[[str], None]) -> None:
    # Calls hook(ticker) every time get_price_history reads prices, then reads them as usual
    original = agent_tools.price_service.get_prices

    def wrapper(db, ticker, days):
        hook(ticker)
        return original(db, ticker, days)

    monkeypatch.setattr(agent_tools.price_service, "get_prices", wrapper)


# ---------- a normal run ----------


def test_a_two_step_run_streams_the_events_in_order_and_saves_everything(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat(tool_calls_message(NVDA_PRICE_CALL), "NVDA rose from 10 to 12.")

    events = ask(db, users["owner"], chat_session, ticker="NVDA")

    types = [event["type"] for event in events]
    assert types[:3] == ["route", "step", "step_result"]
    assert set(types[3:-1]) == {"token"}
    assert types[-1] == "done"
    run = only_run(db)
    assert events[0] == {"type": "route", "route": "agent", "mode": "agent"}
    assert events[1] == {
        "type": "step",
        "step": 1,
        "tool": "get_price_history",
        "args": NVDA_PRICE_CALL[1],
    }
    (call,) = tool_calls_of(db, run)
    assert events[2] == {
        "type": "step_result",
        "step": 1,
        "tool": "get_price_history",
        "ok": True,
        "chars": len(call.output),
    }
    assert "".join(event["text"] for event in events if event["type"] == "token") == (
        "NVDA rose from 10 to 12."
    )

    # The two messages: the user's question first, then the answer
    user_message, assistant_message = messages_of(db, chat_session)
    assert (user_message.role, user_message.content, user_message.ticker) == (
        "user",
        QUESTION,
        "NVDA",
    )
    assert (assistant_message.role, assistant_message.content) == (
        "assistant",
        "NVDA rose from 10 to 12.",
    )
    assert assistant_message.model == "gpt-5.4-mini"
    assert events[-1] == {
        "type": "done",
        "message_id": assistant_message.id,
        "run_id": run.id,
        "status": "completed",
        "cited_numbers": [],
    }

    # The run
    assert (run.org_id, run.user_id, run.session_id) == (
        users["owner"].org_id,
        users["owner"].id,
        chat_session.id,
    )
    assert run.status == "completed"
    assert run.step_count == 2
    assert run.message_id == user_message.id
    assert run.answer_message_id == assistant_message.id
    assert run.error is None
    assert run.finished_at is not None and run.finished_at >= run.started_at

    # The trace
    assert call.run_id == run.id
    assert (call.step, call.tool_name, call.input) == (1, "get_price_history", NVDA_PRICE_CALL[1])
    assert call.tool_call_id == "call_0_get_price_history"
    assert call.is_error is False
    assert call.approval_status == "not_required"
    assert call.duration_ms >= 0
    assert json.loads(call.output)["ticker"] == "NVDA"

    # The session got its title from the question and was touched
    assert chat_session.title == QUESTION
    assert count_rows(db, Citation) == 0


def test_the_stored_tool_output_is_what_the_tool_returned(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat(tool_calls_message(NVDA_PRICE_CALL), "ok")
    config = {"configurable": {"org_id": users["owner"].org_id, "user_id": users["owner"].id}}
    direct = agent_tools.get_price_history.invoke(NVDA_PRICE_CALL[1], config=config)

    ask(db, users["owner"], chat_session)

    (call,) = tool_calls_of(db, only_run(db))
    assert json.loads(call.output) == json.loads(json.dumps(direct))


def test_parallel_tool_calls_are_all_traced_with_the_same_step_and_duration(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat(tool_calls_message(NVDA_PRICE_CALL, AAPL_PRICE_CALL), "both moved up")

    events = ask(db, users["owner"], chat_session)

    assert [event["type"] for event in events].count("step") == 2
    assert [event["type"] for event in events].count("step_result") == 2
    calls = tool_calls_of(db, only_run(db))
    assert [(call.step, call.tool_name, call.input["ticker"]) for call in calls] == [
        (1, "get_price_history", "NVDA"),
        (1, "get_price_history", "AAPL"),
    ]
    assert calls[0].duration_ms == calls[1].duration_ms
    assert only_run(db).step_count == 2


def test_a_run_without_tools_is_one_model_call(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    model = script_chat("Hello, I am the research assistant.")

    events = ask(db, users["owner"], chat_session, question="hi")

    assert [event["type"] for event in events if event["type"] != "token"] == ["route", "done"]
    assert only_run(db).step_count == 1
    assert tool_calls_of(db, only_run(db)) == []
    assert len(model.received) == 1


def test_the_first_title_is_kept_by_later_questions(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat("first answer", "second answer")

    ask(db, users["owner"], chat_session, question="First question " + "x" * 200)
    ask(db, users["owner"], chat_session, question="Second question")

    assert chat_session.title == ("First question " + "x" * 200)[:100]
    assert count_rows(db, AgentRun) == 2


# ---------- what the model is given ----------


def test_the_history_reaches_the_model_without_tool_messages_and_is_capped(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    for number in range(1, 9):
        role = "user" if number % 2 else "assistant"
        chat_repository.create_message(db, chat_session.id, role, f"message {number}")
    model = script_chat(tool_calls_message(NVDA_PRICE_CALL), "done")

    ask(db, users["owner"], chat_session, question="the new question")

    first_call = model.received[0]
    types = [type(message).__name__ for message in first_call]
    # The system prompt, the LAST 6 of the 8 stored messages (3 to 8), then the new question
    assert types == ["SystemMessage"] + ["HumanMessage", "AIMessage"] * 3 + ["HumanMessage"]
    assert [message.content for message in first_call[1:]] == [
        "message 3",
        "message 4",
        "message 5",
        "message 6",
        "message 7",
        "message 8",
        "the new question",
    ]
    # The second call sees the tool result of this run, but never any from an earlier run
    assert "ToolMessage" in [type(message).__name__ for message in model.received[1]]


def test_the_system_prompt_has_todays_date_the_scope_and_the_focus(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    model = script_chat("a", "b")

    ask(db, users["owner"], chat_session, ticker="NVDA")
    ask(db, users["owner"], chat_session)

    focused = model.prompt_text(0)
    unfocused = model.prompt_text(1)
    assert f"Today's date is {datetime.now(UTC).date().isoformat()}." in focused
    assert "AAPL (Apple) and NVDA (NVIDIA) only" in focused
    assert "The user focused on NVDA; comparisons may still use both companies." in focused
    assert "The user focused on" not in unfocused


def test_org_and_user_reach_the_tools_and_a_run_is_never_written_for_someone_else(
    db: Session,
    world: dict,
    users: dict[str, User],
    script_chat: Callable[..., ScriptedChatModel],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.agent.tools")
    colleague = users["colleague"]
    colleague_session = chat.create_session(db, colleague.org_id, colleague.id)
    script_chat(tool_calls_message(NVDA_PRICE_CALL), "done")

    ask(db, colleague, colleague_session)

    assert f"org_id={colleague.org_id} user_id={colleague.id} ticker=NVDA" in caplog.text
    run = only_run(db)
    assert (run.org_id, run.user_id) == (colleague.org_id, colleague.id)
    owner = users["owner"]
    assert agent_repository.get_run(db, owner.org_id, owner.id, run.id) is None
    assert agent_repository.get_run(db, colleague.org_id, colleague.id, run.id) is not None


# ---------- tool errors ----------


def test_a_tool_exception_is_an_error_row_and_the_run_continues(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat(
        tool_calls_message(("get_price_history", {"ticker": "MSFT"})),
        "I can only look at AAPL and NVDA.",
    )

    events = ask(db, users["owner"], chat_session)

    (call,) = tool_calls_of(db, only_run(db))
    assert call.is_error is True
    assert "Ticker MSFT is not available. Available tickers: AAPL, NVDA" in call.output
    assert [event for event in events if event["type"] == "step_result"][0]["ok"] is False
    assert events[-1]["status"] == "completed"
    assert messages_of(db, chat_session)[-1].content == "I can only look at AAPL and NVDA."


def test_a_bug_in_a_tool_fails_the_run_without_leaking_the_exception_text(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    def explode(ticker: str) -> None:
        if ticker == "NVDA":
            raise ValueError("SECRET internal detail")

    during_price_calls(monkeypatch, explode)
    script_chat(tool_calls_message(AAPL_PRICE_CALL), tool_calls_message(NVDA_PRICE_CALL), "unused")

    events = ask(db, users["owner"], chat_session)

    assert [event["type"] for event in events] == [
        "route",
        "step",
        "step_result",
        "step",
        "error",
    ]
    assert events[-1] == {
        "type": "error",
        "detail": "The research could not be completed. Try again.",
    }
    run = only_run(db)
    assert (run.status, run.error, run.step_count) == ("failed", "ValueError", 2)
    assert run.finished_at is not None
    assert messages_of(db, chat_session)[-1].content == agent_graph.FAILED_ANSWER
    # The partial trace: the finished call, and the call that was running when the bug happened
    first, second = tool_calls_of(db, run)
    assert (first.step, first.is_error) == (1, False)
    assert (second.step, second.is_error, second.output) == (2, True, "ToolFailed: ValueError")
    # The text of the exception is nowhere: not in the stream, the log, or the database
    everything = json.dumps(events) + caplog.text + str(run.error) + second.output
    assert "SECRET" not in everything
    assert "agent run" in caplog.text and "ValueError" in caplog.text
    assert count_rows(db, Citation) == 0


def test_an_openai_error_in_the_model_fails_the_run_with_the_class_name_only(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = script_chat(tool_calls_message(NVDA_PRICE_CALL), "unused")

    def break_the_model(ticker: str) -> None:
        model.fail_on_call = True  # the second model call raises an openai.APIConnectionError

    during_price_calls(monkeypatch, break_the_model)

    events = ask(db, users["owner"], chat_session)

    assert events[-1]["type"] == "error"
    assert "done" not in [event["type"] for event in events]
    run = only_run(db)
    assert (run.status, run.error) == ("failed", "APIConnectionError")
    assert len(tool_calls_of(db, run)) == 1


def test_an_empty_final_answer_is_a_failed_run(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat("   ")

    events = ask(db, users["owner"], chat_session)

    assert events[-1]["type"] == "error"
    run = only_run(db)
    assert (run.status, run.error) == ("failed", "EmptyAnswer")
    assert messages_of(db, chat_session)[-1].content == agent_graph.FAILED_ANSWER


def test_an_unexpected_exception_propagates_but_the_run_is_closed(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bug(ticker: str) -> None:
        raise KeyError("a bug")

    during_price_calls(monkeypatch, bug)
    script_chat(tool_calls_message(NVDA_PRICE_CALL), "unused")

    with pytest.raises(KeyError):
        ask(db, users["owner"], chat_session)

    run = only_run(db)
    assert (run.status, run.error) == ("failed", "KeyError")
    assert messages_of(db, chat_session)[-1].content == agent_graph.FAILED_ANSWER


# ---------- limits ----------


def test_a_model_that_always_asks_for_a_tool_stops_after_exactly_max_steps_calls(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "AGENT_MAX_STEPS", 3)
    model = script_chat(*[tool_calls_message(NVDA_PRICE_CALL) for _ in range(10)])

    events = ask(db, users["owner"], chat_session)

    assert len(model.received) == 3
    run = only_run(db)
    assert (run.status, run.step_count, run.error) == ("step_limit", 3, None)
    assert [call.step for call in tool_calls_of(db, run)] == [1, 2, 3]
    assert messages_of(db, chat_session)[-1].content == agent_graph.STEP_LIMIT_ANSWER
    assert events[-2] == {"type": "token", "text": agent_graph.STEP_LIMIT_ANSWER}
    assert events[-1]["status"] == "step_limit"
    assert count_rows(db, Citation) == 0


def test_the_last_allowed_model_call_can_still_give_the_answer(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "AGENT_MAX_STEPS", 3)
    script_chat(tool_calls_message(NVDA_PRICE_CALL), tool_calls_message(NVDA_PRICE_CALL), "answer")

    events = ask(db, users["owner"], chat_session)

    assert events[-1]["status"] == "completed"
    assert only_run(db).step_count == 3


def test_a_slow_step_ends_the_run_as_a_timeout_with_the_trace_saved(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(agent_run, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
    during_price_calls(monkeypatch, lambda ticker: clock.update(now=clock["now"] + 500))
    script_chat(tool_calls_message(NVDA_PRICE_CALL), "too late")

    events = ask(db, users["owner"], chat_session)

    run = only_run(db)
    assert (run.status, run.error) == ("timeout", None)
    assert messages_of(db, chat_session)[-1].content == agent_graph.TIMEOUT_ANSWER
    assert events[-2] == {"type": "token", "text": agent_graph.TIMEOUT_ANSWER}
    assert events[-1]["status"] == "timeout"
    (call,) = tool_calls_of(db, run)
    assert call.duration_ms == 500_000
    assert count_rows(db, Citation) == 0


def test_closing_the_generator_mid_run_marks_the_run_cancelled(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat(tool_calls_message(NVDA_PRICE_CALL), "never")
    owner = users["owner"]
    generator = agent_run.ask_question(
        db, owner.org_id, owner.id, chat_session.id, QUESTION, None, "agent"
    )

    assert next(generator)["type"] == "route"
    assert next(generator)["type"] == "step"
    generator.close()  # what happens when the client disconnects

    run = only_run(db)
    assert (run.status, run.error) == ("cancelled", None)
    assert run.finished_at is not None
    answer = messages_of(db, chat_session)[-1]
    assert (answer.role, answer.content) == ("assistant", agent_graph.CANCELLED_ANSWER)
    assert run.answer_message_id == answer.id
    assert count_rows(db, Citation) == 0
    # The tools step never ran, so there is nothing in the trace
    assert tool_calls_of(db, run) == []


# ---------- citations ----------


def search_call(text: str) -> tuple[str, dict]:
    # A question equal to a chunk's text has similarity 1 with it (fake embeddings)
    return ("search_filings", {"query": text})


def test_chunk_ids_become_numbers_in_order_of_first_appearance_with_snapshots(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    nvda_id = world["chunks"]["nvda_export"].id
    apple_id = world["chunks"]["aapl_risk"].id
    answer = (
        f"Apple relies on partners in Asia [{apple_id}]. NVIDIA faces export limits "
        f"[{nvda_id}]. Both are exposed [{nvda_id}][{apple_id}]."
    )
    script_chat(tool_calls_message(search_call(EXPORT_TEXT), search_call(APPLE_TEXT)), answer)

    events = ask(db, users["owner"], chat_session)

    expected = (
        "Apple relies on partners in Asia [1]. NVIDIA faces export limits [2]. "
        "Both are exposed [2][1]."
    )
    assistant = messages_of(db, chat_session)[-1]
    assert assistant.content == expected
    assert events[-1]["cited_numbers"] == [1, 2]
    first, second = citations_of(db, assistant.id)
    assert (first.number, first.chunk_id, first.ticker, first.section) == (
        1,
        apple_id,
        "AAPL",
        "risk_factors",
    )
    assert (second.number, second.chunk_id, second.ticker, second.section) == (
        2,
        nvda_id,
        "NVDA",
        "risk_factors",
    )
    assert first.content == APPLE_TEXT and second.content == EXPORT_TEXT
    assert first.filing_id == world["chunks"]["aapl_risk"].filing_id
    assert second.fiscal_year == world["chunks"]["nvda_export"].fiscal_year
    assert first.score == pytest.approx(1.0, abs=1e-3)
    assert second.score == pytest.approx(1.0, abs=1e-3)


def test_a_group_of_ids_is_renumbered_inside_the_brackets(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    nvda_id = world["chunks"]["nvda_export"].id
    apple_id = world["chunks"]["aapl_risk"].id
    script_chat(
        tool_calls_message(search_call(EXPORT_TEXT), search_call(APPLE_TEXT)),
        f"Both [{nvda_id}, {apple_id}].",
    )

    ask(db, users["owner"], chat_session)

    assert messages_of(db, chat_session)[-1].content == "Both [1, 2]."


def test_markers_that_are_not_ids_returned_by_a_search_of_this_run_stay_and_get_no_row(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    nvda_id = world["chunks"]["nvda_export"].id
    never_searched = world["chunks"]["nvda_mdna"].id  # exists, but no search returned it
    script_chat(
        tool_calls_message(search_call(EXPORT_TEXT)),
        f"Export limits [{nvda_id}]. Growth [{never_searched}]. Earlier answer [2]. "
        "Nothing [999999].",
    )

    events = ask(db, users["owner"], chat_session)

    assistant = messages_of(db, chat_session)[-1]
    assert assistant.content == (
        f"Export limits [1]. Growth [{never_searched}]. Earlier answer [2]. Nothing [999999]."
    )
    assert events[-1]["cited_numbers"] == [1]
    assert [citation.chunk_id for citation in citations_of(db, assistant.id)] == [nvda_id]


def test_a_chunk_deleted_during_the_run_is_not_cited(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = world["chunks"]["nvda_export"]
    chunk_id = chunk.id

    def re_embed(ticker: str) -> None:
        db.delete(chunk)
        db.flush()

    during_price_calls(monkeypatch, re_embed)
    script_chat(
        tool_calls_message(search_call(EXPORT_TEXT)),
        tool_calls_message(NVDA_PRICE_CALL),
        f"Export limits [{chunk_id}].",
    )

    ask(db, users["owner"], chat_session)

    assistant = messages_of(db, chat_session)[-1]
    assert assistant.content == f"Export limits [{chunk_id}]."
    assert citations_of(db, assistant.id) == []


def test_an_answer_without_markers_saves_no_citations(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat(tool_calls_message(search_call(EXPORT_TEXT)), "Export limits exist.")

    events = ask(db, users["owner"], chat_session)

    assert events[-1]["cited_numbers"] == []
    assert count_rows(db, Citation) == 0


def test_a_search_without_relevant_passages_cannot_be_cited(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    other = world["chunks"]["nvda_mdna"].id
    script_chat(tool_calls_message(search_call("How do I bake banana bread?")), f"Maybe [{other}].")

    ask(db, users["owner"], chat_session)

    assert messages_of(db, chat_session)[-1].content == f"Maybe [{other}]."
    assert count_rows(db, Citation) == 0


# ---------- routing ----------


def test_agent_mode_makes_no_router_call(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    # structured is empty: a router call would fail with an IndexError
    model = script_chat("answer")

    events = ask(db, users["owner"], chat_session, mode="agent")

    assert events[0] == {"type": "route", "route": "agent", "mode": "agent"}
    assert len(model.received) == 1  # the agent call only


def test_auto_mode_calls_the_router_once_and_runs_the_agent_when_it_says_agent(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    model = script_chat("answer", structured=[RouteDecision(route="agent")])

    events = ask(db, users["owner"], chat_session, mode="auto")

    assert events[0] == {"type": "route", "route": "agent", "mode": "auto"}
    assert events[-1]["type"] == "done" and "run_id" in events[-1]
    assert len(model.received) == 2  # one router call, one agent call
    router_prompt = model.prompt_text(0)
    assert QUESTION in router_prompt
    assert "(none)" in router_prompt  # no history yet


def test_auto_mode_routes_to_the_rag_chat_when_the_router_says_rag(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat("Export controls matter [1].", structured=[RouteDecision(route="rag")])

    events = ask(db, users["owner"], chat_session, question=EXPORT_TEXT, mode="auto")

    assert events[0] == {"type": "route", "route": "rag", "mode": "auto"}
    assert [event["type"] for event in events[1:] if event["type"] != "token"] == [
        "sources",
        "done",
    ]
    assert events[-1]["cited_numbers"] == [1]
    assert "run_id" not in events[-1]
    # The RAG path saved its own messages and no agent run
    assert count_rows(db, AgentRun) == 0
    assert [message.role for message in messages_of(db, chat_session)] == ["user", "assistant"]


def test_rag_mode_streams_exactly_what_the_2_4_chat_streams_and_makes_no_router_call(
    db: Session,
    world: dict,
    users: dict[str, User],
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    first = chat.create_session(db, owner.org_id, owner.id)
    second = chat.create_session(db, owner.org_id, owner.id)
    model = script_chat("Export controls matter [1].", "Export controls matter [1].")

    direct = list(chat.ask(db, owner.org_id, owner.id, first.id, EXPORT_TEXT, None))
    through_dispatcher = ask(db, owner, second, question=EXPORT_TEXT, mode="rag")

    def without_ids(events: list[dict]) -> list[dict]:
        return [{k: v for k, v in event.items() if k != "message_id"} for event in events]

    assert without_ids(through_dispatcher) == without_ids(direct)
    assert "route" not in [event["type"] for event in through_dispatcher]
    assert len(model.received) == 2  # the two answers, no router
    assert count_rows(db, AgentRun) == 0


def test_a_router_failure_falls_back_to_rag_and_logs_only_the_class_name(
    script_chat: Callable[..., ScriptedChatModel], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="app.agent.run")
    model = script_chat()
    model.fail_on_call = True

    assert route_question("anything", []) == "rag"
    assert "router failed" in caplog.text and "APIConnectionError" in caplog.text


def test_a_router_without_a_decision_chooses_the_agent(
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat(structured=[None])

    assert route_question("anything", []) == "agent"


def test_the_router_sees_the_history_and_its_prompt_has_the_two_paths(
    db: Session,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    model = script_chat(structured=[RouteDecision(route="rag")])
    history = [
        chat_repository.create_message(db, chat_session.id, "user", "What are NVDA's risks?"),
        chat_repository.create_message(db, chat_session.id, "assistant", "Export controls."),
    ]

    assert route_question("And Apple's?", history) == "rag"

    prompt = model.prompt_text(0)
    assert "User: What are NVDA's risks?" in prompt
    assert "Assistant: Export controls." in prompt
    assert "Question: And Apple's?" in prompt
    system = agent_run.ROUTER_PROMPT.messages[0].prompt.template
    for rule in [
        "rag:",
        "agent:",
        "ONE company",
        "comparison",
        "When you are unsure, choose agent",
    ]:
        assert rule in system


# ---------- guards ----------


def test_a_second_ask_while_a_run_is_running_is_a_409_before_any_stream(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    first_message = chat_repository.create_message(db, chat_session.id, "user", "earlier")
    agent_repository.create_run(db, owner.org_id, owner.id, chat_session.id, first_message.id)
    model = script_chat()

    for mode in ("agent", "auto"):
        with pytest.raises(ConflictError, match="An agent run is already in progress"):
            agent_run.ask_question(
                db, owner.org_id, owner.id, chat_session.id, QUESTION, None, mode
            )

    assert model.received == []
    assert count_rows(db, AgentRun) == 1
    assert len(messages_of(db, chat_session)) == 1


def test_the_guard_looks_only_at_the_users_own_running_runs(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    colleague = users["colleague"]
    colleague_session = chat.create_session(db, colleague.org_id, colleague.id)
    message = chat_repository.create_message(db, colleague_session.id, "user", "earlier")
    agent_repository.create_run(
        db, colleague.org_id, colleague.id, colleague_session.id, message.id
    )
    script_chat("answer")

    events = ask(db, owner, chat_session)  # the colleague's run does not block the owner

    assert events[-1]["status"] == "completed"


def test_a_stale_running_row_is_ignored_and_a_finished_run_does_not_block(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    message = chat_repository.create_message(db, chat_session.id, "user", "earlier")
    stale = agent_repository.create_run(db, owner.org_id, owner.id, chat_session.id, message.id)
    stale.started_at = datetime.now(UTC) - timedelta(seconds=2 * 120 + 5)
    finished = agent_repository.create_run(db, owner.org_id, owner.id, chat_session.id, message.id)
    agent_repository.update_run(db, finished, status="completed")
    script_chat("answer")

    events = ask(db, owner, chat_session)

    assert events[-1]["status"] == "completed"
    # The cutoff is exactly twice the timeout: just inside it still blocks
    stale.started_at = datetime.now(UTC) - timedelta(seconds=2 * 120 - 5)
    db.flush()
    with pytest.raises(ConflictError):
        agent_run.ask_question(db, owner.org_id, owner.id, chat_session.id, QUESTION, None, "agent")


def test_the_guard_also_stops_a_run_that_starts_while_one_is_running(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = users["owner"]
    other_session = chat.create_session(db, owner.org_id, owner.id)
    refused = []

    def ask_again(ticker: str) -> None:
        # Called while the first run is in its tools step: its row is "running"
        try:
            agent_run.ask_question(db, owner.org_id, owner.id, other_session.id, "x", None, "agent")
        except ConflictError as exc:
            refused.append(exc.message)

    during_price_calls(monkeypatch, ask_again)
    script_chat(tool_calls_message(NVDA_PRICE_CALL), "done")

    ask(db, owner, chat_session)

    assert refused == ["An agent run is already in progress"]


@pytest.mark.parametrize("who", ["colleague", "outsider", "missing"])
def test_someone_elses_or_a_missing_session_is_a_404_before_the_stream_and_creates_nothing(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    who: str,
) -> None:
    model = script_chat()
    caller = users["owner"] if who == "missing" else users[who]
    session_id = 999_999 if who == "missing" else chat_session.id

    with pytest.raises(NotFoundError, match="Chat session not found"):
        agent_run.ask_question(db, caller.org_id, caller.id, session_id, QUESTION, None, "agent")

    assert model.received == []
    assert count_rows(db, AgentRun) == 0
    assert messages_of(db, chat_session) == []


def test_a_missing_key_is_a_503_before_the_stream_and_creates_nothing(
    db: Session,
    users: dict[str, User],
    chat_session: ChatSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "get_chat_model", real_get_chat_model)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    owner = users["owner"]

    for mode in ("agent", "auto"):
        with pytest.raises(ServiceUnavailableError, match="OPENAI_API_KEY not set"):
            agent_run.ask_question(
                db, owner.org_id, owner.id, chat_session.id, QUESTION, None, mode
            )

    assert count_rows(db, AgentRun) == 0


def test_the_request_session_is_released_before_the_agent_works(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commits = []
    original_commit = db.commit
    monkeypatch.setattr(db, "commit", lambda: commits.append("commit") or original_commit())
    script_chat("answer")
    owner = users["owner"]

    generator = agent_run.ask_question(
        db, owner.org_id, owner.id, chat_session.id, QUESTION, None, "agent"
    )

    # The checks ran and the read transaction ended before the generator produced anything
    assert commits == ["commit"]
    generator.close()


# ---------- persistence ----------


def test_a_real_postgres_checkpointer_keeps_the_thread_of_a_run(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Undo the in-memory replacement of the fixture: this test uses the real saver and the
    # tables created by the migration
    monkeypatch.setattr(agent_run, "open_checkpointer", real_open_checkpointer)
    script_chat(tool_calls_message(NVDA_PRICE_CALL), "NVDA rose.")

    ask(db, users["owner"], chat_session)

    run = only_run(db)
    config = {"configurable": {"thread_id": str(run.id)}}
    with real_open_checkpointer() as saver:
        try:
            state = build_graph(saver).get_state(config)
            kinds = [type(message).__name__ for message in state.values["messages"]]
            assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
            assert state.values["messages"][-1].text == "NVDA rose."
            assert state.next == ()
        finally:
            # The checkpoint tables are not rolled back with the test transaction
            saver.delete_thread(str(run.id))


def test_deleting_the_chat_session_removes_its_runs_and_tool_calls_in_the_database(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    script_chat(tool_calls_message(NVDA_PRICE_CALL), "done")
    ask(db, owner, chat_session)
    assert count_rows(db, ToolCall) == 1

    chat.delete_session(db, owner.org_id, owner.id, chat_session.id)

    assert count_rows(db, AgentRun) == 0
    assert count_rows(db, ToolCall) == 0


def test_the_database_rejects_an_unknown_run_status_and_approval_status(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat(tool_calls_message(NVDA_PRICE_CALL), "done")
    ask(db, users["owner"], chat_session)
    run = only_run(db)
    (call,) = tool_calls_of(db, run)

    with pytest.raises(IntegrityError, match="ck_agent_runs_status"):
        with db.begin_nested():
            run.status = "paused"
            db.flush()
    db.expire_all()
    with pytest.raises(IntegrityError, match="ck_tool_calls_approval_status"):
        with db.begin_nested():
            call.approval_status = "maybe"
            db.flush()
    db.expire_all()
    for status in ("approved", "pending", "rejected"):
        call.approval_status = status
        db.flush()  # 3.3 will use these


def test_every_chunk_snapshot_survives_a_run_for_a_chunk_that_is_replaced_later(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    chunk_id = world["chunks"]["nvda_export"].id
    script_chat(tool_calls_message(search_call(EXPORT_TEXT)), f"Limits [{chunk_id}].")
    ask(db, users["owner"], chat_session)
    assistant = messages_of(db, chat_session)[-1]

    # A re-embed replaces the chunk rows: chunk_id becomes null, the snapshot stays
    db.execute(DocumentChunk.__table__.delete().where(DocumentChunk.id == chunk_id))
    db.expire_all()

    (citation,) = citations_of(db, assistant.id)
    assert citation.chunk_id is None
    assert citation.content == EXPORT_TEXT
