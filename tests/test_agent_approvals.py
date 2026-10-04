import json
import logging
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agent import graph as agent_graph
from app.agent import run as agent_run
from app.agent.graph import build_graph
from app.agent.run import open_checkpointer as real_open_checkpointer
from app.agent.write_tools import WRITE_TOOLS, create_alert, summarize_write_call
from app.config import settings
from app.exceptions import ConflictError, NotFoundError, ServiceUnavailableError
from app.models.agent import AgentRun, ToolCall
from app.models.alerts import Alert, AlertType
from app.models.audit import AuditLog
from app.models.chat import ChatMessage, ChatSession, Citation
from app.models.users import User
from app.rag import chat, llm
from app.rag.llm import get_chat_model as real_get_chat_model  # the real one, not the test fake
from app.repositories import agent as agent_repository
from app.repositories import alerts as alert_repository
from app.repositories import chat as chat_repository
from app.repositories import organizations as organization_repository
from app.repositories import users as user_repository
from tests.conftest import CHAT_CHUNK_DATA, ScriptedChatModel, add_bars, tool_calls_message

EXPORT_TEXT = CHAT_CHUNK_DATA["nvda_export"][2]
APPLE_TEXT = CHAT_CHUNK_DATA["aapl_risk"][2]
NVDA_PRICE_CALL = ("get_price_history", {"ticker": "NVDA", "days": 30})
CREATE_CALL = ("create_alert", {"ticker": "NVDA", "alert_type": "price_above", "threshold": 250})
SECOND_CREATE_CALL = (
    "create_alert",
    {"ticker": "AAPL", "alert_type": "price_below", "threshold": 100.5},
)
QUESTION = "Set an alert for me when NVDA closes above 250"
SUMMARY = (
    "Create a price alert: notify you when NVDA's close rises above 250.00. It fires when the "
    "price crosses the level; if the close is already above it, nothing fires until the next "
    "crossing."
)
NOT_WAITING = "This run is not waiting for this approval"


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
    for name in ("app.agent.run", "app.agent.tools", "app.agent.write_tools", "app.rag.chat"):
        logging.getLogger(name).disabled = False


@pytest.fixture
def world(agent_environment: None, market: dict, chat_chunks: dict, db: Session) -> dict:
    add_bars(db, market["NVDA"], [10, 11, 12])
    return {"chunks": chat_chunks, "companies": market}


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


def ask(db: Session, user: User, chat_session: ChatSession, question: str = QUESTION) -> list[dict]:
    return list(
        agent_run.ask_question(db, user.org_id, user.id, chat_session.id, question, None, "agent")
    )


def decide(
    db: Session, user: User, run: AgentRun, row_id: int, decision: str = "approve"
) -> list[dict]:
    return list(agent_run.decide_run(db, user.org_id, user.id, run.id, row_id, decision))


def only_run(db: Session) -> AgentRun:
    db.expire_all()
    return db.execute(select(AgentRun)).scalar_one()


def tool_calls_of(db: Session, run: AgentRun) -> list[ToolCall]:
    db.expire_all()
    return agent_repository.list_tool_calls(db, run.id)


def messages_of(db: Session, chat_session: ChatSession) -> list[ChatMessage]:
    statement = select(ChatMessage).where(ChatMessage.session_id == chat_session.id)
    return list(db.execute(statement.order_by(ChatMessage.id)).scalars().all())


def count_rows(db: Session, model: type) -> int:
    return db.execute(select(func.count()).select_from(model)).scalar_one()


def alerts_of(db: Session) -> list[Alert]:
    db.expire_all()
    return list(db.execute(select(Alert).order_by(Alert.id)).scalars().all())


def audit_actions(db: Session) -> list[tuple[str, int | None]]:
    statement = select(AuditLog.action, AuditLog.entity_id).order_by(AuditLog.id)
    return [(action, entity_id) for action, entity_id in db.execute(statement).all()]


def pause_a_run(
    db: Session,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    *later_messages: str | AIMessage,
) -> tuple[ScriptedChatModel, AgentRun, dict]:
    # A run whose first model call asks for the alert: returns the model, the (paused) run and
    # the approval_required event. later_messages are what the model says after the decision
    model = script_chat(tool_calls_message(CREATE_CALL), *later_messages)
    events = ask(db, users["owner"], chat_session)
    approval = next(event for event in events if event["type"] == "approval_required")
    return model, only_run(db), approval


def during_price_calls(monkeypatch: pytest.MonkeyPatch, hook: Callable[[str], None]) -> None:
    from app.agent import tools as agent_tools

    original = agent_tools.price_service.get_prices

    def wrapper(db, ticker, days):
        hook(ticker)
        return original(db, ticker, days)

    monkeypatch.setattr(agent_tools.price_service, "get_prices", wrapper)


# ---------- the pause ----------


def test_a_create_alert_call_pauses_the_run_and_creates_nothing(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    model, run, approval = pause_a_run(db, users, chat_session, script_chat, "unused")

    # Nothing was created, and the model was called once: the run is waiting, not finished
    assert alerts_of(db) == []
    assert len(model.received) == 1
    assert (run.status, run.step_count, run.error) == ("waiting_approval", 1, None)
    assert run.finished_at is None

    # The pending row holds exactly the call the model made
    (pending,) = tool_calls_of(db, run)
    assert (pending.step, pending.tool_name, pending.approval_status) == (
        1,
        "create_alert",
        "pending",
    )
    assert pending.input == CREATE_CALL[1]
    assert pending.is_error is False

    # The assistant message is the placeholder, and the run points to it
    answer = messages_of(db, chat_session)[-1]
    assert (answer.role, answer.content) == ("assistant", agent_graph.APPROVAL_PENDING_ANSWER)
    assert run.answer_message_id == answer.id

    # The stream: route, step, approval_required, the placeholder as ONE token, done
    events = approval  # the approval_required event
    assert events == {
        "type": "approval_required",
        "run_id": run.id,
        "tool_call_id": pending.id,
        "tool": "create_alert",
        "args": CREATE_CALL[1],
        "summary": SUMMARY,
        "expires_at": (pending.created_at + timedelta(minutes=60)).isoformat(),
    }


def test_the_stream_of_a_paused_run_has_the_events_in_order(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat(tool_calls_message(CREATE_CALL), "unused")

    events = ask(db, users["owner"], chat_session)

    assert [event["type"] for event in events] == [
        "route",
        "step",
        "approval_required",
        "token",
        "done",
    ]
    run = only_run(db)
    assert events[3] == {"type": "token", "text": agent_graph.APPROVAL_PENDING_ANSWER}
    assert events[4] == {
        "type": "done",
        "message_id": run.answer_message_id,
        "run_id": run.id,
        "status": "waiting_approval",
        "cited_numbers": [],
    }


def test_a_paused_run_is_not_cancelled_and_never_executes_without_a_decision(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    _model, run, _approval = pause_a_run(db, users, chat_session, script_chat, "unused")
    owner = users["owner"]

    # The user reads it (twice) and never decides
    agent_run.get_run_detail(db, owner.org_id, owner.id, run.id)
    agent_run.get_run_detail(db, owner.org_id, owner.id, run.id)

    assert alerts_of(db) == []
    assert only_run(db).status == "waiting_approval"
    assert [action for action, _ in audit_actions(db)] == ["chat_session.create"]


def test_the_run_detail_shows_the_pending_approval_with_the_same_summary(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    _model, run, approval = pause_a_run(db, users, chat_session, script_chat, "unused")
    owner = users["owner"]

    detail = agent_run.get_run_detail(db, owner.org_id, owner.id, run.id)

    assert detail.status == "waiting_approval"
    pending = detail.pending_approval
    assert pending is not None
    assert (pending.tool_call_id, pending.tool_name, pending.args) == (
        approval["tool_call_id"],
        "create_alert",
        CREATE_CALL[1],
    )
    assert pending.summary == approval["summary"] == SUMMARY
    assert pending.expires_at.isoformat() == approval["expires_at"]
    assert pending.expired is False
    assert [call.approval_status for call in detail.tool_calls] == ["pending"]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            {"ticker": "AAPL", "alert_type": "price_below", "threshold": 100.5},
            "Create a price alert: notify you when AAPL's close falls below 100.50. It fires "
            "when the price crosses the level; if the close is already below it, nothing fires "
            "until the next crossing.",
        ),
        (
            {"ticker": "nvda", "alert_type": "daily_change_pct", "threshold": 5},
            "Create a daily change alert: notify you when NVDA's close moves by at least 5.00% "
            "in one day versus the previous close, up or down.",
        ),
        (
            # Four decimals are shown in full, never rounded to two
            {"ticker": "NVDA", "alert_type": "price_above", "threshold": 250.1234},
            SUMMARY.replace("250.00", "250.1234"),
        ),
        (
            {"ticker": "NVDA", "alert_type": "price_above", "threshold": 250.5},
            SUMMARY.replace("250.00", "250.50"),
        ),
    ],
)
def test_the_summary_is_generated_by_code_from_the_arguments(args: dict, expected: str) -> None:
    assert summarize_write_call("create_alert", args) == expected


def test_a_summary_for_an_unknown_write_tool_is_a_bug() -> None:
    with pytest.raises(ValueError, match="No summary"):
        summarize_write_call("delete_everything", {})


# ---------- approve and reject ----------


def test_approving_creates_the_alert_updates_the_row_and_replaces_the_placeholder(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    _model, run, approval = pause_a_run(db, users, chat_session, script_chat, "Alert created.")
    placeholder_id = run.answer_message_id

    events = decide(db, owner, run, approval["tool_call_id"])

    assert [event["type"] for event in events if event["type"] != "token"] == [
        "decision",
        "step_result",
        "done",
    ]
    assert events[0] == {"type": "decision", "decision": "approve", "tool": "create_alert"}
    assert events[1] == {
        "type": "step_result",
        "step": 1,
        "tool": "create_alert",
        "ok": True,
        "chars": events[1]["chars"],
    }

    # The alert belongs to the run's owner, with the right company, type and threshold
    (alert,) = alerts_of(db)
    assert (alert.org_id, alert.user_id) == (owner.org_id, owner.id)
    assert alert.company.ticker == "NVDA"
    assert (alert.alert_type, alert.threshold, alert.active) == (
        AlertType.price_above,
        Decimal("250"),
        True,
    )
    assert alert.watch_from == date.today()

    # The pending row was UPDATED (one row, no duplicate), now approved, with the tool's output
    (row,) = tool_calls_of(db, run)
    assert row.id == approval["tool_call_id"]
    assert row.approval_status == "approved"
    output = json.loads(row.output)
    assert output["status"] == "created" and output["alert"]["id"] == alert.id
    assert row.is_error is False

    # The audit rows: the decision (entity = the tool_calls row) and the alert itself
    assert ("agent.approve", row.id) in audit_actions(db)
    assert ("alert.create", alert.id) in audit_actions(db)

    # The placeholder is now the real answer: same message, no second assistant message
    assistant_messages = [m for m in messages_of(db, chat_session) if m.role == "assistant"]
    assert [m.id for m in assistant_messages] == [placeholder_id]
    assert assistant_messages[0].content == "Alert created."
    assert assistant_messages[0].model == "gpt-5.4-mini"
    run = only_run(db)
    assert (run.status, run.step_count, run.error) == ("completed", 2, None)
    assert run.finished_at is not None
    assert run.answer_message_id == placeholder_id
    assert events[-1] == {
        "type": "done",
        "message_id": placeholder_id,
        "run_id": run.id,
        "status": "completed",
        "cited_numbers": [],
    }


def test_rejecting_creates_nothing_and_the_model_gets_the_rejection(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    model, run, approval = pause_a_run(
        db, users, chat_session, script_chat, "Understood, no alert."
    )

    events = decide(db, owner, run, approval["tool_call_id"], "reject")

    assert events[0] == {"type": "decision", "decision": "reject", "tool": "create_alert"}
    assert alerts_of(db) == []
    (row,) = tool_calls_of(db, run)
    assert row.approval_status == "rejected"
    assert json.loads(row.output) == {
        "status": "rejected",
        "message": "The user rejected this action. Do not retry it.",
    }
    assert row.is_error is False
    actions = [action for action, _ in audit_actions(db)]
    assert "agent.reject" in actions
    assert "agent.approve" not in actions and "alert.create" not in actions
    # The model saw the rejection as the tool result and answered
    last_messages = model.received[-1]
    assert "The user rejected this action" in last_messages[-1].content
    assert messages_of(db, chat_session)[-1].content == "Understood, no alert."
    run = only_run(db)
    assert (run.status, run.step_count) == ("completed", 2)


def test_citations_from_searches_before_and_after_the_pause_are_numbered_together(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    nvda_id = world["chunks"]["nvda_export"].id
    apple_id = world["chunks"]["aapl_risk"].id
    answer = f"Apple relies on partners [{apple_id}]. NVIDIA faces export limits [{nvda_id}]."
    script_chat(
        tool_calls_message(("search_filings", {"query": EXPORT_TEXT})),  # before the pause
        tool_calls_message(CREATE_CALL),
        tool_calls_message(("search_filings", {"query": APPLE_TEXT})),  # after the pause
        answer,
    )

    events = ask(db, owner, chat_session)
    approval = next(event for event in events if event["type"] == "approval_required")
    assert approval["run_id"] == only_run(db).id
    run = only_run(db)
    assert run.step_count == 2
    # Only the search that finished before the pause is in the trace, plus the pending call
    assert [(c.tool_name, c.approval_status) for c in tool_calls_of(db, run)] == [
        ("search_filings", "not_required"),
        ("create_alert", "pending"),
    ]

    events = decide(db, owner, run, approval["tool_call_id"])

    run = only_run(db)
    assert (run.status, run.step_count) == ("completed", 4)
    assistant = messages_of(db, chat_session)[-1]
    assert assistant.content == "Apple relies on partners [1]. NVIDIA faces export limits [2]."
    assert events[-1]["cited_numbers"] == [1, 2]
    citations = list(
        db.execute(
            select(Citation).where(Citation.message_id == assistant.id).order_by(Citation.number)
        )
        .scalars()
        .all()
    )
    assert [(c.number, c.chunk_id, c.ticker) for c in citations] == [
        (1, apple_id, "AAPL"),
        (2, nvda_id, "NVDA"),
    ]
    assert citations[1].content == EXPORT_TEXT
    assert [c.step for c in tool_calls_of(db, run)] == [1, 2, 3]


def test_a_read_tool_and_a_write_tool_in_one_step_are_each_saved_once(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = users["owner"]
    price_reads = []
    during_price_calls(monkeypatch, price_reads.append)
    script_chat(tool_calls_message(NVDA_PRICE_CALL, CREATE_CALL), "Done.")

    events = ask(db, owner, chat_session)
    approval = next(event for event in events if event["type"] == "approval_required")
    run = only_run(db)

    # At the pause only the write call is saved: the read has no result yet
    assert [(c.tool_name, c.approval_status) for c in tool_calls_of(db, run)] == [
        ("create_alert", "pending")
    ]
    reads_before = len(price_reads)

    decide(db, owner, run, approval["tool_call_id"])

    rows = tool_calls_of(db, run)
    assert sorted((c.tool_name, c.approval_status) for c in rows) == [
        ("create_alert", "approved"),
        ("get_price_history", "not_required"),
    ]
    # The tools node restarts on resume, so the read tool ran a second time (and was saved once)
    assert len(price_reads) == reads_before + 1
    assert len(alerts_of(db)) == 1


def test_two_write_calls_in_one_step_pause_one_at_a_time_and_a_run_can_pause_twice(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    script_chat(
        tool_calls_message(CREATE_CALL, SECOND_CREATE_CALL),
        tool_calls_message(SECOND_CREATE_CALL),
        "Both are set.",
    )

    events = ask(db, owner, chat_session)
    first = next(event for event in events if event["type"] == "approval_required")
    run = only_run(db)
    assert first["args"] == CREATE_CALL[1]
    assert [(c.tool_name, c.approval_status) for c in tool_calls_of(db, run)] == [
        ("create_alert", "pending")
    ]

    # Approve the first: it is created, the second was turned away, the model proposes it again
    events = decide(db, owner, run, first["tool_call_id"])
    assert [event["type"] for event in events if event["type"] != "token"] == [
        "decision",
        "step_result",
        "step_result",
        "step",
        "approval_required",
        "done",
    ]
    assert events[-1]["status"] == "waiting_approval"
    second = next(event for event in events if event["type"] == "approval_required")
    assert second["args"] == SECOND_CREATE_CALL[1]
    assert second["tool_call_id"] != first["tool_call_id"]
    assert [alert.company.ticker for alert in alerts_of(db)] == ["NVDA"]
    run = only_run(db)
    assert (run.status, run.step_count) == ("waiting_approval", 2)
    rows = tool_calls_of(db, run)
    turned_away = next(row for row in rows if row.step == 1 and row.is_error)
    assert "Only one action can be proposed at a time" in turned_away.output
    assert turned_away.approval_status == "not_required"
    # The placeholder message is the same one (no second assistant message)
    assert len([m for m in messages_of(db, chat_session) if m.role == "assistant"]) == 1

    # Approve the second one: now both alerts exist and the run completes
    events = decide(db, owner, run, second["tool_call_id"])
    assert events[-1]["status"] == "completed"
    assert [alert.company.ticker for alert in alerts_of(db)] == ["NVDA", "AAPL"]
    assert [a for a, _ in audit_actions(db)].count("agent.approve") == 2
    run = only_run(db)
    assert (run.status, run.step_count) == ("completed", 3)
    assert messages_of(db, chat_session)[-1].content == "Both are set."
    assert sorted(row.approval_status for row in tool_calls_of(db, run)) == [
        "approved",
        "approved",
        "not_required",
    ]


# ---------- the safety tests: nothing executes without an exact "approve" ----------


def test_the_write_tool_called_outside_a_graph_never_writes(
    db: Session, world: dict, users: dict[str, User]
) -> None:
    owner = users["owner"]
    state = {"messages": [AIMessage(content="", tool_calls=[{**_call(CREATE_CALL), "id": "c1"}])]}

    with pytest.raises(RuntimeError):
        create_alert.func(
            ticker="NVDA",
            alert_type="price_above",
            threshold=250,
            state=state,
            tool_call_id="c1",
            config={"configurable": {"org_id": owner.org_id, "user_id": owner.id}},
        )

    assert alerts_of(db) == []
    assert audit_actions(db) == []


def _call(call: tuple[str, dict]) -> dict:
    return {"name": call[0], "args": call[1], "type": "tool_call"}


@pytest.mark.parametrize("value", [True, "yes", "APPROVE", "approved", {"approved": True}, 1, ""])
def test_a_resume_value_that_is_not_exactly_approve_creates_nothing(
    db: Session,
    world: dict,
    users: dict[str, User],
    script_chat: Callable[..., ScriptedChatModel],
    value: object,
) -> None:
    owner = users["owner"]
    script_chat(tool_calls_message(CREATE_CALL), "ok")
    graph = build_graph(InMemorySaver())
    config = {
        "configurable": {
            "org_id": owner.org_id,
            "user_id": owner.id,
            "thread_id": "t",
            "today": "2026-10-04",
        }
    }
    graph.invoke({"messages": [HumanMessage(QUESTION)]}, config)
    assert graph.get_state(config).next == ("tools",)

    graph.invoke(Command(resume=value), config)

    assert alerts_of(db) == []
    last_tool_message = graph.get_state(config).values["messages"][-2]
    assert json.loads(last_tool_message.content)["status"] == "rejected"


def test_a_missing_resume_value_creates_nothing(
    db: Session,
    world: dict,
    users: dict[str, User],
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    script_chat(tool_calls_message(CREATE_CALL), "ok")
    graph = build_graph(InMemorySaver())
    config = {
        "configurable": {
            "org_id": owner.org_id,
            "user_id": owner.id,
            "thread_id": "t",
            "today": "2026-10-04",
        }
    }
    graph.invoke({"messages": [HumanMessage(QUESTION)]}, config)

    # LangGraph 1.2.12 does not even accept a resume value of None (an UnboundLocalError inside the
    # library, before the tool runs). Either way nothing may be created
    with pytest.raises(UnboundLocalError):
        graph.invoke(Command(resume=None), config)

    assert alerts_of(db) == []


def test_org_and_user_are_not_arguments_the_model_can_set() -> None:
    schema = create_alert.tool_call_schema.model_json_schema()

    assert sorted(schema["properties"]) == ["alert_type", "threshold", "ticker"]
    assert [tool.name for tool in WRITE_TOOLS] == ["create_alert"]
    assert create_alert.handle_tool_error is True


def test_org_and_user_in_the_model_arguments_are_ignored_by_the_run(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner, outsider = users["owner"], users["outsider"]
    sneaky = {**CREATE_CALL[1], "org_id": outsider.org_id, "user_id": outsider.id}
    script_chat(tool_calls_message(("create_alert", sneaky)), "done")

    events = ask(db, owner, chat_session)
    approval = next(event for event in events if event["type"] == "approval_required")
    decide(db, owner, only_run(db), approval["tool_call_id"])

    (alert,) = alerts_of(db)
    assert (alert.org_id, alert.user_id) == (owner.org_id, owner.id)


# ---------- the pre-check: impossible alerts are refused BEFORE asking ----------


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            ("create_alert", {"ticker": "ZZZZ", "alert_type": "price_above", "threshold": 5}),
            "Company not found",
        ),
        (
            ("create_alert", {"ticker": "MSFT", "alert_type": "price_above", "threshold": 5}),
            "Ticker MSFT is not available",
        ),
        (
            ("create_alert", {"ticker": "NVDA", "alert_type": "price_above", "threshold": 0}),
            "threshold must be a number greater than 0",
        ),
        (
            ("create_alert", {"ticker": "NVDA", "alert_type": "price_above", "threshold": 1.23456}),
            "at most 4 decimals",
        ),
    ],
)
def test_an_impossible_alert_is_an_error_before_any_pause(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
    call: tuple[str, dict],
    message: str,
) -> None:
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA,ZZZZ")
    script_chat(tool_calls_message(call), "I could not do that.")

    events = ask(db, users["owner"], chat_session)

    assert [event["type"] for event in events if event["type"] != "token"] == [
        "route",
        "step",
        "step_result",
        "done",
    ]
    assert next(e for e in events if e["type"] == "step_result")["ok"] is False
    run = only_run(db)
    assert run.status == "completed"
    (row,) = tool_calls_of(db, run)
    assert row.is_error is True and message in row.output
    assert row.approval_status == "not_required"
    assert alerts_of(db) == []


def test_the_alert_limit_and_a_duplicate_are_errors_before_any_pause(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    nvda = world["companies"]["NVDA"]
    alert_repository.create(
        db, owner.org_id, owner.id, nvda.id, AlertType.price_above, Decimal("250"), date.today()
    )
    db.commit()

    # A duplicate
    script_chat(tool_calls_message(CREATE_CALL), "Already there.")
    ask(db, owner, chat_session)
    run = only_run(db)
    (row,) = tool_calls_of(db, run)
    assert row.is_error and "You already have this alert" in row.output
    assert run.status == "completed"

    # The limit of 50 per user
    for threshold in range(300, 349):
        alert_repository.create(
            db,
            owner.org_id,
            owner.id,
            nvda.id,
            AlertType.price_above,
            Decimal(threshold),
            date.today(),
        )
    db.commit()
    assert len(alerts_of(db)) == 50
    script_chat(
        tool_calls_message(("create_alert", {**CREATE_CALL[1], "threshold": 999})), "Too many."
    )
    second_session = chat.create_session(db, owner.org_id, owner.id)
    events = ask(db, owner, second_session)
    assert "approval_required" not in [event["type"] for event in events]
    rows = db.execute(select(ToolCall).order_by(ToolCall.id)).scalars().all()
    assert "Alert limit reached" in rows[-1].output
    assert count_rows(db, Alert) == 50
    assert all(row.approval_status == "not_required" for row in rows)


# ---------- the decision guards ----------


def test_a_colleague_and_an_outsider_cannot_decide_and_the_run_is_untouched(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    _model, run, approval = pause_a_run(db, users, chat_session, script_chat, "unused")

    for name in ("colleague", "outsider"):
        for decision in ("approve", "reject"):
            with pytest.raises(NotFoundError, match="Agent run not found"):
                decide(db, users[name], run, approval["tool_call_id"], decision)

    assert alerts_of(db) == []
    assert only_run(db).status == "waiting_approval"
    assert tool_calls_of(db, run)[0].approval_status == "pending"
    assert "agent.approve" not in [action for action, _ in audit_actions(db)]


def test_a_second_decision_on_the_same_run_is_a_409_and_creates_nothing(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    _model, run, approval = pause_a_run(db, users, chat_session, script_chat, "Done.")
    decide(db, owner, run, approval["tool_call_id"])
    assert len(alerts_of(db)) == 1

    with pytest.raises(ConflictError, match=NOT_WAITING):
        decide(db, owner, run, approval["tool_call_id"])
    with pytest.raises(ConflictError, match=NOT_WAITING):
        decide(db, owner, run, approval["tool_call_id"], "reject")

    assert len(alerts_of(db)) == 1
    assert [a for a, _ in audit_actions(db)].count("agent.approve") == 1


def test_a_decision_naming_another_call_is_a_409(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    _model, run, approval = pause_a_run(db, users, chat_session, script_chat, "unused")

    for wrong_id in (approval["tool_call_id"] + 1000, 0):
        with pytest.raises(ConflictError, match=NOT_WAITING):
            decide(db, owner, run, wrong_id)

    assert alerts_of(db) == []
    assert only_run(db).status == "waiting_approval"


def test_a_decision_on_a_run_that_is_not_waiting_is_a_409(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    script_chat(tool_calls_message(NVDA_PRICE_CALL), "NVDA rose.")
    ask(db, owner, chat_session, "How did NVDA move?")
    run = only_run(db)
    (row,) = tool_calls_of(db, run)

    with pytest.raises(ConflictError, match=NOT_WAITING):
        decide(db, owner, run, row.id)

    assert run.status == "completed"
    assert alerts_of(db) == []


def test_an_expired_approval_is_closed_and_never_executed(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    model, run, approval = pause_a_run(db, users, chat_session, script_chat, "unused")
    (pending,) = tool_calls_of(db, run)
    pending.created_at = datetime.now(UTC) - timedelta(minutes=61)
    db.commit()

    # Before anyone decides, the API already reports it as expired
    detail = agent_run.get_run_detail(db, owner.org_id, owner.id, run.id)
    assert detail.status == "waiting_approval"
    assert detail.pending_approval.expired is True

    with pytest.raises(ConflictError, match="This approval has expired"):
        decide(db, owner, run, approval["tool_call_id"])

    run = only_run(db)
    assert run.status == "expired" and run.finished_at is not None
    (row,) = tool_calls_of(db, run)
    assert row.approval_status == "expired"
    assert messages_of(db, chat_session)[-1].content == agent_graph.EXPIRED_ANSWER
    assert ("agent.expire", row.id) in audit_actions(db)
    assert alerts_of(db) == []
    assert len(model.received) == 1  # the model was never called again
    detail = agent_run.get_run_detail(db, owner.org_id, owner.id, run.id)
    assert (detail.status, detail.pending_approval) == ("expired", None)

    # The run is closed: another decision is the plain 409
    with pytest.raises(ConflictError, match=NOT_WAITING):
        decide(db, owner, run, approval["tool_call_id"])


def test_an_approval_just_inside_the_ttl_can_still_be_approved(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    _model, run, approval = pause_a_run(db, users, chat_session, script_chat, "Done.")
    (pending,) = tool_calls_of(db, run)
    pending.created_at = datetime.now(UTC) - timedelta(minutes=59)
    db.commit()

    decide(db, owner, run, approval["tool_call_id"])

    assert len(alerts_of(db)) == 1


def test_a_missing_key_is_a_503_and_the_run_keeps_waiting(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _model, run, approval = pause_a_run(db, users, chat_session, script_chat, "unused")
    monkeypatch.setattr(llm, "get_chat_model", real_get_chat_model)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")

    with pytest.raises(ServiceUnavailableError, match="OPENAI_API_KEY not set"):
        decide(db, users["owner"], run, approval["tool_call_id"])

    assert only_run(db).status == "waiting_approval"
    assert tool_calls_of(db, run)[0].approval_status == "pending"
    assert alerts_of(db) == []


def test_another_running_run_of_the_user_blocks_the_resume(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    _model, run, approval = pause_a_run(db, users, chat_session, script_chat, "unused")
    # The user started something else meanwhile, and it is still running
    other = chat_repository.create_message(db, chat_session.id, "user", "another question")
    agent_repository.create_run(db, owner.org_id, owner.id, chat_session.id, other.id)
    db.commit()

    with pytest.raises(ConflictError, match="An agent run is already in progress"):
        decide(db, owner, run, approval["tool_call_id"])

    assert tool_calls_of(db, run)[0].approval_status == "pending"
    assert alerts_of(db) == []


def test_a_waiting_run_does_not_block_a_new_question(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    pause_a_run(db, users, chat_session, script_chat, "unused")
    script_chat("A plain answer.")

    events = ask(db, owner, chat_session, "What is 2 + 2?")

    assert events[-1]["status"] == "completed"


def test_the_resumed_run_counts_as_started_again(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    _model, run, approval = pause_a_run(db, users, chat_session, script_chat, "Done.")
    run.started_at = datetime.now(UTC) - timedelta(hours=1)
    db.commit()

    generator = agent_run.decide_run(
        db, owner.org_id, owner.id, run.id, approval["tool_call_id"], "approve"
    )

    # Recorded before the stream starts, so the active-run guard and the stale cutoff can see it
    db.expire_all()
    recorded = agent_repository.get_run(db, owner.org_id, owner.id, run.id)
    assert recorded.status == "running"
    assert recorded.started_at > datetime.now(UTC) - timedelta(minutes=1)
    list(generator)


def test_two_simultaneous_decisions_exactly_one_wins(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, run, approval = pause_a_run(db, users, chat_session, script_chat, "unused")
    original = agent_repository.change_run_status

    def other_request_wins_first(*args, **kwargs):
        # The other decision gets through the compare-and-set just before ours
        original(*args, **kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(agent_repository, "change_run_status", other_request_wins_first)

    with pytest.raises(ConflictError, match=NOT_WAITING):
        decide(db, users["owner"], run, approval["tool_call_id"])

    assert alerts_of(db) == []
    assert len(model.received) == 1  # the graph was never resumed by the loser
    assert "agent.approve" not in [action for action, _ in audit_actions(db)]


def test_the_compare_and_set_changes_a_row_only_from_the_expected_status(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    _model, run, approval = pause_a_run(db, users, chat_session, script_chat, "unused")

    args = (db, owner.org_id, owner.id, run.id)
    assert agent_repository.change_run_status(*args, "running", "completed") is False
    assert agent_repository.change_run_status(*args, "waiting_approval", "running") is True
    assert agent_repository.change_run_status(*args, "waiting_approval", "running") is False
    # The owner filter is part of the statement
    other = (db, users["colleague"].org_id, users["colleague"].id, run.id)
    assert agent_repository.change_run_status(*other, "running", "failed") is False
    row_id = approval["tool_call_id"]
    assert agent_repository.change_tool_call_approval(db, run.id, row_id, "pending", "approved")
    assert not agent_repository.change_tool_call_approval(db, run.id, row_id, "pending", "approved")


# ---------- the limits ----------


@pytest.mark.parametrize("paused_at", [1, 2, 3, 4])
def test_model_calls_before_and_after_a_pause_stop_together_at_max_steps(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
    paused_at: int,
) -> None:
    monkeypatch.setattr(settings, "AGENT_MAX_STEPS", 4)
    owner = users["owner"]
    reads = [tool_calls_message(NVDA_PRICE_CALL) for _ in range(paused_at - 1)]
    endless = [tool_calls_message(NVDA_PRICE_CALL) for _ in range(10)]
    model = script_chat(*reads, tool_calls_message(CREATE_CALL), *endless)

    events = ask(db, owner, chat_session)
    approval = next(event for event in events if event["type"] == "approval_required")
    assert len(model.received) == paused_at

    events = decide(db, owner, only_run(db), approval["tool_call_id"])

    # The model was called exactly AGENT_MAX_STEPS times in total, the approval was executed, and
    # the run ended as step_limit
    assert len(model.received) == 4
    run = only_run(db)
    assert (run.status, run.step_count) == ("step_limit", 4)
    assert len(alerts_of(db)) == 1
    assert events[-1]["status"] == "step_limit"
    assert events[-2] == {"type": "token", "text": agent_graph.STEP_LIMIT_ANSWER}
    assert messages_of(db, chat_session)[-1].content == agent_graph.STEP_LIMIT_ANSWER


def test_a_pause_on_the_last_allowed_call_still_ends_the_run_cleanly(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "AGENT_MAX_STEPS", 1)
    owner = users["owner"]
    model = script_chat(tool_calls_message(CREATE_CALL), "never asked")

    events = ask(db, owner, chat_session)
    approval = next(event for event in events if event["type"] == "approval_required")
    decide(db, owner, only_run(db), approval["tool_call_id"], "reject")

    assert len(model.received) == 1
    run = only_run(db)
    assert (run.status, run.step_count) == ("step_limit", 1)
    assert tool_calls_of(db, run)[0].approval_status == "rejected"
    assert alerts_of(db) == []


def test_the_wait_for_the_human_does_not_count_against_the_run_timeout(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(agent_run, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
    owner = users["owner"]
    _model, run, approval = pause_a_run(db, users, chat_session, script_chat, "Done.")

    clock["now"] += 10_000  # the user took a long time (far beyond AGENT_TIMEOUT_SECONDS)
    events = decide(db, owner, run, approval["tool_call_id"])

    assert events[-1]["status"] == "completed"
    assert len(alerts_of(db)) == 1


def test_the_resumed_part_has_its_own_timeout(
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
    owner = users["owner"]
    _model, run, approval = pause_a_run(
        db, users, chat_session, script_chat, tool_calls_message(NVDA_PRICE_CALL), "too late"
    )

    events = decide(db, owner, run, approval["tool_call_id"])

    assert events[-1]["status"] == "timeout"
    assert only_run(db).status == "timeout"
    assert len(alerts_of(db)) == 1  # the approved action had been executed before the timeout
    assert messages_of(db, chat_session)[-1].content == agent_graph.TIMEOUT_ANSWER


def test_a_client_that_leaves_during_the_resume_cancels_the_run_but_keeps_the_decision(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    _model, run, approval = pause_a_run(db, users, chat_session, script_chat, "never")

    generator = agent_run.decide_run(
        db, owner.org_id, owner.id, run.id, approval["tool_call_id"], "approve"
    )
    assert next(generator)["type"] == "decision"
    assert next(generator)["type"] == "step_result"
    generator.close()  # the client disconnects

    run = only_run(db)
    assert run.status == "cancelled" and run.finished_at is not None
    assert messages_of(db, chat_session)[-1].content == agent_graph.CANCELLED_ANSWER
    (row,) = tool_calls_of(db, run)
    assert row.approval_status == "approved"  # the decision stays recorded
    assert ("agent.approve", row.id) in audit_actions(db)
    assert len(alerts_of(db)) == 1  # and the approved action did run


def test_a_client_that_leaves_right_after_the_decision_event_still_closes_the_run(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    _model, run, approval = pause_a_run(db, users, chat_session, script_chat, "never")

    generator = agent_run.decide_run(
        db, owner.org_id, owner.id, run.id, approval["tool_call_id"], "reject"
    )
    assert next(generator)["type"] == "decision"
    generator.close()

    run = only_run(db)
    assert run.status == "cancelled"
    assert tool_calls_of(db, run)[0].approval_status == "rejected"


def test_a_tool_bug_after_the_decision_fails_the_run_and_keeps_the_pending_row(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = users["owner"]
    _model, run, approval = pause_a_run(db, users, chat_session, script_chat, "never")

    def explode(db, org_id, user_id, data):
        raise ValueError("SECRET internal detail")

    from app.agent import write_tools

    monkeypatch.setattr(write_tools.alert_service, "create_alert", explode)

    events = decide(db, owner, run, approval["tool_call_id"])

    assert events[-1]["type"] == "error"
    assert "SECRET" not in json.dumps(events)
    run = only_run(db)
    assert (run.status, run.error) == ("failed", "ValueError")
    (row,) = tool_calls_of(db, run)  # updated, not duplicated
    assert row.approval_status == "approved"
    assert row.is_error and row.output == "ToolFailed: ValueError"
    assert messages_of(db, chat_session)[-1].content == agent_graph.FAILED_ANSWER
    assert alerts_of(db) == []


# ---------- persistence ----------


def test_a_real_postgres_checkpointer_survives_a_restart_between_pause_and_decision(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Undo the in-memory saver of the fixture. Every stream_run opens (and closes) its own
    # connection, so the resume uses a new saver, like a new process would
    monkeypatch.setattr(agent_run, "open_checkpointer", real_open_checkpointer)
    owner = users["owner"]
    script_chat(tool_calls_message(CREATE_CALL), "Alert created.")
    try:
        events = ask(db, owner, chat_session)
        approval = next(event for event in events if event["type"] == "approval_required")
        run = only_run(db)
        assert alerts_of(db) == []

        # The pause is in the database: a fresh saver sees it
        with real_open_checkpointer() as saver:
            state = build_graph(saver).get_state({"configurable": {"thread_id": str(run.id)}})
            assert state.next == ("tools",)

        events = decide(db, owner, run, approval["tool_call_id"])

        assert events[-1]["status"] == "completed"
        assert len(alerts_of(db)) == 1
        assert messages_of(db, chat_session)[-1].content == "Alert created."
    finally:
        # The checkpoint tables are not rolled back with the test transaction
        with real_open_checkpointer() as saver:
            saver.delete_thread(str(only_run(db).id))


# ---------- the database ----------


def test_the_database_accepts_the_new_values_and_rejects_others(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    _model, run, _approval = pause_a_run(db, users, chat_session, script_chat, "unused")
    (row,) = tool_calls_of(db, run)

    for status in ("waiting_approval", "expired", "running"):
        run.status = status
        db.flush()
    for status in ("pending", "approved", "rejected", "expired", "not_required"):
        row.approval_status = status
        db.flush()

    with pytest.raises(IntegrityError, match="ck_agent_runs_status"):
        with db.begin_nested():
            run.status = "paused"
            db.flush()
    db.expire_all()
    with pytest.raises(IntegrityError, match="ck_tool_calls_approval_status"):
        with db.begin_nested():
            row.approval_status = "maybe"
            db.flush()


def test_a_run_cannot_have_two_rows_for_the_same_tool_call_id(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    _model, run, _approval = pause_a_run(db, users, chat_session, script_chat, "unused")
    (row,) = tool_calls_of(db, run)

    duplicate = ToolCall(
        run_id=run.id,
        step=1,
        tool_call_id=row.tool_call_id,
        tool_name="create_alert",
        input={},
        output="x",
        duration_ms=0,
    )
    with pytest.raises(IntegrityError, match="uq_tool_calls_run_id_tool_call_id"):
        with db.begin_nested():
            db.add(duplicate)
            db.flush()
    db.expire_all()

    # Another run may use the same id
    other_message = chat_repository.create_message(db, chat_session.id, "user", "another")
    other_run = agent_repository.create_run(
        db, run.org_id, run.user_id, chat_session.id, other_message.id
    )
    db.add(
        ToolCall(
            run_id=other_run.id,
            step=1,
            tool_call_id=row.tool_call_id,
            tool_name="create_alert",
            input={},
            output="x",
            duration_ms=0,
        )
    )
    db.flush()


def test_deleting_the_chat_session_removes_a_paused_run_and_its_pending_call(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    owner = users["owner"]
    pause_a_run(db, users, chat_session, script_chat, "unused")
    assert count_rows(db, ToolCall) == 1

    chat.delete_session(db, owner.org_id, owner.id, chat_session.id)

    assert count_rows(db, AgentRun) == 0
    assert count_rows(db, ToolCall) == 0


def test_the_unique_constraint_exists_in_the_schema(db: Session) -> None:
    names = db.execute(
        text("select conname from pg_constraint where conrelid = 'tool_calls'::regclass")
    ).scalars()

    assert "uq_tool_calls_run_id_tool_call_id" in set(names)


# ---------- through HTTP ----------


def parse_events(response) -> list[dict]:
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = response.text.split("\n")
    assert lines[-1] == ""
    return [json.loads(line) for line in lines[:-1]]


def ask_over_http(client: TestClient, person: dict) -> tuple[int, list[dict]]:
    session_id = client.post("/chat/sessions", headers=person["headers"]).json()["id"]
    response = client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"question": QUESTION, "mode": "agent"},
        headers=person["headers"],
    )
    return session_id, parse_events(response)


def decide_over_http(client: TestClient, person: dict, run_id: int, body: dict):
    return client.post(f"/agent/runs/{run_id}/decision", json=body, headers=person["headers"])


def test_a_viewer_approves_their_own_alert_over_http(
    client: TestClient,
    people: dict,
    world: dict,
    db: Session,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    viewer = people["viewer"]
    script_chat(tool_calls_message(CREATE_CALL), "Alert created.")
    session_id, events = ask_over_http(client, viewer)
    assert events[-1]["status"] == "waiting_approval"
    approval = next(event for event in events if event["type"] == "approval_required")
    run_id = approval["run_id"]

    # The paused run shows its pending approval
    body = client.get(f"/agent/runs/{run_id}", headers=viewer["headers"]).json()
    assert body["status"] == "waiting_approval"
    assert set(body["pending_approval"]) == {
        "tool_call_id",
        "tool_name",
        "args",
        "summary",
        "expires_at",
        "expired",
    }
    assert body["pending_approval"]["summary"] == SUMMARY
    assert body["pending_approval"]["tool_call_id"] == approval["tool_call_id"]
    assert [call["approval_status"] for call in body["tool_calls"]] == ["pending"]
    # The conversation already contains the placeholder, tied to the run
    detail = client.get(f"/chat/sessions/{session_id}", headers=viewer["headers"]).json()
    assert detail["messages"][-1]["content"] == agent_graph.APPROVAL_PENDING_ANSWER
    assert detail["messages"][-1]["run_id"] == run_id

    response = decide_over_http(
        client, viewer, run_id, {"tool_call_id": approval["tool_call_id"], "decision": "approve"}
    )

    resumed = parse_events(response)
    assert [event["type"] for event in resumed if event["type"] != "token"] == [
        "decision",
        "step_result",
        "done",
    ]
    assert resumed[-1]["status"] == "completed"
    (alert,) = alerts_of(db)
    assert (alert.org_id, alert.user_id) == (viewer["org_id"], viewer["user_id"])
    # The alert is visible on the normal alerts endpoint of its owner
    listed = client.get("/alerts", headers=viewer["headers"]).json()
    assert [(item["ticker"], item["alert_type"], item["threshold"]) for item in listed] == [
        ("NVDA", "price_above", 250.0)
    ]
    body = client.get(f"/agent/runs/{run_id}", headers=viewer["headers"]).json()
    assert body["status"] == "completed" and body["pending_approval"] is None
    assert [call["approval_status"] for call in body["tool_calls"]] == ["approved"]
    detail = client.get(f"/chat/sessions/{session_id}", headers=viewer["headers"]).json()
    assert detail["messages"][-1]["content"] == "Alert created."


def test_other_people_get_a_404_for_a_decision_and_the_alert_is_not_created(
    client: TestClient,
    people: dict,
    world: dict,
    db: Session,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat(tool_calls_message(CREATE_CALL), "unused")
    _session_id, events = ask_over_http(client, people["viewer"])
    approval = next(event for event in events if event["type"] == "approval_required")
    body = {"tool_call_id": approval["tool_call_id"], "decision": "approve"}

    responses = [
        decide_over_http(client, people[name], approval["run_id"], body)
        for name in ("colleague", "outsider", "admin")
    ]
    missing = decide_over_http(client, people["viewer"], approval["run_id"] + 999, body)

    for response in responses + [missing]:
        assert response.status_code == 404
        assert response.json() == {"detail": "Agent run not found"}
    assert alerts_of(db) == []


def test_decision_errors_over_http_have_the_json_format(
    client: TestClient,
    people: dict,
    world: dict,
    db: Session,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    viewer = people["viewer"]
    script_chat(tool_calls_message(CREATE_CALL), "Done.")
    _session_id, events = ask_over_http(client, viewer)
    approval = next(event for event in events if event["type"] == "approval_required")
    run_id, row_id = approval["run_id"], approval["tool_call_id"]

    # 422: a bad decision, a missing id, a wrong type
    for bad in (
        {"tool_call_id": row_id, "decision": "maybe"},
        {"decision": "approve"},
        {"tool_call_id": "abc", "decision": "approve"},
        {"tool_call_id": row_id},
    ):
        response = decide_over_http(client, viewer, run_id, bad)
        assert response.status_code == 422
        assert set(response.json()) == {"detail"} and isinstance(response.json()["detail"], str)

    # 409: a wrong row id
    response = decide_over_http(
        client, viewer, run_id, {"tool_call_id": row_id + 1000, "decision": "approve"}
    )
    assert (response.status_code, response.json()) == (409, {"detail": NOT_WAITING})

    # 503: no key (the run keeps waiting)
    monkeypatch.setattr(llm, "get_chat_model", real_get_chat_model)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    response = decide_over_http(
        client, viewer, run_id, {"tool_call_id": row_id, "decision": "approve"}
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "Chat is disabled: OPENAI_API_KEY not set"}
    assert alerts_of(db) == []

    # 401 without a token
    response = client.post(
        f"/agent/runs/{run_id}/decision", json={"tool_call_id": row_id, "decision": "approve"}
    )
    assert response.status_code == 401


def test_a_second_decision_over_http_is_a_409_with_the_json_format(
    client: TestClient,
    people: dict,
    world: dict,
    db: Session,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    viewer = people["viewer"]
    script_chat(tool_calls_message(CREATE_CALL), "Done.")
    _session_id, events = ask_over_http(client, viewer)
    approval = next(event for event in events if event["type"] == "approval_required")
    body = {"tool_call_id": approval["tool_call_id"], "decision": "approve"}

    first = decide_over_http(client, viewer, approval["run_id"], body)
    second = decide_over_http(client, viewer, approval["run_id"], body)

    assert first.status_code == 200
    assert (second.status_code, second.json()) == (409, {"detail": NOT_WAITING})
    assert len(alerts_of(db)) == 1


def test_an_expired_decision_over_http_is_a_409_and_closes_the_run(
    client: TestClient,
    people: dict,
    world: dict,
    db: Session,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    viewer = people["viewer"]
    script_chat(tool_calls_message(CREATE_CALL), "Done.")
    _session_id, events = ask_over_http(client, viewer)
    approval = next(event for event in events if event["type"] == "approval_required")
    db.execute(text("update tool_calls set created_at = now() - interval '2 hours'"))
    db.commit()

    body = client.get(f"/agent/runs/{approval['run_id']}", headers=viewer["headers"]).json()
    assert body["pending_approval"]["expired"] is True

    response = decide_over_http(
        client,
        viewer,
        approval["run_id"],
        {"tool_call_id": approval["tool_call_id"], "decision": "approve"},
    )

    assert (response.status_code, response.json()) == (409, {"detail": "This approval has expired"})
    body = client.get(f"/agent/runs/{approval['run_id']}", headers=viewer["headers"]).json()
    assert (body["status"], body["pending_approval"]) == ("expired", None)
    assert body["tool_calls"][0]["approval_status"] == "expired"
    assert alerts_of(db) == []
