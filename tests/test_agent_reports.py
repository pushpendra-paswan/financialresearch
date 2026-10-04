import json
import logging
from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent import graph as agent_graph
from app.agent import run as agent_run
from app.agent.graph import build_graph
from app.agent.run import open_checkpointer as real_open_checkpointer
from app.agent.write_tools import WRITE_TOOLS, save_report, summarize_write_call
from app.config import settings
from app.models.agent import AgentRun, ToolCall
from app.models.alerts import Alert
from app.models.audit import AuditLog
from app.models.chat import ChatMessage, ChatSession
from app.models.reports import Report, ReportCitation, ReportCompany
from app.models.users import User
from app.rag import chat
from app.repositories import agent as agent_repository
from app.repositories import organizations as organization_repository
from app.repositories import reports as report_repository
from app.repositories import users as user_repository
from tests.conftest import CHAT_CHUNK_DATA, ScriptedChatModel, add_bars, tool_calls_message

EXPORT_TEXT = CHAT_CHUNK_DATA["nvda_export"][2]
APPLE_TEXT = CHAT_CHUNK_DATA["aapl_risk"][2]
SEARCH_APPLE = ("search_filings", {"query": APPLE_TEXT})
SEARCH_NVDA = ("search_filings", {"query": EXPORT_TEXT})
NVDA_REVENUE = ("get_financials", {"ticker": "NVDA", "metric": "revenue"})
OUT_OF_SCOPE = ("get_financials", {"ticker": "MSFT"})
ALERT_CALL = ("create_alert", {"ticker": "NVDA", "alert_type": "price_above", "threshold": 250})
QUESTION = "Write a report comparing Apple and NVIDIA"
TITLE = "Apple vs NVIDIA"
NOT_WAITING = "This run is not waiting for this approval"


@pytest.fixture(autouse=True)
def default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
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
        "analyst": user_repository.create(db, acme.id, "analyst@acme.com", "hash", "analyst"),
        "viewer": user_repository.create(db, acme.id, "viewer@acme.com", "hash", "viewer"),
        "outsider": user_repository.create(db, globex.id, "admin@globex.com", "hash", "admin"),
    }


@pytest.fixture
def chat_session(db: Session, users: dict[str, User]) -> ChatSession:
    return chat.create_session(db, users["analyst"].org_id, users["analyst"].id)


# ---------- helpers ----------


def report_content(*chunk_ids: int, size: int = 700) -> str:
    # A report text of exactly `size` characters that cites the given chunk ids
    sentences = " ".join(
        f"Statement {n} from the filings [{cid}]." for n, cid in enumerate(chunk_ids)
    )
    head = f"## Summary\n\n{sentences}\n\n"
    return head + ("Plain text line.\n" * 2000)[: size - len(head)]


def save_call(*chunk_ids: int, title: str = TITLE, size: int = 700, tickers=("AAPL", "NVDA")):
    return (
        "save_report",
        {
            "title": title,
            "content": report_content(*chunk_ids, size=size),
            "tickers": list(tickers),
        },
    )


def both_ids(world: dict) -> tuple[int, int]:
    return world["chunks"]["aapl_risk"].id, world["chunks"]["nvda_export"].id


def research_script(world: dict, *, size: int = 700) -> list[AIMessage]:
    # Two search steps, one financials step (a repeated call, a failing one), then save_report
    apple_id, nvda_id = both_ids(world)
    return [
        tool_calls_message(SEARCH_APPLE),
        tool_calls_message(SEARCH_NVDA),
        tool_calls_message(NVDA_REVENUE, NVDA_REVENUE, OUT_OF_SCOPE),
        tool_calls_message(save_call(nvda_id, apple_id, size=size)),
    ]


def ask(db: Session, user: User, chat_session: ChatSession, question: str = QUESTION) -> list[dict]:
    return list(
        agent_run.ask_question(db, user.org_id, user.id, chat_session.id, question, None, "agent")
    )


def decide(db: Session, user: User, run: AgentRun, row_id: int, decision: str = "approve"):
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


def reports_of(db: Session) -> list[Report]:
    db.expire_all()
    return list(db.execute(select(Report).order_by(Report.id)).scalars().all())


def audit_actions(db: Session) -> list[str]:
    # The agent and report rows (the chat session's own audit row is not of interest here)
    statement = select(AuditLog.action).where(
        AuditLog.action.like("agent.%") | AuditLog.action.like("report.%")
    )
    return list(db.execute(statement.order_by(AuditLog.id)).scalars().all())


def pause_on_report(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    *later: str,
    size: int = 700,
) -> tuple[ScriptedChatModel, AgentRun, dict]:
    model = script_chat(*research_script(world, size=size), *later)
    events = ask(db, users["analyst"], chat_session)
    approval = next(event for event in events if event["type"] == "approval_required")
    return model, only_run(db), approval


# ---------- the pause ----------


def test_a_save_report_call_pauses_the_run_and_saves_nothing(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    model, run, approval = pause_on_report(db, world, users, chat_session, script_chat, "unused")
    apple_id, nvda_id = both_ids(world)
    analyst = users["analyst"]

    assert reports_of(db) == []
    assert len(model.received) == 4  # four model calls, no more
    assert (run.status, run.step_count, run.error) == ("waiting_approval", 4, None)

    # The pending row holds exactly (and completely) what the model sent
    pending = tool_calls_of(db, run)[-1]
    expected_args = save_call(nvda_id, apple_id)[1]
    assert (pending.step, pending.tool_name, pending.approval_status) == (
        4,
        "save_report",
        "pending",
    )
    assert pending.input == expected_args
    assert len(pending.input["content"]) == 700

    # pending_approval.args is the complete text; the summary comes from code and says who sees it
    detail = agent_run.get_run_detail(db, analyst.org_id, analyst.id, run.id)
    assert detail.pending_approval.args == expected_args
    assert detail.pending_approval.summary == approval["summary"]
    assert "visible to everyone in your organization" in approval["summary"]
    assert approval["summary"].startswith(
        "Save a report titled 'Apple vs NVIDIA' about AAPL, NVDA (700 characters, 2 filing "
        "citations)."
    )
    assert approval["tool"] == "save_report"
    assert audit_actions(db) == []


def test_the_summary_is_generated_by_code_and_normalizes_the_arguments() -> None:
    content = "  " + "x" * 1_200 + " [5] [5, 6] [7]  "
    summary = summarize_write_call(
        "save_report",
        {"title": "  My title ", "content": content, "tickers": [" nvda", "AAPL", "NVDA"]},
    )
    assert summary == (
        "Save a report titled 'My title' about NVDA, AAPL (1,215 characters, 3 filing citations). "
        "It will be visible to everyone in your organization. Bracketed numbers in the text are "
        "filing passage ids; they become [1], [2], ... when the report is saved."
    )
    one = summarize_write_call(
        "save_report", {"title": "T", "content": "a [1]", "tickers": ["AAPL"]}
    )
    assert "(5 characters, 1 filing citation)" in one


def test_long_string_arguments_are_cut_in_the_two_streamed_events_only(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat(*research_script(world, size=5_000))
    events = ask(db, users["analyst"], chat_session)
    apple_id, nvda_id = both_ids(world)
    full = save_call(nvda_id, apple_id, size=5_000)[1]

    step = next(e for e in events if e["type"] == "step" and e["tool"] == "save_report")
    approval = next(e for e in events if e["type"] == "approval_required")
    for event in (step, approval):
        assert event["args"]["content"] == full["content"][:200] + "..."
        assert len(event["args"]["content"]) == 203
        assert event["args"]["title"] == TITLE  # short values stay as they are
        assert event["args"]["tickers"] == ["AAPL", "NVDA"]
    # The search query of the other steps is short and untouched
    first_step = next(e for e in events if e["type"] == "step")
    assert first_step["args"] == SEARCH_APPLE[1]

    # Everything stored is complete
    run = only_run(db)
    pending = tool_calls_of(db, run)[-1]
    assert pending.input == full
    detail = agent_run.get_run_detail(db, users["analyst"].org_id, users["analyst"].id, run.id)
    assert detail.pending_approval.args["content"] == full["content"]
    with agent_run.open_checkpointer() as saver:
        checkpoint = build_graph(saver).get_state({"configurable": {"thread_id": str(run.id)}})
    assert checkpoint.tasks[0].interrupts[0].value["content"] == full["content"].strip()


# ---------- approve ----------


def test_approving_saves_the_report_with_citations_from_earlier_searches_and_data_sources(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    analyst = users["analyst"]
    _model, run, approval = pause_on_report(
        db, world, users, chat_session, script_chat, "Saved. It is on the Reports page."
    )
    placeholder_id = run.answer_message_id
    apple_id, nvda_id = both_ids(world)

    events = decide(db, analyst, run, approval["tool_call_id"])

    assert events[0] == {"type": "decision", "decision": "approve", "tool": "save_report"}
    assert events[-1]["status"] == "completed"
    (report,) = reports_of(db)
    assert (report.org_id, report.user_id, report.agent_run_id) == (
        analyst.org_id,
        analyst.id,
        run.id,
    )
    assert report.title == TITLE
    # The model cited nvda first, then apple: renumbered by first appearance
    assert report.content.startswith(
        "## Summary\n\nStatement 0 from the filings [1]. Statement 1 from the filings [2]."
    )
    assert str(nvda_id) not in report.content.split("\n")[2]
    companies = report_repository.list_companies(db, analyst.org_id, report.id)
    assert [company.ticker for company in companies] == ["AAPL", "NVDA"]
    citations = report_repository.list_citations(db, analyst.org_id, report.id)
    assert [(c.number, c.chunk_id, c.ticker) for c in citations] == [
        (1, nvda_id, "NVDA"),
        (2, apple_id, "AAPL"),
    ]
    assert citations[0].content == EXPORT_TEXT
    assert all(c.score == pytest.approx(1.0, abs=1e-3) for c in citations)
    # Only the successful, non-search calls, each once (the failing MSFT call is not a source)
    assert report.data_sources == [{"tool": "get_financials", "args": NVDA_REVENUE[1]}]

    # The decision and the report are in the audit log; the pending row became approved
    assert audit_actions(db) == ["agent.approve", "report.create"]
    rows = tool_calls_of(db, run)
    assert rows[-1].tool_name == "save_report" and rows[-1].approval_status == "approved"
    assert json.loads(rows[-1].output) == {
        "status": "created",
        "report_id": report.id,
        "title": TITLE,
    }
    # The final answer replaced the placeholder
    run = only_run(db)
    assert (run.status, run.step_count) == ("completed", 5)
    assert run.answer_message_id == placeholder_id
    assert messages_of(db, chat_session)[-1].content == "Saved. It is on the Reports page."


def test_citations_come_from_searches_before_and_after_an_earlier_pause(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    # A run can propose an alert and then a report: two pauses. The first search is before the
    # first pause, the second after it; the report cites both
    analyst = users["analyst"]
    apple_id, nvda_id = both_ids(world)
    script_chat(
        tool_calls_message(SEARCH_APPLE),
        tool_calls_message(ALERT_CALL),
        tool_calls_message(SEARCH_NVDA),
        tool_calls_message(save_call(apple_id, nvda_id)),
        "Both done.",
    )

    events = ask(db, analyst, chat_session)
    first = next(e for e in events if e["type"] == "approval_required")
    assert first["tool"] == "create_alert"
    run = only_run(db)
    events = decide(db, analyst, run, first["tool_call_id"])
    second = next(e for e in events if e["type"] == "approval_required")
    assert second["tool"] == "save_report"
    assert events[-1]["status"] == "waiting_approval"
    assert reports_of(db) == [] and count_rows(db, Alert) == 1

    events = decide(db, analyst, only_run(db), second["tool_call_id"])

    assert events[-1]["status"] == "completed"
    (report,) = reports_of(db)
    citations = report_repository.list_citations(db, analyst.org_id, report.id)
    assert [(c.number, c.chunk_id) for c in citations] == [(1, apple_id), (2, nvda_id)]
    # The alert tool is not a data source
    assert report.data_sources == []
    assert audit_actions(db) == ["agent.approve", "agent.approve", "report.create"]


def test_a_second_write_call_in_the_same_step_is_turned_away(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    analyst = users["analyst"]
    apple_id, nvda_id = both_ids(world)
    script_chat(
        tool_calls_message(SEARCH_APPLE, SEARCH_NVDA),
        tool_calls_message(save_call(apple_id, nvda_id), ALERT_CALL),
        "Only the report.",
    )

    events = ask(db, analyst, chat_session)
    approval = next(e for e in events if e["type"] == "approval_required")
    assert approval["tool"] == "save_report"
    run = only_run(db)
    assert [(c.tool_name, c.approval_status) for c in tool_calls_of(db, run)[-1:]] == [
        ("save_report", "pending")
    ]

    events = decide(db, analyst, run, approval["tool_call_id"])

    assert events[-1]["status"] == "completed"
    assert len(reports_of(db)) == 1
    assert count_rows(db, Alert) == 0
    turned_away = next(row for row in tool_calls_of(db, run) if row.tool_name == "create_alert")
    assert (
        turned_away.is_error and "Only one action can be proposed at a time" in turned_away.output
    )


def test_save_report_after_a_create_alert_in_the_same_step_is_turned_away(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    analyst = users["analyst"]
    apple_id, nvda_id = both_ids(world)
    script_chat(
        tool_calls_message(SEARCH_APPLE, SEARCH_NVDA),
        tool_calls_message(ALERT_CALL, save_call(apple_id, nvda_id)),
        "Only the alert.",
    )

    events = ask(db, analyst, chat_session)
    approval = next(e for e in events if e["type"] == "approval_required")
    assert approval["tool"] == "create_alert"
    run = only_run(db)

    decide(db, analyst, run, approval["tool_call_id"])

    assert reports_of(db) == []
    assert count_rows(db, Alert) == 1
    turned_away = next(row for row in tool_calls_of(db, run) if row.tool_name == "save_report")
    assert (
        turned_away.is_error and "Only one action can be proposed at a time" in turned_away.output
    )


# ---------- reject ----------


def test_rejecting_saves_nothing_and_the_model_is_told_not_to_retry(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    analyst = users["analyst"]
    model, run, approval = pause_on_report(
        db, world, users, chat_session, script_chat, "Understood, nothing was saved."
    )

    events = decide(db, analyst, run, approval["tool_call_id"], "reject")

    assert events[-1]["status"] == "completed"
    assert reports_of(db) == []
    assert count_rows(db, ReportCitation) == 0 and count_rows(db, ReportCompany) == 0
    rows = tool_calls_of(db, run)
    assert rows[-1].approval_status == "rejected"
    assert json.loads(rows[-1].output)["status"] == "rejected"
    assert audit_actions(db) == ["agent.reject"]
    assert "Do not retry" in model.prompt_text(4)


# ---------- the role ----------


def test_a_viewers_run_is_refused_before_any_pause(
    db: Session,
    world: dict,
    users: dict[str, User],
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    viewer = users["viewer"]
    session = chat.create_session(db, viewer.org_id, viewer.id)
    script_chat(*research_script(world), "I cannot save reports for your role.")

    events = ask(db, viewer, session)

    assert not any(event["type"] == "approval_required" for event in events)
    assert events[-1]["status"] == "completed"
    run = only_run(db)
    rows = tool_calls_of(db, run)
    refused = rows[-1]
    assert (refused.tool_name, refused.is_error, refused.approval_status) == (
        "save_report",
        True,
        "not_required",
    )
    assert refused.output == "Your role cannot save reports"
    assert not any(row.approval_status == "pending" for row in rows)
    assert reports_of(db) == []
    assert audit_actions(db) == []


def test_a_role_changed_while_the_approval_waits_saves_nothing(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    analyst = users["analyst"]
    _model, run, approval = pause_on_report(
        db, world, users, chat_session, script_chat, "I could not save it."
    )
    analyst.role = "viewer"
    db.flush()

    events = decide(db, analyst, run, approval["tool_call_id"])

    assert events[-1]["status"] == "completed"
    assert reports_of(db) == []
    saved = tool_calls_of(db, run)[-1]
    assert (saved.approval_status, saved.is_error) == ("approved", True)
    assert saved.output == "Your role cannot save reports"
    assert audit_actions(db) == ["agent.approve"]


def test_a_chunk_replaced_between_the_approval_and_the_save_refuses_the_save(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    analyst = users["analyst"]
    _model, run, approval = pause_on_report(
        db, world, users, chat_session, script_chat, "I could not save it."
    )
    gone = world["chunks"]["aapl_risk"]
    gone_id = gone.id
    db.delete(gone)
    db.flush()

    decide(db, analyst, run, approval["tool_call_id"])

    assert reports_of(db) == []
    saved = tool_calls_of(db, run)[-1]
    assert saved.is_error and f"[{gone_id}]" in saved.output and "no longer exist" in saved.output


# ---------- refusals before the pause ----------


def test_invalid_markers_are_refused_before_the_pause_and_the_message_lists_them(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    apple_id, nvda_id = both_ids(world)
    never_searched = world["chunks"]["nvda_mdna"].id  # exists, but no search returned it
    bad_content = (
        f"## Summary\n\nFine [{apple_id}]. Not searched [{never_searched}]. FY [2025]. "
        + ("Plain text line.\n" * 40)
    )
    script_chat(
        tool_calls_message(SEARCH_APPLE),
        tool_calls_message(
            ("save_report", {"title": TITLE, "content": bad_content, "tickers": ["AAPL"]})
        ),
        tool_calls_message(save_call(apple_id, tickers=("AAPL",))),
    )

    events = ask(db, users["analyst"], chat_session)

    # The refusal is a tool error row (no pause); the model fixed it and the second call paused
    run = only_run(db)
    rows = tool_calls_of(db, run)
    assert [(r.tool_name, r.is_error, r.approval_status) for r in rows] == [
        ("search_filings", False, "not_required"),
        ("save_report", True, "not_required"),
        ("save_report", False, "pending"),
    ]
    assert f"[{never_searched}, 2025]" in rows[1].output
    assert str(apple_id) not in rows[1].output
    assert [e["type"] for e in events if e["type"] == "approval_required"] == ["approval_required"]
    assert reports_of(db) == []


def test_a_report_without_citations_is_refused_before_the_pause(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    content = "## Summary\n\n" + "A claim without any citation. " * 20
    script_chat(
        tool_calls_message(SEARCH_APPLE),
        tool_calls_message(
            ("save_report", {"title": TITLE, "content": content, "tickers": ["AAPL"]})
        ),
        "I could not save it.",
    )

    events = ask(db, users["analyst"], chat_session)

    assert events[-1]["status"] == "completed"
    run = only_run(db)
    refused = tool_calls_of(db, run)[-1]
    assert refused.is_error and "no citations" in refused.output
    assert not any(row.approval_status == "pending" for row in tool_calls_of(db, run))


def test_a_search_in_the_same_step_is_not_yet_known_so_its_ids_are_refused(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    apple_id, _nvda_id = both_ids(world)
    script_chat(
        tool_calls_message(SEARCH_APPLE, save_call(apple_id, tickers=("AAPL",))),
        "I could not save it.",
    )

    ask(db, users["analyst"], chat_session)

    refused = tool_calls_of(db, only_run(db))[-1]
    assert refused.tool_name == "save_report" and refused.is_error
    assert f"[{apple_id}]" in refused.output
    assert reports_of(db) == []


def test_an_out_of_scope_ticker_is_refused_by_the_tool(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    apple_id, _ = both_ids(world)
    script_chat(
        tool_calls_message(SEARCH_APPLE),
        tool_calls_message(save_call(apple_id, tickers=("AAPL", "MSFT"))),
        "No.",
    )
    ask(db, users["analyst"], chat_session)
    refused = tool_calls_of(db, only_run(db))[-1]
    assert refused.is_error and "Ticker MSFT is not available" in refused.output


# ---------- the safety tests ----------


def test_the_tool_called_outside_a_graph_never_writes(
    db: Session, world: dict, users: dict[str, User]
) -> None:
    analyst = users["analyst"]
    apple_id, _ = both_ids(world)
    title, args = save_call(apple_id, tickers=("AAPL",))
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "search_filings", "args": {}, "id": "s1"}]),
            ToolMessage(
                content=json.dumps(
                    {"results": [{"chunk_id": apple_id, "similarity": 0.9, "text": APPLE_TEXT}]}
                ),
                name="search_filings",
                tool_call_id="s1",
            ),
            AIMessage(content="", tool_calls=[{"name": title, "args": args, "id": "c1"}]),
        ]
    }

    with pytest.raises(RuntimeError):
        save_report.func(
            **args,
            state=state,
            tool_call_id="c1",
            config={
                "configurable": {"org_id": analyst.org_id, "user_id": analyst.id, "thread_id": "1"}
            },
        )

    assert reports_of(db) == []
    assert audit_actions(db) == []


@pytest.mark.parametrize("value", [True, "yes", "APPROVE", "approved", {"approved": True}, 1, ""])
def test_a_resume_value_that_is_not_exactly_approve_saves_nothing(
    db: Session,
    world: dict,
    users: dict[str, User],
    script_chat: Callable[..., ScriptedChatModel],
    value: object,
) -> None:
    analyst = users["analyst"]
    script_chat(*research_script(world), "ok")
    graph = build_graph(InMemorySaver())
    config = {
        "configurable": {
            "org_id": analyst.org_id,
            "user_id": analyst.id,
            "thread_id": "99",
            "today": "2026-10-04",
        }
    }
    graph.invoke({"messages": [HumanMessage(QUESTION)]}, config)
    assert graph.get_state(config).next == ("tools",)

    graph.invoke(Command(resume=value), config)

    assert reports_of(db) == []
    assert (
        json.loads(graph.get_state(config).values["messages"][-2].content)["status"] == "rejected"
    )


def test_org_and_user_are_not_arguments_the_model_can_set() -> None:
    schema = save_report.tool_call_schema.model_json_schema()

    assert sorted(schema["properties"]) == ["content", "tickers", "title"]
    assert [tool.name for tool in WRITE_TOOLS] == ["create_alert", "save_report"]
    assert save_report.handle_tool_error is True


def test_the_model_gets_seven_tools_and_the_prompt_describes_reports(
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    model = script_chat("Hello.")
    graph = build_graph(InMemorySaver())
    graph.invoke(
        {"messages": [HumanMessage("hi")]},
        {"configurable": {"org_id": 1, "user_id": 1, "thread_id": "t", "today": "2026-10-04"}},
    )

    assert len(model.bound_tools) == 7 and "save_report" in model.bound_tools
    system = agent_graph.AGENT_PROMPT.messages[0].prompt.template
    for rule in [
        "exactly five tools",
        "save_report ALONE",
        "Never retry a rejected report",
        "600 to 1,500 words",
        "THIS run",
        "not found in the filings",
        "the Reports page",
        "role cannot save reports",
    ]:
        assert rule in system, rule


def test_org_and_user_in_the_model_arguments_are_ignored(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    analyst = users["analyst"]
    apple_id, _ = both_ids(world)
    title, args = save_call(apple_id, tickers=("AAPL",))
    script_chat(
        tool_calls_message(SEARCH_APPLE),
        tool_calls_message((title, {**args, "org_id": users["outsider"].org_id, "user_id": 999})),
        "Saved.",
    )
    events = ask(db, analyst, chat_session)
    approval = next(e for e in events if e["type"] == "approval_required")
    decide(db, analyst, only_run(db), approval["tool_call_id"])

    (report,) = reports_of(db)
    assert (report.org_id, report.user_id) == (analyst.org_id, analyst.id)


# ---------- a real Postgres checkpointer ----------


def test_a_real_postgres_checkpointer_keeps_a_report_pause_across_a_restart(
    db: Session,
    world: dict,
    users: dict[str, User],
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pause goes through the real PostgresSaver (a 19,000 character report), the saver is
    # closed, and the decision opens a new one, like a new process
    monkeypatch.setattr(agent_run, "open_checkpointer", real_open_checkpointer)
    analyst = users["analyst"]
    script_chat(*research_script(world, size=19_000), "Saved.")
    try:
        events = ask(db, analyst, chat_session)
        approval = next(e for e in events if e["type"] == "approval_required")
        run = only_run(db)
        assert reports_of(db) == []
        with real_open_checkpointer() as saver:
            state = build_graph(saver).get_state({"configurable": {"thread_id": str(run.id)}})
            assert state.next == ("tools",)
            payload = state.tasks[0].interrupts[0].value
            assert len(payload["content"]) <= 19_000 and payload["tickers"] == ["AAPL", "NVDA"]

        events = decide(db, analyst, run, approval["tool_call_id"])

        assert events[-1]["status"] == "completed"
        (report,) = reports_of(db)
        assert len(report.content) > 18_000
        assert len(tool_calls_of(db, run)[-1].input["content"]) == 19_000
        assert audit_actions(db) == ["agent.approve", "report.create"]
    finally:
        with real_open_checkpointer() as saver:
            saver.delete_thread(str(only_run(db).id))
