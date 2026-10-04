import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from typing import Literal

import openai
import psycopg
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.agent.graph import (
    APPROVAL_PENDING_ANSWER,
    CANCELLED_ANSWER,
    EXPIRED_ANSWER,
    FAILED_ANSWER,
    STEP_LIMIT_ANSWER,
    TIMEOUT_ANSWER,
    build_graph,
)
from app.agent.write_tools import summarize_write_call
from app.config import settings
from app.database import SessionLocal
from app.exceptions import ConflictError, NotFoundError, ServiceUnavailableError
from app.models.agent import AgentRunStatus, ApprovalStatus, ToolCall
from app.models.chat import Citation
from app.rag import chat as rag_chat
from app.rag import llm
from app.repositories import agent as agent_repository
from app.repositories import audit as audit_repository
from app.repositories import chat as chat_repository
from app.repositories import chunks as chunk_repository
from app.schemas.agent import AgentRunResponse, PendingApproval, ToolCallResponse
from app.services import reports as report_service

logger = logging.getLogger(__name__)

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide how a question about Apple (AAPL) and NVIDIA (NVDA) is answered.\n"
            "- rag: the question asks what the filings of ONE company say: its risks, its "
            "business, the wording or the explanations of its MD&A, or one fact that sits in one "
            "passage of a filing.\n"
            "- agent: the question needs numbers, prices or financial figures, a calculation, a "
            "comparison of companies, or several steps, or it combines filings with data.\n"
            "Use the chat history only to resolve references such as 'it' or 'that company'. "
            "When you are unsure, choose agent.",
        ),
        ("human", "Chat history:\n{history}\n\nQuestion: {question}"),
    ]
)


class RouteDecision(BaseModel):
    route: Literal["rag", "agent"]


# Chooses between the simple filings chat (2.4) and the agent with one structured-output call. A
# failure falls back to "rag", the cheaper path, so a router problem never blocks a question.
# Used by ask_question and by scripts/try_agent.py --route-samples
def route_question(question: str, history: list) -> str:
    history_text = (
        "\n".join(
            f"{'User' if message.role == 'user' else 'Assistant'}: {message.content}"
            for message in history
        )
        or "(none)"
    )
    chain = ROUTER_PROMPT | llm.get_chat_model().with_structured_output(RouteDecision)
    try:
        decision = chain.invoke({"history": history_text, "question": question})
    except (openai.OpenAIError, OutputParserException) as exc:
        # Only the class name: an OpenAI error message can contain part of the key
        logger.warning("router failed, falling back to rag: %s", type(exc).__name__)
        return "rag"
    if decision is None:
        return "agent"
    return decision.route


# The checkpointer is opened once per run (about 20 ms: one extra connection). Its tables are
# created by a migration, never here. Tests replace this function with an in-memory saver
@contextmanager
def open_checkpointer() -> Iterator[PostgresSaver]:
    # psycopg wants a plain postgresql:// URL, not the SQLAlchemy "+psycopg" form
    url = settings.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://", 1)
    with PostgresSaver.from_conn_string(url) as saver:
        yield saver


# Applies a search_filings output to {chunk_id: best similarity}. Used for every tools step and,
# when a paused run is resumed, for the searches that were saved before the pause
def add_search_similarity(search_similarity: dict[int, float], output: str) -> None:
    for found in json.loads(output).get("results", []):
        similarity = max(found["similarity"], search_similarity.get(found["chunk_id"], 0.0))
        search_similarity[found["chunk_id"]] = similarity


# String arguments longer than 200 characters (a whole report) are cut for the two STREAMED events
# that show tool arguments (step, approval_required). The stored input, the checkpoint and
# pending_approval.args stay complete
def shorten_strings(args: dict) -> dict:
    return {
        key: value[:200] + "..." if isinstance(value, str) and len(value) > 200 else value
        for key, value in args.items()
    }


# The graph loop and the close of a run, shared by a first run (messages given, decision None) and a
# resumed run (messages None, decision "approve" or "reject"). It yields NDJSON events as dicts:
#   [decision,] step, step_result, token..., done  (or error; or approval_required + token + done
#   when a write tool paused the run). A paused run is neither finished nor cancelled: it is
#   saved as waiting_approval and nothing is closed
def stream_run(
    org_id: int,
    user_id: int,
    session_id: int,
    run_id: int,
    config: dict,
    messages: list[BaseMessage] | None,
    decision: str | None,
    first_event: dict | None = None,
) -> Iterator[dict]:
    status = None
    error = None
    step = 0
    final_text = ""
    requested: dict[str, tuple[int, str, dict]] = {}  # tool_call_id -> (step, tool, args)
    search_similarity: dict[int, float] = {}  # chunk_id -> similarity, from search_filings
    tools_started = 0.0
    stop_after_tools = False
    interrupt_payload = None
    approval_event = None
    # A resumed run gets its own deadline: the time spent waiting for the human does not count
    deadline = time.monotonic() + settings.AGENT_TIMEOUT_SECONDS

    try:
        # The decision event of a resumed run goes out from inside the try: a client that leaves
        # right after it still closes the run as cancelled
        if first_event is not None:
            yield first_event

        # 5. Run the graph and follow its events. The deadline is checked between events: it
        # cannot interrupt one long model call or tools step, those are bounded by their own
        # timeouts
        with open_checkpointer() as checkpointer:
            graph = build_graph(checkpointer)

            if decision is None:
                graph_input = {"messages": messages}
                # 2 supersteps per model call (model node + tools node). This limit allows exactly
                # AGENT_MAX_STEPS model calls and the final answer on the last of them
                config = {**config, "recursion_limit": 2 * settings.AGENT_MAX_STEPS}
            else:
                # The resumed stream starts with the tools node (it restarts from its beginning)
                # and replays nothing, so what the first part knew is rebuilt: the model call
                # count, the calls of the paused step, and the similarity of earlier searches
                with SessionLocal() as short_db:
                    saved_run = agent_repository.get_run(short_db, org_id, user_id, run_id)
                    step = saved_run.step_count
                    for saved_call in agent_repository.list_tool_calls(short_db, run_id):
                        if saved_call.tool_name == "search_filings" and not saved_call.is_error:
                            add_search_similarity(search_similarity, saved_call.output)
                state = graph.get_state(config)
                if not state.next:
                    raise ValueError("The graph of this run is not paused")
                for call in state.values["messages"][-1].tool_calls:
                    requested[call["id"]] = (step, call["name"], call["args"])
                graph_input = Command(resume=decision)
                # recursion_limit restarts on a resumed call, with an offset of 2: this formula
                # keeps the model calls before and after the pause within AGENT_MAX_STEPS. When
                # the last allowed call is the one that paused, the smallest limit (1) would still
                # allow one more call, so the run stops right after the tools step instead
                config = {
                    **config,
                    "recursion_limit": max(2 * (settings.AGENT_MAX_STEPS - step) - 1, 1),
                }
                stop_after_tools = step >= settings.AGENT_MAX_STEPS
                tools_started = time.monotonic()

            stream = graph.stream(graph_input, config, stream_mode=["updates", "messages"])
            try:
                for event_mode, data in stream:
                    tools_finished = False
                    if event_mode == "messages":
                        # Tokens of the model node. gpt-5.4-mini writes no text before a tool
                        # call, so all of its text is forwarded
                        chunk, metadata = data
                        if (
                            metadata.get("langgraph_node") == "agent"
                            and isinstance(chunk, AIMessageChunk)
                            and chunk.text
                        ):
                            yield {"type": "token", "text": chunk.text}
                    else:
                        for node, update in data.items():
                            if update is None:
                                # A resumed node that raised: LangGraph reports it with an empty
                                # update just before it re-raises the exception
                                continue
                            if node == "__interrupt__":
                                # A write tool paused the run. The stream ends by itself; the
                                # pause is saved in the finally block below
                                interrupt_payload = update[0].value
                                status = AgentRunStatus.waiting_approval

                            elif node == "agent":
                                ai_message = update["messages"][-1]
                                step += 1
                                if not ai_message.tool_calls:
                                    final_text = ai_message.text
                                    continue
                                for call in ai_message.tool_calls:
                                    requested[call["id"]] = (step, call["name"], call["args"])
                                    yield {
                                        "type": "step",
                                        "step": step,
                                        "tool": call["name"],
                                        "args": shorten_strings(call["args"]),
                                    }
                                # The graph is lazy: the tools node starts when the next event
                                # is requested, so this is the start of its wall time
                                tools_started = time.monotonic()

                            elif node == "tools":
                                duration_ms = int((time.monotonic() - tools_started) * 1000)
                                rows = []
                                results = []
                                for tool_message in update["messages"]:
                                    call_step, tool_name, args = requested.pop(
                                        tool_message.tool_call_id
                                    )
                                    is_error = tool_message.status == "error"
                                    output = tool_message.text
                                    if tool_name == "search_filings" and not is_error:
                                        add_search_similarity(search_similarity, output)
                                    rows.append(
                                        ToolCall(
                                            run_id=run_id,
                                            step=call_step,
                                            tool_call_id=tool_message.tool_call_id,
                                            tool_name=tool_name,
                                            input=args,
                                            output=output,
                                            is_error=is_error,
                                            approval_status=ApprovalStatus.not_required,
                                            duration_ms=duration_ms,
                                        )
                                    )
                                    results.append(
                                        {
                                            "type": "step_result",
                                            "step": call_step,
                                            "tool": tool_name,
                                            "ok": not is_error,
                                            "chars": len(output),
                                        }
                                    )
                                with SessionLocal() as short_db:
                                    # The pending row of an approved write call is updated here
                                    agent_repository.save_tool_calls(short_db, run_id, rows)
                                    saved_run = agent_repository.get_run(
                                        short_db, org_id, user_id, run_id
                                    )
                                    agent_repository.update_run(
                                        short_db, saved_run, step_count=step
                                    )
                                    short_db.commit()
                                yield from results
                                tools_finished = True

                    # The step limit of a resumed run whose paused call was the last allowed model
                    # call: its tools step is done and saved, no further model call may start
                    if stop_after_tools and tools_finished:
                        status = AgentRunStatus.step_limit
                        break
                    # Checked after the event, and not while tool results are still on their way
                    # (their ToolMessages arrive before the tools update), so a finished tools
                    # step is always saved. A run that has its final answer is never a timeout
                    if (
                        time.monotonic() > deadline
                        and not final_text
                        and not requested
                        and status is None
                    ):
                        status = AgentRunStatus.timeout
                        break
            finally:
                # Waits for a step that is still running, then stops the graph
                stream.close()

        if status is None:
            if final_text.strip():
                status = AgentRunStatus.completed
            else:
                status = AgentRunStatus.failed
                error = "EmptyAnswer"
    except GraphRecursionError:
        status = AgentRunStatus.step_limit
    except (openai.OpenAIError, SQLAlchemyError, psycopg.Error, ValueError) as exc:
        # ValueError is a bug inside a tool. Only the class name is kept: an OpenAI error
        # message can contain part of the key
        status = AgentRunStatus.failed
        error = type(exc).__name__
        logger.error("agent run %s failed: %s", run_id, error)
    except GeneratorExit:
        # The client left: the generator was closed. Nothing may be yielded from here on. A pause
        # that was already seen stays a pause (the user can still decide after a reload)
        if status != AgentRunStatus.waiting_approval:
            status = AgentRunStatus.cancelled
        raise
    except Exception as exc:
        # Any other exception is a bug and propagates, but the run is closed below
        status = AgentRunStatus.failed
        error = type(exc).__name__
        logger.error("agent run %s crashed: %s", run_id, error)
        raise
    finally:
        if status is None:
            status = AgentRunStatus.cancelled

        if status == AgentRunStatus.waiting_approval:
            # 6a. Save the pause in ONE short transaction: the pending row of the write call
            # (the other calls of the step have no result yet and are not saved: the tools node
            # runs again on resume), the placeholder answer, the run status
            call_id, (call_step, tool_name, args) = next(
                (call_id, info)
                for call_id, info in requested.items()
                if info[1] == interrupt_payload["action"]
            )
            with SessionLocal() as short_db:
                pending_row = ToolCall(
                    run_id=run_id,
                    step=call_step,
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    input=args,
                    output="Waiting for the user's decision.",
                    is_error=False,
                    approval_status=ApprovalStatus.pending,
                    duration_ms=0,
                    created_at=datetime.now(UTC),
                )
                agent_repository.save_tool_calls(short_db, run_id, [pending_row])
                saved_run = agent_repository.get_run(short_db, org_id, user_id, run_id)
                answer_message_id = saved_run.answer_message_id
                if answer_message_id is None:
                    placeholder = chat_repository.create_message(
                        short_db, session_id, "assistant", APPROVAL_PENDING_ANSWER
                    )
                    answer_message_id = placeholder.id
                agent_repository.update_run(
                    short_db,
                    saved_run,
                    status=status,
                    step_count=step,
                    answer_message_id=answer_message_id,
                )
                run_session = chat_repository.get_session(short_db, org_id, user_id, session_id)
                if run_session is not None:
                    chat_repository.touch_session(short_db, run_session)
                short_db.commit()
                assistant_message_id = answer_message_id
                answer_text = APPROVAL_PENDING_ANSWER
                approval_event = {
                    "type": "approval_required",
                    "run_id": run_id,
                    "tool_call_id": pending_row.id,
                    "tool": tool_name,
                    "args": shorten_strings(args),
                    "summary": interrupt_payload["summary"],
                    "expires_at": (
                        pending_row.created_at + timedelta(minutes=settings.APPROVAL_TTL_MINUTES)
                    ).isoformat(),
                }
            cited_numbers = []
        else:
            # 6b. Close the run in ONE short transaction: the assistant message, the citations
            # (completed runs only), the final status. This also runs when the client left
            answer_text = {
                AgentRunStatus.step_limit: STEP_LIMIT_ANSWER,
                AgentRunStatus.timeout: TIMEOUT_ANSWER,
                AgentRunStatus.failed: FAILED_ANSWER,
                AgentRunStatus.cancelled: CANCELLED_ANSWER,
            }.get(status, final_text)

            with SessionLocal() as short_db:
                citations = []
                cited_numbers = []
                if status == AgentRunStatus.completed:
                    # A marker [n] is a citation only when n is a chunk_id that a search_filings
                    # call of THIS run (before and after a pause) returned and that still exists.
                    # Valid ids are renumbered 1..k in order of first appearance. Other markers
                    # stay as they are (the marker rules are shared with reports)
                    mentioned = [
                        chunk_id
                        for chunk_id in report_service.find_marker_ids(final_text)
                        if chunk_id in search_similarity
                    ]
                    found_chunks = {}
                    if mentioned:
                        for chunk, chunk_ticker in chunk_repository.list_by_ids(
                            short_db, mentioned
                        ):
                            found_chunks[chunk.id] = (chunk, chunk_ticker)

                    answer_text, numbers, ignored_ids = report_service.renumber_markers(
                        final_text, set(found_chunks)
                    )
                    if ignored_ids:
                        logger.info(
                            "agent run %s: %d markers are not valid chunk ids",
                            run_id,
                            len(ignored_ids),
                        )
                    cited_numbers = sorted(numbers.values())
                    for chunk_id, number in numbers.items():
                        chunk, chunk_ticker = found_chunks[chunk_id]
                        citations.append(
                            Citation(
                                number=number,
                                chunk_id=chunk.id,
                                score=search_similarity[chunk_id],
                                filing_id=chunk.filing_id,
                                ticker=chunk_ticker,
                                fiscal_year=chunk.fiscal_year,
                                section=chunk.section,
                                content=chunk.content,
                            )
                        )

                if status == AgentRunStatus.failed and requested:
                    # Calls the model asked for whose tools step did not finish: keep them in the
                    # trace, marked as failed, so the partial trace shows what was attempted
                    unfinished = [
                        ToolCall(
                            run_id=run_id,
                            step=call_step,
                            tool_call_id=call_id,
                            tool_name=tool_name,
                            input=args,
                            output=f"ToolFailed: {error}",
                            is_error=True,
                            approval_status=ApprovalStatus.not_required,
                            duration_ms=0,
                        )
                        for call_id, (call_step, tool_name, args) in requested.items()
                    ]
                    agent_repository.save_tool_calls(short_db, run_id, unfinished)

                saved_run = agent_repository.get_run(short_db, org_id, user_id, run_id)
                answer_model = settings.CHAT_MODEL if status == AgentRunStatus.completed else None
                if saved_run.answer_message_id is not None:
                    # A run that was paused already has its assistant message (the placeholder):
                    # it becomes the real answer
                    assistant_message = chat_repository.get_message(
                        short_db, session_id, saved_run.answer_message_id
                    )
                    chat_repository.update_message(
                        short_db, assistant_message, answer_text, answer_model
                    )
                else:
                    assistant_message = chat_repository.create_message(
                        short_db, session_id, "assistant", answer_text, model=answer_model
                    )
                for citation in citations:
                    citation.message_id = assistant_message.id
                chat_repository.create_citations(short_db, citations)
                agent_repository.update_run(
                    short_db,
                    saved_run,
                    status=status,
                    step_count=step,
                    answer_message_id=assistant_message.id,
                    error=error,
                    finished_at=datetime.now(UTC),
                )
                run_session = chat_repository.get_session(short_db, org_id, user_id, session_id)
                if run_session is not None:
                    chat_repository.touch_session(short_db, run_session)
                short_db.commit()
                assistant_message_id = assistant_message.id

    # 7. Reached only when the run ended normally (not when the client left)
    if status == AgentRunStatus.failed:
        yield {"type": "error", "detail": "The research could not be completed. Try again."}
        return
    if status == AgentRunStatus.waiting_approval:
        yield approval_event
        yield {"type": "token", "text": answer_text}
    elif status in (AgentRunStatus.step_limit, AgentRunStatus.timeout):
        yield {"type": "token", "text": answer_text}
    yield {
        "type": "done",
        "message_id": assistant_message_id,
        "run_id": run_id,
        "status": status,
        "cited_numbers": cited_numbers,
    }


# The one entry point of the chat route. Mode "rag" is the 2.4 chat, unchanged. For "agent" and
# "auto" the checks that can fail run NOW, so they are normal JSON errors (404, 503, 409), then the
# request's database session is released, so nothing sits idle in a transaction while the agent
# works for minutes. The returned generator yields NDJSON events as dicts:
#   route, step, step_result, token..., done  (or error, when something fails)
def ask_question(
    db: Session,
    org_id: int,
    user_id: int,
    session_id: int,
    question: str,
    ticker: str | None,
    mode: str,
) -> Iterator[dict]:
    if mode == "rag":
        return rag_chat.ask(db, org_id, user_id, session_id, question, ticker)

    # 1. Eager checks. A missing session, a colleague's and another organization's give one 404
    chat_session = chat_repository.get_session(db, org_id, user_id, session_id)
    if chat_session is None:
        raise NotFoundError("Chat session not found")
    try:
        llm.get_chat_model()
    except ValueError:
        raise ServiceUnavailableError("Chat is disabled: OPENAI_API_KEY not set") from None

    # One active run per user. A run still "running" after twice the timeout is stale (the
    # process died) and ignored. This check is not safe against two simultaneous requests
    stale_before = datetime.now(UTC) - timedelta(seconds=2 * settings.AGENT_TIMEOUT_SECONDS)
    if agent_repository.get_active_run(db, org_id, user_id, stale_before) is not None:
        raise ConflictError("An agent run is already in progress")

    # Ends the read transaction and returns the connection to the pool. The generator below
    # opens a short session for each write
    db.commit()

    def events() -> Iterator[dict]:
        # 2. The last messages as plain Human / AI messages (no tool messages, no citations). They
        # are read before the new question is saved
        with SessionLocal() as short_db:
            history = chat_repository.list_recent_messages(
                short_db, session_id, settings.CHAT_HISTORY_MESSAGES
            )
        messages: list[BaseMessage] = [
            HumanMessage(row.content) if row.role == "user" else AIMessage(row.content)
            for row in history
        ]

        # 3. The route. "auto" asks the router once; "agent" skips it
        route = "agent"
        if mode == "auto":
            route = route_question(question, history)
        yield {"type": "route", "route": route, "mode": mode}
        if route == "rag":
            try:
                yield from rag_chat.ask(db, org_id, user_id, session_id, question, ticker)
            except (NotFoundError, ServiceUnavailableError) as exc:
                # The session or the key vanished after the checks above
                logger.error("chat failed after the route: %s", type(exc).__name__)
                yield {"type": "error", "detail": "The answer could not be completed. Try again."}
            return

        # 4. Save the question and create the run FIRST (the one deliberate difference from RAG
        # chat): the trace of an agent run is valuable even when the run fails
        with SessionLocal() as short_db:
            user_message = chat_repository.create_message(
                short_db, session_id, "user", question, ticker=ticker
            )
            run = agent_repository.create_run(
                short_db, org_id, user_id, session_id, user_message.id
            )
            run_session = chat_repository.get_session(short_db, org_id, user_id, session_id)
            if run_session is not None and run_session.title is None:
                run_session.title = question.strip()[:100]
            short_db.commit()
            run_id = run.id
        messages.append(HumanMessage(question))

        focus = ""
        if ticker:
            focus = f"The user focused on {ticker}; comparisons may still use both companies."
        config = {
            "configurable": {
                "org_id": org_id,
                "user_id": user_id,
                "thread_id": str(run_id),
                "today": date.today().isoformat(),
                "focus": focus,
            }
        }
        yield from stream_run(org_id, user_id, session_id, run_id, config, messages, None)

    return events()


# The entry point of POST /agent/runs/{run_id}/decision. Like ask_question, every check that can
# fail runs NOW (normal JSON errors, before any stream), then the decision is recorded atomically,
# and only then does the returned generator resume the graph. Nothing executes without a recorded
# approval: the write tool acts only when the graph is resumed with exactly "approve"
def decide_run(
    db: Session, org_id: int, user_id: int, run_id: int, tool_call_id: int, decision: str
) -> Iterator[dict]:
    # 1. The run through the scoped query: a missing run, a colleague's and another organization's
    # give the same 404. Only the owner can decide
    run = agent_repository.get_run(db, org_id, user_id, run_id)
    if run is None:
        raise NotFoundError("Agent run not found")
    try:
        llm.get_chat_model()
    except ValueError:
        raise ServiceUnavailableError("Chat is disabled: OPENAI_API_KEY not set") from None

    # 2. The decision must name the pending call of a run that is waiting for it
    pending = agent_repository.get_tool_call(db, run.id, tool_call_id)
    if (
        run.status != AgentRunStatus.waiting_approval
        or pending is None
        or pending.approval_status != ApprovalStatus.pending
    ):
        raise ConflictError("This run is not waiting for this approval")
    tool_name = pending.tool_name
    session_id = run.session_id
    user_message_id = run.message_id

    # 3. An approval expires: the first decision that arrives after the time closes it, and
    # nothing is executed (the expiry is also reported on every read of the run)
    expires_at = pending.created_at + timedelta(minutes=settings.APPROVAL_TTL_MINUTES)
    if datetime.now(UTC) > expires_at:
        closed = agent_repository.change_run_status(
            db,
            org_id,
            user_id,
            run.id,
            AgentRunStatus.waiting_approval,
            AgentRunStatus.expired,
            finished_at=datetime.now(UTC),
        )
        if closed:
            agent_repository.change_tool_call_approval(
                db, run.id, pending.id, ApprovalStatus.pending, ApprovalStatus.expired
            )
            placeholder = chat_repository.get_message(db, session_id, run.answer_message_id)
            if placeholder is not None:
                chat_repository.update_message(db, placeholder, EXPIRED_ANSWER, None)
            audit_repository.create(
                db, org_id, user_id, action="agent.expire", entity_id=pending.id
            )
            db.commit()
            raise ConflictError("This approval has expired")
        db.rollback()
        raise ConflictError("This run is not waiting for this approval")

    # 4. One active run per user: the resumed run is running again
    stale_before = datetime.now(UTC) - timedelta(seconds=2 * settings.AGENT_TIMEOUT_SECONDS)
    if agent_repository.get_active_run(db, org_id, user_id, stale_before) is not None:
        raise ConflictError("An agent run is already in progress")

    # 5. Record the decision in ONE transaction, as an atomic compare-and-set: of two simultaneous
    # decisions exactly one wins, the other gets the 409. started_at restarts, because the one
    # active run guard and the stale-run cutoff both look at it
    user_message = chat_repository.get_message(db, session_id, user_message_id)
    ticker = user_message.ticker if user_message is not None else None
    approved = decision == "approve"
    won = agent_repository.change_run_status(
        db,
        org_id,
        user_id,
        run.id,
        AgentRunStatus.waiting_approval,
        AgentRunStatus.running,
        started_at=datetime.now(UTC),
    ) and agent_repository.change_tool_call_approval(
        db,
        run.id,
        pending.id,
        ApprovalStatus.pending,
        ApprovalStatus.approved if approved else ApprovalStatus.rejected,
    )
    if not won:
        db.rollback()
        raise ConflictError("This run is not waiting for this approval")
    audit_repository.create(
        db,
        org_id,
        user_id,
        action="agent.approve" if approved else "agent.reject",
        entity_id=pending.id,
    )
    db.commit()

    focus = ""
    if ticker:
        focus = f"The user focused on {ticker}; comparisons may still use both companies."
    config = {
        "configurable": {
            "org_id": org_id,
            "user_id": user_id,
            "thread_id": str(run_id),
            "today": date.today().isoformat(),
            "focus": focus,
        }
    }

    return stream_run(
        org_id,
        user_id,
        session_id,
        run_id,
        config,
        None,
        decision,
        first_event={"type": "decision", "decision": decision, "tool": tool_name},
    )


def get_run_detail(db: Session, org_id: int, user_id: int, run_id: int) -> AgentRunResponse:
    # A missing run, a colleague's and another organization's give the same 404
    run = agent_repository.get_run(db, org_id, user_id, run_id)
    if run is None:
        raise NotFoundError("Agent run not found")

    # The run was loaded through the scoped query, so reading its tool calls by id is safe
    tool_calls = agent_repository.list_tool_calls(db, run.id)

    # The text the user must read comes from the stored arguments through the same function that
    # made the interrupt payload; the expiry is reported here even if nobody decided yet
    pending_approval = None
    if run.status == AgentRunStatus.waiting_approval:
        pending = agent_repository.get_pending_tool_call(db, run.id)
        if pending is not None:
            expires_at = pending.created_at + timedelta(minutes=settings.APPROVAL_TTL_MINUTES)
            pending_approval = PendingApproval(
                tool_call_id=pending.id,
                tool_name=pending.tool_name,
                args=pending.input,
                summary=summarize_write_call(pending.tool_name, pending.input),
                expires_at=expires_at,
                expired=datetime.now(UTC) > expires_at,
            )
    return AgentRunResponse(
        id=run.id,
        status=run.status,
        step_count=run.step_count,
        error=run.error,
        started_at=run.started_at,
        finished_at=run.finished_at,
        message_id=run.message_id,
        answer_message_id=run.answer_message_id,
        tool_calls=[ToolCallResponse.model_validate(tool_call) for tool_call in tool_calls],
        pending_approval=pending_approval,
    )
