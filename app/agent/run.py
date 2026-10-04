import json
import logging
import re
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
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.agent.graph import (
    CANCELLED_ANSWER,
    FAILED_ANSWER,
    STEP_LIMIT_ANSWER,
    TIMEOUT_ANSWER,
    build_graph,
)
from app.config import settings
from app.database import SessionLocal
from app.exceptions import ConflictError, NotFoundError, ServiceUnavailableError
from app.models.agent import AgentRunStatus, ApprovalStatus, ToolCall
from app.models.chat import Citation
from app.rag import chat as rag_chat
from app.rag import llm
from app.repositories import agent as agent_repository
from app.repositories import chat as chat_repository
from app.repositories import chunks as chunk_repository
from app.schemas.agent import AgentRunResponse, ToolCallResponse

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
            },
            # 2 supersteps per model call (model node + tools node). This limit allows exactly
            # AGENT_MAX_STEPS model calls and the final answer on the last of them
            "recursion_limit": 2 * settings.AGENT_MAX_STEPS,
        }

        status = None
        error = None
        step = 0
        final_text = ""
        requested: dict[str, tuple[int, str, dict]] = {}  # tool_call_id -> (step, tool, args)
        search_similarity: dict[int, float] = {}  # chunk_id -> similarity, from search_filings
        tools_started = 0.0
        deadline = time.monotonic() + settings.AGENT_TIMEOUT_SECONDS

        try:
            # 5. Run the graph and follow its events. The deadline is checked between events: it
            # cannot interrupt one long model call or tools step, those are bounded by their own
            # timeouts
            with open_checkpointer() as checkpointer:
                graph = build_graph(checkpointer)
                stream = graph.stream(
                    {"messages": messages}, config, stream_mode=["updates", "messages"]
                )
                try:
                    for event_mode, data in stream:
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
                                if node == "agent":
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
                                            "args": call["args"],
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
                                            for found in json.loads(output).get("results", []):
                                                similarity = max(
                                                    found["similarity"],
                                                    search_similarity.get(found["chunk_id"], 0.0),
                                                )
                                                search_similarity[found["chunk_id"]] = similarity
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
                                        agent_repository.create_tool_calls(short_db, rows)
                                        saved_run = agent_repository.get_run(
                                            short_db, org_id, user_id, run_id
                                        )
                                        agent_repository.update_run(
                                            short_db, saved_run, step_count=step
                                        )
                                        short_db.commit()
                                    yield from results

                        # Checked after the event, and not while tool results are still on their way
                        # (their ToolMessages arrive before the tools update), so a finished tools
                        # step is always saved. A run that has its final answer is never a timeout
                        if time.monotonic() > deadline and not final_text and not requested:
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
            # The client left: the generator was closed. Nothing may be yielded from here on
            status = AgentRunStatus.cancelled
            raise
        except Exception as exc:
            # Any other exception is a bug and propagates, but the run is closed below
            status = AgentRunStatus.failed
            error = type(exc).__name__
            logger.error("agent run %s crashed: %s", run_id, error)
            raise
        finally:
            # 6. Close the run in ONE short transaction: the assistant message, the citations
            # (completed runs only), the final status. This also runs when the client left
            if status is None:
                status = AgentRunStatus.cancelled
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
                    # call of THIS run returned and that still exists. Valid ids are renumbered
                    # 1..k in order of first appearance. Other markers stay as they are
                    pattern = r"\[(\d+(?:\s*,\s*\d+)*)\]"
                    mentioned = {
                        int(id_text)
                        for group in re.findall(pattern, final_text)
                        for id_text in group.split(",")
                        if int(id_text) in search_similarity
                    }
                    found_chunks = {}
                    if mentioned:
                        for chunk, chunk_ticker in chunk_repository.list_by_ids(
                            short_db, list(mentioned)
                        ):
                            found_chunks[chunk.id] = (chunk, chunk_ticker)

                    numbers = {}  # chunk_id -> new number
                    pieces = []
                    ignored_markers = 0
                    position = 0
                    for match in re.finditer(pattern, final_text):
                        pieces.append(final_text[position : match.start()])
                        parts = []
                        for id_text in match.group(1).split(","):
                            chunk_id = int(id_text)
                            if chunk_id in found_chunks:
                                if chunk_id not in numbers:
                                    numbers[chunk_id] = len(numbers) + 1
                                parts.append(str(numbers[chunk_id]))
                            else:
                                ignored_markers += 1
                                parts.append(id_text.strip())
                        pieces.append("[" + ", ".join(parts) + "]")
                        position = match.end()
                    pieces.append(final_text[position:])
                    answer_text = "".join(pieces)
                    if ignored_markers:
                        logger.info(
                            "agent run %s: %d markers are not valid chunk ids",
                            run_id,
                            ignored_markers,
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
                    agent_repository.create_tool_calls(short_db, unfinished)

                assistant_message = chat_repository.create_message(
                    short_db,
                    session_id,
                    "assistant",
                    answer_text,
                    model=settings.CHAT_MODEL if status == AgentRunStatus.completed else None,
                )
                for citation in citations:
                    citation.message_id = assistant_message.id
                chat_repository.create_citations(short_db, citations)
                saved_run = agent_repository.get_run(short_db, org_id, user_id, run_id)
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
        if status in (AgentRunStatus.step_limit, AgentRunStatus.timeout):
            yield {"type": "token", "text": answer_text}
        yield {
            "type": "done",
            "message_id": assistant_message_id,
            "run_id": run_id,
            "status": status,
            "cited_numbers": cited_numbers,
        }

    return events()


def get_run_detail(db: Session, org_id: int, user_id: int, run_id: int) -> AgentRunResponse:
    # A missing run, a colleague's and another organization's give the same 404
    run = agent_repository.get_run(db, org_id, user_id, run_id)
    if run is None:
        raise NotFoundError("Agent run not found")

    # The run was loaded through the scoped query, so reading its tool calls by id is safe
    tool_calls = agent_repository.list_tool_calls(db, run.id)
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
    )
