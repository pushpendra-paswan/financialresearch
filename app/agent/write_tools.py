import json
import logging
from decimal import Decimal
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, ToolException, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt
from pydantic import ValidationError

from app.agent.tools import check_ticker, require_context
from app.database import SessionLocal
from app.exceptions import ConflictError, ForbiddenError, NotFoundError
from app.schemas.alerts import AlertCreate
from app.services import alerts as alert_service
from app.services import reports as report_service

logger = logging.getLogger(__name__)

# The WRITE tools of the research agent. A write tool never executes on its own: it calls
# interrupt(), which pauses the run (the graph state is saved by the checkpointer), and it acts
# only when the run is resumed with exactly "approve". Because the tools node restarts from its
# beginning on resume, everything BEFORE interrupt() runs twice and must be safe to repeat: here
# that is only reading. org_id and user_id come from config["configurable"], never from model
# arguments.


# The text the user reads before approving. It is generated from the tool arguments by CODE, never
# written by the model, and this one function serves both the interrupt payload and the API
# response (GET /agent/runs/{id}), so the user sees the same words everywhere
def summarize_write_call(name: str, args: dict) -> str:
    if name == "save_report":
        # The same normalization as the tool: stripped title and content, distinct uppercase
        # tickers. The citation count is the number of distinct passages cited in the text
        tickers = ", ".join(
            dict.fromkeys(str(ticker).strip().upper() for ticker in args["tickers"])
        )
        content = str(args["content"]).strip()
        citation_count = len(report_service.find_marker_ids(content))
        citations = f"{citation_count} filing citation{'s' if citation_count != 1 else ''}"
        return (
            f"Save a report titled '{str(args['title']).strip()}' about {tickers} "
            f"({len(content):,} characters, {citations}). It will be visible to everyone in your "
            "organization. Bracketed numbers in the text are filing passage ids; they become [1], "
            "[2], ... when the report is saved."
        )
    if name != "create_alert":
        raise ValueError(f"No summary for the write tool {name}")

    ticker = str(args["ticker"]).strip().upper()
    # The threshold as the exact number the alert will get: at least 2 and at most 4 decimals
    threshold = f"{Decimal(str(args['threshold'])):.4f}"
    while threshold.endswith("0") and threshold[-3] != ".":
        threshold = threshold[:-1]

    if args["alert_type"] == "price_above":
        return (
            f"Create a price alert: notify you when {ticker}'s close rises above {threshold}. "
            "It fires when the price crosses the level; if the close is already above it, "
            "nothing fires until the next crossing."
        )
    if args["alert_type"] == "price_below":
        return (
            f"Create a price alert: notify you when {ticker}'s close falls below {threshold}. "
            "It fires when the price crosses the level; if the close is already below it, "
            "nothing fires until the next crossing."
        )
    return (
        f"Create a daily change alert: notify you when {ticker}'s close moves by at least "
        f"{threshold}% in one day versus the previous close, up or down."
    )


# Only the FIRST write call of the model's step is proposed. Several interrupts in one tools node
# run in parallel threads and their resume values are matched by order, which is not guaranteed, so
# a second write call (of any write tool) is turned away with a message
def check_first_write_call(state: dict, tool_call_id: str) -> None:
    last_ai = next(
        (message for message in reversed(state["messages"]) if isinstance(message, AIMessage)),
        None,
    )
    if last_ai is None:
        raise ValueError("A write tool needs the model's message in the graph state")
    write_names = [write_tool.name for write_tool in WRITE_TOOLS]
    write_call_ids = [call["id"] for call in last_ai.tool_calls if call["name"] in write_names]
    if write_call_ids[0] != tool_call_id:
        raise ToolException(
            "Only one action can be proposed at a time, so this one was NOT proposed. Wait for "
            "the user's decision on the first one, then propose this again if it is still wanted."
        )


@tool
def create_alert(
    ticker: str,
    alert_type: Literal["price_above", "price_below", "daily_change_pct"],
    threshold: float,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> str:
    """Propose a personal price alert for the user. This does NOT create it: the user is shown
    what would be created and must approve it first, so call this only when the user asked for an
    alert in their own message. alert_type is price_above (the close rises above the threshold, a
    price), price_below (the close falls below it, a price) or daily_change_pct (the close moves
    by at least the threshold percent in one day). threshold is a number greater than 0 with at
    most 4 decimals. Propose one alert at a time. If the result says the user rejected it, do not
    propose it again."""
    # 1. Context and scope
    org_id, user_id = require_context(config)
    ticker = check_ticker(ticker)

    # 2. Pre-check: the user is never asked to approve something that cannot happen. Reading only,
    # so it is safe when the node runs again after the approval
    try:
        data = AlertCreate(ticker=ticker, alert_type=alert_type, threshold=str(threshold))
    except ValidationError:
        raise ToolException(
            "Invalid alert: threshold must be a number greater than 0 with at most 4 decimals"
        ) from None
    with SessionLocal() as db:
        try:
            alert_service.check_can_create(db, org_id, user_id, data)
        except (NotFoundError, ConflictError) as exc:
            raise ToolException(exc.message) from None

    # 3. One write call per step
    check_first_write_call(state, tool_call_id)

    # 4. Pause. Only strings and floats go in the payload: it is saved by the checkpointer and
    # sent to the browser as JSON
    arguments = {
        "ticker": ticker,
        "alert_type": data.alert_type.value,
        "threshold": float(data.threshold),
    }
    decision = interrupt(
        {
            "action": "create_alert",
            **arguments,
            "summary": summarize_write_call("create_alert", arguments),
        }
    )

    # 5. Resume. Only the exact string "approve" executes; anything else (reject, a wrong type, a
    # missing value) does nothing
    if not isinstance(decision, str) or decision != "approve":
        return json.dumps(
            {
                "status": "rejected",
                "message": "The user rejected this action. Do not retry it.",
            }
        )
    with SessionLocal() as db:
        try:
            created = alert_service.create_alert(db, org_id, user_id, data)
        except (NotFoundError, ConflictError) as exc:
            # Something changed between the proposal and the approval
            raise ToolException(exc.message) from None
    return json.dumps({"status": "created", "alert": created.model_dump(mode="json")})


@tool
def save_report(
    title: str,
    content: str,
    tickers: list[str],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> str:
    """Propose to save a research report for the user's organization. This does NOT save it: the
    user is shown the whole text and must approve it first, so call this only when the user asked
    for a report in their own message, ALONE (never together with other tool calls), once, with the
    complete report. title is at most 200 characters. content is the whole report in Markdown (only
    # ## ### headings, paragraphs, '- ' bullets, pipe tables and **bold**; no links, images, HTML or
    code blocks), 300 to 20000 characters, and every statement taken from filing text must be
    followed by the chunk_id of a search_filings result of THIS run in square brackets, for
    example [1780]. tickers lists the 1 to 5 companies the report covers. If the result says the
    report cannot be saved, fix what it names and call this again; if the user rejected it, do not
    propose it again."""
    # 1. Context and scope
    org_id, user_id = require_context(config)
    tickers = [check_ticker(ticker) for ticker in tickers]
    thread_id = (config.get("configurable") or {}).get("thread_id")
    if thread_id is None:
        raise ValueError("save_report needs the run id (thread_id) in config['configurable']")
    run_id = int(thread_id)

    # 2. What this run found, from the graph state: the passages of every search_filings call so
    # far ({chunk_id: best similarity}) and the successful data tool calls ({tool, args}, each
    # distinct call once). The state holds all earlier tool messages, also after a pause. A search
    # in the SAME model step is not in it yet, which is why this tool must be called alone
    args_by_call_id = {
        call["id"]: (call["name"], call["args"])
        for message in state["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    write_names = [write_tool.name for write_tool in WRITE_TOOLS]
    found_chunks: dict[int, float] = {}
    data_sources: list[dict] = []
    for message in state["messages"]:
        if not isinstance(message, ToolMessage) or message.status != "success":
            continue
        called_name, called_args = args_by_call_id[message.tool_call_id]
        if called_name == "search_filings":
            for found in json.loads(message.text).get("results", []):
                similarity = max(found["similarity"], found_chunks.get(found["chunk_id"], 0.0))
                found_chunks[found["chunk_id"]] = similarity
        elif called_name not in write_names:
            source = {"tool": called_name, "args": called_args}
            if source not in data_sources:
                data_sources.append(source)

    # 3. Pre-check: nothing is asked of the user when the report could not be saved anyway.
    # Reading only, so it is safe when the node runs again after the approval
    with SessionLocal() as db:
        try:
            title, content, companies, _chunks = report_service.check_can_create(
                db, org_id, user_id, title, content, tickers, found_chunks
            )
        except (NotFoundError, ConflictError, ForbiddenError) as exc:
            raise ToolException(exc.message) from None

    # 4. One write call per step
    check_first_write_call(state, tool_call_id)

    # 5. Pause. The payload holds the whole text, because the user approves exactly what is saved
    arguments = {"title": title, "content": content, "tickers": [c.ticker for c in companies]}
    decision = interrupt(
        {
            "action": "save_report",
            **arguments,
            "summary": summarize_write_call("save_report", arguments),
        }
    )

    # 6. Resume. Only the exact string "approve" saves; the service checks everything again
    if not isinstance(decision, str) or decision != "approve":
        return json.dumps(
            {
                "status": "rejected",
                "message": "The user rejected this action. Do not retry it.",
            }
        )
    with SessionLocal() as db:
        try:
            report = report_service.create_report(
                db,
                org_id,
                user_id,
                run_id,
                title,
                content,
                arguments["tickers"],
                found_chunks,
                data_sources,
            )
        except (NotFoundError, ConflictError, ForbiddenError) as exc:
            # Something changed between the proposal and the approval (a role, a passage)
            raise ToolException(exc.message) from None
    return json.dumps({"status": "created", "report_id": report.id, "title": report.title})


WRITE_TOOLS = [create_alert, save_report]

for write_tool in WRITE_TOOLS:
    write_tool.handle_tool_error = True
