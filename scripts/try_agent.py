# Run with: docker compose exec api python -m scripts.try_agent --email you@example.com "question"
#   [--mode agent|auto] [--ticker NVDA] [--follow-up "second question"] [--keep]
# or:       docker compose exec api python -m scripts.try_agent --email you@example.com
#             --resume <run id> --decision approve|reject        (3.3: decide a paused run)
# or:       docker compose exec api python -m scripts.try_agent --route-samples
# Creates a chat session for an EXISTING user, asks the question (and the follow-up in the same
# session) through the same function the chat route uses, prints the events as they stream (steps
# and tokens), then the stored run, its tool calls, the citations and how many messages the
# checkpointer holds for the run's thread. The session is deleted at the end unless --keep is
# given. It calls OpenAI (several chat calls per question, embeddings and Cohere for searches), so
# it needs OPENAI_API_KEY. --route-samples only calls the router for 10 built-in questions.
# A run that PAUSES for the user's approval (a write tool such as create_alert) prints "waiting for
# approval: run <id>, call <id>: <summary>", keeps its session and checkpoint, and the script exits
# normally, so a SECOND process can decide it with --resume <run id> --decision approve|reject
# (3.4: after a run that saved a report it also prints the stored report: id, title, companies,
# number of citations and data sources)
# (it finds the pending call itself, prints the stream and the stored result, and keeps the
# session unless --cleanup is given).
import argparse
import logging
import sys
import time

from sqlalchemy import func, select

from app.agent import run as agent_run
from app.agent.graph import build_graph
from app.config import settings
from app.database import SessionLocal
from app.exceptions import ConflictError, NotFoundError, ServiceUnavailableError
from app.models.agent import AgentRun, AgentRunStatus
from app.models.alerts import Alert
from app.models.reports import Report
from app.rag import chat, llm
from app.repositories import agent as agent_repository
from app.repositories import chat as chat_repository
from app.repositories import reports as report_repository
from app.repositories import users as user_repository

# INFO for our own modules (what they decided) and for httpx (one line per real OpenAI request).
# httpx logs only the URL, never the key
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
for name in ("app.agent.run", "app.agent.tools", "app.rag.chat", "httpx", "httpx2"):
    logging.getLogger(name).setLevel(logging.INFO)

ROUTE_SAMPLES = [
    ("rag", "What are the main risk factors NVIDIA reports about export controls?"),
    ("rag", "How does Apple describe its dependence on outsourcing partners in Asia?"),
    ("rag", "What does NVIDIA's MD&A say drove Data Center revenue growth?"),
    ("rag", "What does Apple's 10-K say about competition in the smartphone market?"),
    ("rag", "Which supply chain risks does NVIDIA disclose in its filings?"),
    ("agent", "What was NVIDIA's closing price trend over the last 5 days?"),
    ("agent", "Compare Apple's and NVIDIA's revenue growth and net margin."),
    ("agent", "What is Apple's net margin for the latest fiscal year?"),
    ("agent", "How did NVDA stock move over the last 90 days and what risks does it report?"),
    ("agent", "Which company has the higher return on equity, AAPL or NVDA?"),
]

parser = argparse.ArgumentParser(description="Ask the research agent a question")
parser.add_argument("question", nargs="?", help="the question")
parser.add_argument("--email", help="email of an existing user (owns the session)")
parser.add_argument("--mode", choices=["agent", "auto"], default="agent")
parser.add_argument("--ticker", help="a hint: the company the user focused on (one of RAG_TICKERS)")
parser.add_argument("--follow-up", help="a second question asked in the same session")
parser.add_argument("--keep", action="store_true", help="do not delete the session at the end")
parser.add_argument("--route-samples", action="store_true", help="print the router decisions")
parser.add_argument("--resume", type=int, help="the id of a paused run to decide (with --decision)")
parser.add_argument("--decision", choices=["approve", "reject"], help="the decision for --resume")
parser.add_argument(
    "--cleanup", action="store_true", help="with --resume: delete the session after"
)
args = parser.parse_args()

if not settings.OPENAI_API_KEY:
    print("OPENAI_API_KEY is not set: the agent is disabled")
    sys.exit(1)

if args.route_samples:
    for expected, question in ROUTE_SAMPLES:
        decision = agent_run.route_question(question, [])
        verdict = "OK" if decision == expected else "DIFF"
        print(f"{decision:5} (expected {expected:5}) {verdict:4} {question}")
    sys.exit(0)

if args.resume is not None:
    if not args.email or not args.decision:
        print("--resume needs --email and --decision")
        sys.exit(1)
elif not args.question or not args.email:
    print("Give a question and --email (or use --route-samples, or --resume)")
    sys.exit(1)
ticker = args.ticker.strip().upper() if args.ticker else None

db = SessionLocal()
try:
    user = user_repository.get_by_email(db, args.email.strip().lower())
    if user is None:
        print(f"No user with email {args.email}")
        sys.exit(1)

    if args.resume is not None:
        # Resume mode: the session and the run already exist (from another process)
        paused = agent_repository.get_run(db, user.org_id, user.id, args.resume)
        if paused is None:
            print(f"No run {args.resume} for user {user.email}")
            sys.exit(1)
        chat_session = chat_repository.get_session(db, user.org_id, user.id, paused.session_id)
        work = [("resume", None)]
    else:
        chat_session = chat.create_session(db, user.org_id, user.id)
        work = list(enumerate([args.question, args.follow_up], start=1))
    print(f"Session {chat_session.id} (user {user.email})")
    alerts_before = db.execute(select(func.count()).select_from(Alert)).scalar_one()
    reports_before = db.execute(select(func.count()).select_from(Report)).scalar_one()
    last_report_id = db.execute(select(func.max(Report.id))).scalar_one() or 0
    keep_session = args.keep or (args.resume is not None and not args.cleanup)

    for number, question in work:
        if args.resume is None and question is None:
            continue

        started = time.monotonic()
        try:
            if args.resume is not None:
                pending = agent_repository.get_pending_tool_call(db, paused.id)
                print(
                    f"\n=== Decision for run {paused.id}: {args.decision} "
                    f"(status {paused.status}, pending call "
                    f"{pending.id if pending else None})"
                )
                events = agent_run.decide_run(
                    db,
                    user.org_id,
                    user.id,
                    paused.id,
                    pending.id if pending else 0,
                    args.decision,
                )
            else:
                print(
                    f"\n=== Question {number}: {question}  "
                    f"(mode: {args.mode}, ticker: {ticker or 'all'})"
                )
                events = agent_run.ask_question(
                    db, user.org_id, user.id, chat_session.id, question, ticker, args.mode
                )
        except (NotFoundError, ServiceUnavailableError, ConflictError) as exc:
            print(f"Cannot ask: {exc.message}")
            sys.exit(1)

        done = None
        print("--- Events:")
        for event in events:
            at = f"[{time.monotonic() - started:5.1f}s]"
            if event["type"] == "token":
                print(event["text"], end="", flush=True)
                continue
            if event["type"] == "route":
                print(f"{at} route: {event['route']} (mode {event['mode']})")
            elif event["type"] == "step":
                print(f"\n{at} step {event['step']}: {event['tool']} {event['args']}")
            elif event["type"] == "step_result":
                print(
                    f"{at} step {event['step']} result: {event['tool']} ok={event['ok']} "
                    f"chars={event['chars']}"
                )
            elif event["type"] == "decision":
                print(f"{at} decision: {event['decision']} for {event['tool']}")
            elif event["type"] == "approval_required":
                print(
                    f"\n{at} waiting for approval: run {event['run_id']}, call "
                    f"{event['tool_call_id']}: {event['summary']}\n"
                    f"      args {event['args']}, expires {event['expires_at']}"
                )
            elif event["type"] == "done":
                done = event
                print(f"\n{at} done: {event}")
            elif event["type"] == "error":
                print(f"\n{at} ERROR event: {event['detail']}")
            else:
                print(f"\n{at} {event['type']}")
        print(f"--- Total {time.monotonic() - started:.1f} seconds")

        # The stored run of this question (the RAG path has none)
        db.expire_all()
        statement = select(AgentRun).where(AgentRun.session_id == chat_session.id)
        if args.resume is not None:
            statement = statement.where(AgentRun.id == args.resume)
        run = db.execute(statement.order_by(AgentRun.id.desc()).limit(1)).scalar_one_or_none()
        if run is None or (done is not None and done.get("run_id") != run.id):
            print("--- No agent run was stored for this question (the RAG path answered)")
            continue

        stored = agent_repository.get_run(db, user.org_id, user.id, run.id)
        print(
            f"--- Run {stored.id}: status={stored.status} step_count={stored.step_count} "
            f"error={stored.error} finished={stored.finished_at is not None}"
        )
        if stored.status == AgentRunStatus.waiting_approval:
            keep_session = True
            pending = agent_repository.get_pending_tool_call(db, stored.id)
            print(
                f"--- Paused: run {stored.id} waits for approval of call {pending.id}. Decide it "
                f"(from this or another process) with:\n    python -m scripts.try_agent --email "
                f"{user.email} --resume {stored.id} --decision approve|reject"
            )
        for call in agent_repository.list_tool_calls(db, stored.id):
            print(
                f"    tool_call step={call.step} {call.tool_name} {call.input} "
                f"output={len(call.output)} chars is_error={call.is_error} "
                f"{call.duration_ms} ms approval={call.approval_status}"
            )

        rows = chat_repository.list_messages_with_citations(db, chat_session.id)
        for message, citations in rows:
            if message.id == stored.answer_message_id:
                print(f"--- Stored answer:\n{message.content}")
                for citation in citations:
                    print(
                        f"    [{citation.number}] chunk {citation.chunk_id} {citation.ticker} "
                        f"FY{citation.fiscal_year} {citation.section} score {citation.score:.3f}"
                        f"\n        {citation.content[:200]!r}"
                    )

        # What the Postgres checkpointer holds for this run's thread
        with agent_run.open_checkpointer() as saver:
            state = build_graph(saver).get_state({"configurable": {"thread_id": str(stored.id)}})
            messages = state.values.get("messages", [])
            print(f"--- Checkpointer: {len(messages)} messages in thread {stored.id}")
            if not keep_session:
                saver.delete_thread(str(stored.id))

    alerts_after = db.execute(select(func.count()).select_from(Alert)).scalar_one()
    print(f"\nRows in alerts: {alerts_before} before, {alerts_after} after")
    db.expire_all()
    reports_after = db.execute(select(func.count()).select_from(Report)).scalar_one()
    print(f"Rows in reports: {reports_before} before, {reports_after} after")
    # The reports this process created, as the Reports page would show them
    for new_report in db.execute(
        select(Report).where(Report.id > last_report_id, Report.org_id == user.org_id)
    ).scalars():
        detail = report_repository.get_by_id(db, user.org_id, new_report.id)
        companies = report_repository.list_companies(db, user.org_id, new_report.id)
        citations = report_repository.list_citations(db, user.org_id, new_report.id)
        print(
            f"Stored report {new_report.id}: {new_report.title!r}, companies "
            f"{[company.ticker for company in companies]}, {len(citations)} citations, "
            f"{len(new_report.data_sources)} data sources, {len(new_report.content)} characters, "
            f"created by {detail[1]}"
        )
    if keep_session:
        print(f"Session {chat_session.id} kept")
    else:
        chat.delete_session(db, user.org_id, user.id, chat_session.id)
        print(f"\nSession {chat_session.id} deleted")
finally:
    db.close()
    # The tracer uploads in the background: wait for it (a no-op when tracing is off)
    llm.flush_traces()
