# Run with: docker compose exec api python -m scripts.try_agent --email you@example.com "question"
#   [--mode agent|auto] [--ticker NVDA] [--follow-up "second question"] [--keep]
# or:       docker compose exec api python -m scripts.try_agent --route-samples
# Creates a chat session for an EXISTING user, asks the question (and the follow-up in the same
# session) through the same function the chat route uses, prints the events as they stream (steps
# and tokens), then the stored run, its tool calls, the citations and how many messages the
# checkpointer holds for the run's thread. The session is deleted at the end unless --keep is
# given. It calls OpenAI (several chat calls per question, embeddings and Cohere for searches), so
# it needs OPENAI_API_KEY. --route-samples only calls the router for 10 built-in questions.
import argparse
import logging
import sys
import time

from sqlalchemy import select

from app.agent import run as agent_run
from app.agent.graph import build_graph
from app.config import settings
from app.database import SessionLocal
from app.exceptions import ConflictError, NotFoundError, ServiceUnavailableError
from app.models.agent import AgentRun
from app.rag import chat
from app.repositories import agent as agent_repository
from app.repositories import chat as chat_repository
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

if not args.question or not args.email:
    print("Give a question and --email (or use --route-samples)")
    sys.exit(1)
ticker = args.ticker.strip().upper() if args.ticker else None

db = SessionLocal()
try:
    user = user_repository.get_by_email(db, args.email.strip().lower())
    if user is None:
        print(f"No user with email {args.email}")
        sys.exit(1)

    chat_session = chat.create_session(db, user.org_id, user.id)
    print(f"Session {chat_session.id} (user {user.email})")

    for number, question in enumerate([args.question, args.follow_up], start=1):
        if question is None:
            continue
        print(
            f"\n=== Question {number}: {question}  (mode: {args.mode}, ticker: {ticker or 'all'})"
        )

        started = time.monotonic()
        try:
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
        run = db.execute(statement.order_by(AgentRun.id.desc()).limit(1)).scalar_one_or_none()
        if run is None or (done is not None and done.get("run_id") != run.id):
            print("--- No agent run was stored for this question (the RAG path answered)")
            continue

        stored = agent_repository.get_run(db, user.org_id, user.id, run.id)
        print(
            f"--- Run {stored.id}: status={stored.status} step_count={stored.step_count} "
            f"error={stored.error} finished={stored.finished_at is not None}"
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
            if not args.keep:
                saver.delete_thread(str(stored.id))

    if args.keep:
        print(f"\nSession {chat_session.id} kept")
    else:
        chat.delete_session(db, user.org_id, user.id, chat_session.id)
        print(f"\nSession {chat_session.id} deleted")
finally:
    db.close()
