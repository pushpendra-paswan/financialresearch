# Run with: docker compose exec api python -m scripts.try_chat "question" --email you@example.com
#   [--ticker NVDA] [--follow-up "second question"] [--keep]
# Creates a chat session for an EXISTING user, asks the question (and the follow-up in the same
# session), streams the answer to the terminal and prints the cited passages. The session is
# deleted at the end unless --keep is given. It calls OpenAI (one embedding and one chat call per
# question, plus one rewrite call for the follow-up), so it needs OPENAI_API_KEY.
import argparse
import logging
import sys

from app.config import settings
from app.database import SessionLocal
from app.exceptions import NotFoundError, ServiceUnavailableError
from app.rag import chat, llm
from app.repositories import chat as chat_repository
from app.repositories import users as user_repository

# INFO for our chat module (what it decided) and for httpx (one line per real OpenAI request,
# which shows whether the answer model was called). httpx logs only the URL, never the key
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logging.getLogger("app.rag.chat").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpx2").setLevel(logging.INFO)  # the name this install logs under

parser = argparse.ArgumentParser(description="Ask the chat a question on the stored filings")
parser.add_argument("question", help="the question")
parser.add_argument("--email", required=True, help="email of an existing user (owns the session)")
parser.add_argument("--ticker", help="only this ticker (one of RAG_TICKERS)")
parser.add_argument("--follow-up", help="a second question asked in the same session")
parser.add_argument("--keep", action="store_true", help="do not delete the session at the end")
args = parser.parse_args()

if not settings.OPENAI_API_KEY:
    print("OPENAI_API_KEY is not set: chat is disabled")
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
        print(f"\n=== Question {number}: {question}  (ticker: {ticker or 'all'})")

        try:
            events = chat.ask(db, user.org_id, user.id, chat_session.id, question, ticker)
        except (NotFoundError, ServiceUnavailableError) as exc:
            print(f"Cannot ask: {exc.message}")
            sys.exit(1)

        sources = []
        cited_numbers = []
        print("--- Answer (streamed):")
        for event in events:
            if event["type"] == "sources":
                sources = event["sources"]
            elif event["type"] == "token":
                print(event["text"], end="", flush=True)
            elif event["type"] == "done":
                cited_numbers = event["cited_numbers"]
            else:
                print(f"\nERROR event: {event['detail']}")
                sys.exit(1)
        print()

        # The rewritten question is stored on the user message
        messages = chat_repository.list_recent_messages(db, chat_session.id, 2)
        user_message = messages[0]
        if user_message.rewritten_question:
            print(f"--- Rewritten question: {user_message.rewritten_question}")

        print(f"--- Sources sent to the model: {len(sources)}; cited: {cited_numbers}")
        for source in sources:
            marker = "CITED" if source["number"] in cited_numbers else "     "
            print(
                f"[{source['number']}] {marker} chunk {source['chunk_id']} {source['ticker']} "
                f"FY{source['fiscal_year']} {source['section']} similarity {source['score']:.3f}"
            )
        for source in sources:
            if source["number"] in cited_numbers:
                print(f"\n[{source['number']}] cited passage:\n{source['content']}")

    if args.keep:
        print(f"\nSession {chat_session.id} kept")
    else:
        chat.delete_session(db, user.org_id, user.id, chat_session.id)
        print(f"\nSession {chat_session.id} deleted")
finally:
    db.close()
    # The tracer uploads in the background: wait for it (a no-op when tracing is off)
    llm.flush_traces()
