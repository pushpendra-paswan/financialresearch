# Run with: docker compose exec api python -m scripts.try_agent_tools --email you@example.com
#   <tool> ['{"json": "arguments"}']      one call of one tool
#   --samples                             one call of each tool plus a few derived metrics
#   --no-context                          call WITHOUT org_id and user_id (shows the ValueError)
# The manual check of the agent tools (3.1) and a debugging tool for 3.2. It calls
# tool.invoke(args, config={"configurable": {"org_id", "user_id"}}) exactly as the agent graph
# will, prints the JSON result, its size and an estimated token count. --email finds an EXISTING
# user for the context; nothing is created and nothing is written. search_filings makes one
# OpenAI embedding call and, when reranking is on (COHERE_API_KEY set), one Cohere call.
import argparse
import json
import logging
import sys
import time

import tiktoken

from app.agent.tools import READ_ONLY_TOOLS
from app.database import SessionLocal
from app.repositories import users as user_repository

# INFO for our tools (one line per call with the context) and for httpx (one line per real
# OpenAI or Cohere request). httpx logs only the URL, never the key
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logging.getLogger("app.agent.tools").setLevel(logging.INFO)
logging.getLogger("app.rag.retrieval").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpx2").setLevel(logging.INFO)  # the name this install logs under

SAMPLES = [
    (
        "search_filings",
        {
            "query": "What risks does NVIDIA describe from US export controls on China?",
            "ticker": "NVDA",
        },
    ),
    ("get_financials", {"ticker": "AAPL", "metric": "revenue", "years": 5}),
    ("get_financials", {"ticker": "NVDA"}),
    ("get_price_history", {"ticker": "NVDA", "days": 90}),
    ("compute_metrics", {"ticker": "NVDA", "metric": "net_margin_pct"}),
    ("compute_metrics", {"ticker": "NVDA", "metric": "revenue_growth_pct"}),
    ("compute_metrics", {"ticker": "AAPL", "metric": "max_drawdown_pct", "days": 365}),
    ("compare_companies", {"tickers": ["AAPL", "NVDA"], "metric": "revenue"}),
    ("compare_companies", {"tickers": ["AAPL", "NVDA"], "metric": "net_margin_pct"}),
]

parser = argparse.ArgumentParser(description="Call the read-only agent tools by hand")
parser.add_argument(
    "tool", nargs="?", help="tool name: " + ", ".join(t.name for t in READ_ONLY_TOOLS)
)
parser.add_argument("arguments", nargs="?", default="{}", help="the tool arguments as JSON")
parser.add_argument("--email", required=True, help="email of an existing user (the context)")
parser.add_argument("--samples", action="store_true", help="run the sample calls")
parser.add_argument("--no-context", action="store_true", help="call without org_id and user_id")
args = parser.parse_args()

tools_by_name = {tool.name: tool for tool in READ_ONLY_TOOLS}
if args.samples:
    calls = SAMPLES
elif args.tool in tools_by_name:
    try:
        calls = [(args.tool, json.loads(args.arguments))]
    except json.JSONDecodeError as exc:
        print(f"The arguments are not valid JSON: {exc}")
        sys.exit(1)
else:
    print(f"Give a tool name ({', '.join(tools_by_name)}) or --samples")
    sys.exit(1)

with SessionLocal() as db:
    user = user_repository.get_by_email(db, args.email.strip().lower())
if user is None:
    print(f"No user with email {args.email}")
    sys.exit(1)
configurable = {} if args.no_context else {"org_id": user.org_id, "user_id": user.id}
print(f"Context: {configurable or 'none'} (user {user.email})")

encoding = tiktoken.get_encoding("cl100k_base")
largest = (0, "")
started_all = time.perf_counter()
for tool_name, tool_arguments in calls:
    print(f"\n=== {tool_name}({json.dumps(tool_arguments)})")
    started = time.perf_counter()
    try:
        result = tools_by_name[tool_name].invoke(
            tool_arguments, config={"configurable": configurable}
        )
    except ValueError as exc:
        # A bug-level error: the graph would crash here (a ToolNode lets it propagate)
        print(f"ValueError (propagates, not a model error): {exc}")
        continue
    seconds = time.perf_counter() - started

    # A ToolException comes back as a plain string (the message the model would see)
    if isinstance(result, str):
        print(f"TOOL ERROR MESSAGE (what the model sees): {result}")
        continue

    text = json.dumps(result, indent=2)
    compact = json.dumps(result)
    tokens = len(encoding.encode(compact))
    print(text)
    print(f"--- {len(compact)} characters (compact JSON), about {tokens} tokens, {seconds:.2f} s")
    if len(compact) > largest[0]:
        largest = (len(compact), f"{tool_name}({json.dumps(tool_arguments)})")

print(f"\nTotal time {time.perf_counter() - started_all:.1f} s")
if largest[0]:
    print(f"Largest output: {largest[0]} characters, {largest[1]} (bound: 16000)")
