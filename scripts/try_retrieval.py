# Run with: docker compose exec api python -m scripts.try_retrieval "question" [--ticker X ...]
#   [--section S ...] [--year-from Y] [--year-to Y] [--top-k N]
# or:       docker compose exec api python -m scripts.try_retrieval --samples
# Runs the hybrid retrieval and prints the results. It writes nothing. It makes one OpenAI
# embed_query call per question, so it needs OPENAI_API_KEY. With COHERE_API_KEY set and
# RERANK_ENABLED true it also reranks (one Cohere call per question); --no-rerank turns that off.
import argparse
import logging
import sys

from app.config import settings
from app.database import SessionLocal
from app.rag.retrieval import retrieve

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# (question, tickers, sections): the manual check for milestone 2.3
SAMPLES = [
    ("What risks does NVIDIA describe from US export controls on China?", ["NVDA"], None),
    ("How did Apple's iPhone net sales change compared with the previous year?", ["AAPL"], None),
    ("How does NVIDIA depend on third-party foundries such as TSMC?", ["NVDA"], None),
    (
        "What does Apple say about manufacturing concentration and outsourcing partners in Asia?",
        ["AAPL"],
        None,
    ),
    ("What drove growth in NVIDIA's Data Center revenue?", ["NVDA"], ["mdna"]),
    ("How do Apple and NVIDIA describe competition in their markets?", None, None),
    ("What is a good recipe for banana bread?", None, None),
]

parser = argparse.ArgumentParser(description="Try the hybrid retrieval on the stored chunks")
parser.add_argument("question", nargs="?", help="the question to search for")
parser.add_argument("--ticker", action="append", help="only this ticker (repeat for several)")
parser.add_argument("--section", action="append", help="only this section (repeat for several)")
parser.add_argument("--year-from", type=int, help="first fiscal year")
parser.add_argument("--year-to", type=int, help="last fiscal year")
parser.add_argument("--top-k", type=int, help="chunks to return (default RETRIEVAL_TOP_K)")
parser.add_argument("--samples", action="store_true", help="run the 7 sample questions")
parser.add_argument("--no-rerank", action="store_true", help="skip the Cohere rerank step")
args = parser.parse_args()

if args.samples == (args.question is not None):
    parser.error("give either a question or --samples")
if not settings.OPENAI_API_KEY:
    print("OPENAI_API_KEY is not set: the question cannot be embedded")
    sys.exit(1)

if args.samples:
    runs = SAMPLES
else:
    runs = [(args.question, args.ticker, args.section)]

db = SessionLocal()
try:
    for number, (question, tickers, sections) in enumerate(runs, start=1):
        try:
            documents = retrieve(
                db,
                question,
                tickers=tickers,
                year_from=args.year_from,
                year_to=args.year_to,
                sections=sections,
                top_k=args.top_k,
                rerank=False if args.no_rerank else None,
            )
        except ValueError as error:
            print(f"Invalid input: {error}")
            sys.exit(1)

        print()
        print(f"[{number}] {question}")
        print(
            f"    filters: tickers={tickers} sections={sections} "
            f"year_from={args.year_from} year_to={args.year_to}"
        )
        if not documents:
            print("    no results")
        for rank, document in enumerate(documents, start=1):
            meta = document.metadata
            rerank_score = meta["rerank_score"]
            rerank_text = "none" if rerank_score is None else f"{rerank_score:.4f}"
            preview = " ".join(document.page_content[:200].split())
            print(
                f"  #{rank} {meta['ticker']} FY{meta['fiscal_year']} {meta['section']} "
                f"chunk_id={meta['chunk_id']} score={meta['score']:.5f} "
                f"similarity={meta['vector_similarity']:.4f} rerank_score={rerank_text} "
                f"vector_rank={meta['vector_rank']} text_rank={meta['text_rank']}"
            )
            print(f"      {preview}")
finally:
    db.close()
