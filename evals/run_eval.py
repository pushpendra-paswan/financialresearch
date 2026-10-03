# Run with: docker compose exec api python -m evals.run_eval --name <run name>
#   [--no-rerank] [--rerank-model rerank-v4.0-pro] [--answers] [--limit N] [--rerank-pause SECONDS]
#   [--validate-only]
#
# Evaluates the retrieval (and with --answers the answers) on the questions in
# evals/questions.json. Every question goes through the SAME code as chat (chat.prepare_context:
# scope, ticker filter, hybrid retrieval, optional rerank, relevance threshold), so the numbers
# describe what a user gets. It writes evals/runs/<name>.json and prints one markdown table row.
#
# Costs: one OpenAI embedding per question; one Cohere rerank per question unless --no-rerank;
# with --answers also one answer call and one judge call per question that passed the threshold.
# Labels are PHRASE based: a retrieved chunk is relevant when its ticker and section match an
# expected entry and its text contains one of that entry's phrases. Chunk ids change on every
# re-embed and every chunk-size change, a distinctive phrase from the filing survives both.
import argparse
import json
import logging
import math
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import openai
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.config import settings
from app.database import SessionLocal
from app.rag import chat, llm
from app.rag.parsing import SECTIONS, list_scope_filing_ids
from app.repositories import chunks as chunk_repository

EVALS_DIR = Path(__file__).parent
QUESTIONS_PATH = EVALS_DIR / "questions.json"
RUNS_DIR = EVALS_DIR / "runs"
# The metrics are hit@1/3/5 and MRR@5, so retrieval must return at least 5 chunks
METRIC_K = 5


class JudgedClaim(BaseModel):
    claim: str
    cited_numbers: list[int]
    supported: bool


class JudgeResult(BaseModel):
    abstained: bool
    claims: list[JudgedClaim]


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You check an answer against the numbered excerpts it was written from. Rules:\n"
            "- Split the answer into factual claims: every statement of a fact (a number, an "
            "event, a reason) is one claim. Ignore statements that only say the information is "
            "not available.\n"
            "- For each claim, give cited_numbers: the excerpt numbers written in square "
            "brackets right after it in the answer (an empty list when it has no citation).\n"
            "- supported is true only when at least one excerpt listed in cited_numbers states "
            "the claim. A claim with no citation, or whose cited excerpts do not state it, is "
            "NOT supported. Use no outside knowledge.\n"
            "- abstained is true when the answer says it does not know, or that the excerpts do "
            "not contain the answer, and states no factual claim. If it states claims, "
            "abstained is false.",
        ),
        ("human", "Excerpts:\n\n{excerpts}\n\nQuestion: {question}\n\nAnswer to check:\n{answer}"),
    ]
)


# ---------- metric functions (plain, so tests can call them) ----------


def normalize(text: str) -> str:
    # Lowercase and collapse every run of whitespace (the filings are hard-wrapped lines)
    return " ".join(text.lower().split())


def is_relevant(ticker: str, section: str, content: str, expected: list[dict]) -> bool:
    text = normalize(content)
    for entry in expected:
        if entry["ticker"] == ticker and entry["section"] == section:
            if any(normalize(phrase) in text for phrase in entry["phrases"]):
                return True
    return False


def first_relevant_rank(documents: list[Document], expected: list[dict]) -> int | None:
    # 1-based rank of the first relevant chunk, None when no retrieved chunk is relevant
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata
        if is_relevant(metadata["ticker"], metadata["section"], document.page_content, expected):
            return rank
    return None


def hit_rate(ranks: list[int | None], k: int) -> float:
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None], k: int) -> float:
    return sum(1 / rank for rank in ranks if rank is not None and rank <= k) / len(ranks)


def rejection_metrics(results: list[dict]) -> dict:
    # Each result has "answerable", "rejected_by_threshold" (no retrieved chunk reached
    # RELEVANCE_THRESHOLD) and "abstained" (None when no answers were generated)
    unanswerable = [result for result in results if not result["answerable"]]
    answerable = [result for result in results if result["answerable"]]
    rejected = [
        result
        for result in unanswerable
        if result["rejected_by_threshold"] or result["abstained"] is True
    ]
    wrongly_rejected = [result for result in answerable if result["rejected_by_threshold"]]
    return {
        "unanswerable_rejected": len(rejected) / len(unanswerable) if unanswerable else None,
        "answerable_wrongly_rejected": (
            len(wrongly_rejected) / len(answerable) if answerable else None
        ),
    }


def answer_metrics(results: list[dict]) -> dict:
    # Each result also has "claims": the judge's claims as dicts with "cited_numbers" and
    # "supported" (None when the answer was not judged) and "source_count" (how many numbered
    # excerpts the model saw). Faithfulness and coverage are averaged over the ANSWERABLE answers
    # that did not abstain and have at least one claim, so every answer counts once
    faithfulness = []
    coverage = []
    for result in results:
        claims = result.get("claims")
        if not result["answerable"] or result["abstained"] or not claims:
            continue
        valid_range = range(1, result["source_count"] + 1)
        faithfulness.append(sum(1 for claim in claims if claim["supported"]) / len(claims))
        coverage.append(
            sum(1 for claim in claims if any(n in valid_range for n in claim["cited_numbers"]))
            / len(claims)
        )
    unanswerable = [result for result in results if not result["answerable"]]
    abstained = [result for result in unanswerable if result["abstained"] is True]
    return {
        "faithfulness": sum(faithfulness) / len(faithfulness) if faithfulness else None,
        "citation_coverage": sum(coverage) / len(coverage) if coverage else None,
        "abstention_on_unanswerable": (
            len(abstained) / len(unanswerable) if unanswerable else None
        ),
        "judged_answers": len(faithfulness),
    }


def validate_questions(
    questions: list[dict], rag_tickers: list[str], sections: list[str]
) -> list[str]:
    # The structure of the question file. Returns one message per problem (empty list = valid)
    errors = []
    seen_ids = set()
    for position, question in enumerate(questions, start=1):
        label = question.get("id", f"#{position}")
        for key in ("id", "question", "ticker", "answerable", "expected"):
            if key not in question:
                errors.append(f"{label}: missing key '{key}'")
        if "id" not in question or "expected" not in question or "answerable" not in question:
            continue
        if question["id"] in seen_ids:
            errors.append(f"{label}: duplicate id")
        seen_ids.add(question["id"])
        if not str(question.get("question", "")).strip():
            errors.append(f"{label}: empty question")
        if question.get("ticker") is not None and question["ticker"] not in rag_tickers:
            errors.append(f"{label}: ticker {question['ticker']} is not in RAG_TICKERS")
        if not isinstance(question["answerable"], bool):
            errors.append(f"{label}: answerable must be true or false")
        elif question["answerable"] and not question["expected"]:
            errors.append(f"{label}: an answerable question needs expected entries")
        elif not question["answerable"] and question["expected"]:
            errors.append(f"{label}: an unanswerable question must have no expected entries")
        for entry in question["expected"]:
            if entry.get("ticker") not in rag_tickers:
                errors.append(f"{label}: expected ticker {entry.get('ticker')} not in RAG_TICKERS")
            if entry.get("section") not in sections:
                errors.append(f"{label}: expected section {entry.get('section')} is unknown")
            phrases = entry.get("phrases")
            if not phrases or not all(isinstance(p, str) and p.strip() for p in phrases):
                errors.append(f"{label}: every expected entry needs non-empty phrases")
    return errors


def find_missing_phrases(
    questions: list[dict], chunk_rows: list[tuple[int, str, int, str, str]]
) -> list[str]:
    # Every phrase must occur in at least one stored chunk of its ticker and section, otherwise
    # the label could never be hit and the metrics would be wrong. chunk_rows come from
    # chunk_repository.list_for_filings
    texts: dict[tuple[str, str], list[str]] = {}
    for _filing_id, section, _chunk_index, ticker, content in chunk_rows:
        texts.setdefault((ticker, section), []).append(normalize(content))
    missing = []
    for question in questions:
        for entry in question["expected"]:
            for phrase in entry["phrases"]:
                chunk_texts = texts.get((entry["ticker"], entry["section"]), [])
                if not any(normalize(phrase) in text for text in chunk_texts):
                    missing.append(
                        f"{question['id']}: '{phrase}' not found in {entry['ticker']} "
                        f"{entry['section']}"
                    )
    return missing


def observed_chunk_layout(chunk_rows: list[tuple[int, str, int, str, str]]) -> tuple[int, int]:
    # The settings used to create the stored chunks are not stored, so they are read back from
    # the chunks: the longest chunk, and the longest text that ends one chunk and starts the next
    # one (the overlap). Rows must be in reading order (filing, section, chunk_index)
    longest_chunk = max(len(row[4]) for row in chunk_rows)
    longest_overlap = 0
    for previous, current in zip(chunk_rows, chunk_rows[1:], strict=False):
        if previous[:2] != current[:2]:
            continue  # another filing or section: the splitter never overlaps across them
        for size in range(min(len(previous[4]), len(current[4])), longest_overlap, -1):
            if previous[4].endswith(current[4][:size]):
                longest_overlap = size
                break
    return longest_chunk, longest_overlap


def format_number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


# ---------- the run ----------


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("app.rag.retrieval").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers")
    parser.add_argument("--name", required=True, help="run name, used for evals/runs/<name>.json")
    parser.add_argument("--no-rerank", action="store_true", help="skip the Cohere rerank step")
    parser.add_argument("--rerank-model", help="Cohere model for this run (default RERANK_MODEL)")
    parser.add_argument("--answers", action="store_true", help="also generate and judge answers")
    parser.add_argument("--limit", type=int, help="only the first N questions (a smoke test)")
    parser.add_argument(
        "--rerank-pause",
        type=float,
        default=6.5,
        help="seconds to wait after each question when reranking (trial key: 10 calls a minute)",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="check the question file and stop (free)"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("--name may only contain letters, digits, _ and -")

    # 1. Refuse to start without the keys this run needs (nothing is called before this)
    rerank_on = not args.no_rerank
    if not args.validate_only:
        if not settings.OPENAI_API_KEY:
            print("OPENAI_API_KEY is not set")
            sys.exit(1)
        if rerank_on and not settings.COHERE_API_KEY:
            print("COHERE_API_KEY is not set (use --no-rerank for a run without Cohere)")
            sys.exit(1)
        if settings.RETRIEVAL_TOP_K < METRIC_K:
            print(f"RETRIEVAL_TOP_K must be at least {METRIC_K} for hit@{METRIC_K} and MRR")
            sys.exit(1)

    # 2. Validate the question file before any paid call: structure first, then the phrases
    questions = json.loads(QUESTIONS_PATH.read_text())
    rag_tickers = [ticker.strip().upper() for ticker in settings.RAG_TICKERS.split(",")]
    errors = validate_questions(questions, rag_tickers, list(SECTIONS))
    db = SessionLocal()
    scope_filing_ids = list_scope_filing_ids(db)
    chunk_rows = chunk_repository.list_for_filings(db, scope_filing_ids)
    if not errors:
        errors = find_missing_phrases(questions, chunk_rows)
    if not chunk_rows:
        errors.append("there are no stored chunks for the in-scope filings")
    if errors:
        print("The question file is not valid, nothing was run:")
        for error in errors:
            print(f"  - {error}")
        db.close()
        sys.exit(1)
    longest_chunk, longest_overlap = observed_chunk_layout(chunk_rows)
    chunk_size = math.ceil(longest_chunk / 100) * 100
    print(
        f"Question file OK: {len(questions)} questions. Stored chunks: {len(chunk_rows)}, longest "
        f"{longest_chunk} characters (chunk size {chunk_size}), overlap up to {longest_overlap}"
    )
    if args.validate_only:
        db.close()
        return

    # 3. Settings of this run. The settings object is changed in memory only (nothing is saved):
    # prepare_context and retrieve read their rerank settings from it
    settings.RERANK_ENABLED = rerank_on
    if args.rerank_model:
        settings.RERANK_MODEL = args.rerank_model
    if args.limit:
        questions = questions[: args.limit]
    model = llm.get_chat_model()
    judge_model = model.with_structured_output(JudgeResult)
    counts = {"embed_query": 0, "rerank": 0, "answer": 0, "judge": 0}
    results = []

    # 4. Run every question
    for question in questions:
        # Retrieval exactly as chat does it. Reranking fails open, which would silently turn this
        # run into a run without rerank, so a question whose rerank failed is retried once and
        # then the run stops
        for attempt in (1, 2):
            retrieved, passing, sources, context_text = chat.prepare_context(
                db, question["question"], question["ticker"]
            )
            counts["embed_query"] += 1
            if rerank_on:
                counts["rerank"] += 1
            if not (rerank_on and retrieved and retrieved[0].metadata["rerank_score"] is None):
                break
            if attempt == 1:
                print(f"{question['id']}: rerank failed (see the warning above), retrying in 30 s")
                time.sleep(30)
            else:
                print(f"{question['id']}: rerank failed again, stopping: the run would be wrong")
                db.close()
                sys.exit(1)

        rank = None
        if question["answerable"]:
            rank = first_relevant_rank(retrieved, question["expected"])
        best_similarity = max(
            (document.metadata["vector_similarity"] for document in retrieved), default=None
        )
        result = {
            "id": question["id"],
            "question": question["question"],
            "ticker": question["ticker"],
            "answerable": question["answerable"],
            "rank": rank,
            "best_similarity": best_similarity,
            "rejected_by_threshold": not passing,
            "abstained": None,
            "source_count": len(sources),
            "answer": None,
            "claims": None,
            "error": None,
            "retrieved": [
                {
                    "rank": position,
                    "chunk_id": document.metadata["chunk_id"],
                    "ticker": document.metadata["ticker"],
                    "fiscal_year": document.metadata["fiscal_year"],
                    "section": document.metadata["section"],
                    "vector_similarity": document.metadata["vector_similarity"],
                    "rerank_score": document.metadata["rerank_score"],
                    "relevant": is_relevant(
                        document.metadata["ticker"],
                        document.metadata["section"],
                        document.page_content,
                        question["expected"],
                    ),
                    "preview": " ".join(document.page_content.split())[:150],
                }
                for position, document in enumerate(retrieved, start=1)
            ],
        }

        # Answer and judge, only with --answers. Below the threshold chat does not call the
        # answer model: the answer is the fixed NO_ANSWER text, which counts as an abstention
        judge_text = ""
        if args.answers:
            if not passing:
                result["answer"] = chat.NO_ANSWER
                result["abstained"] = True
                judge_text = "no answer call (below threshold)"
            else:
                try:
                    answer = (chat.QA_PROMPT | model).invoke(
                        {"excerpts": context_text, "question": question["question"]}
                    )
                    counts["answer"] += 1
                    result["answer"] = answer.text
                    judged = (JUDGE_PROMPT | judge_model).invoke(
                        {
                            "excerpts": context_text,
                            "question": question["question"],
                            "answer": answer.text,
                        }
                    )
                    counts["judge"] += 1
                    result["abstained"] = judged.abstained
                    result["claims"] = [claim.model_dump() for claim in judged.claims]
                    supported = sum(1 for claim in judged.claims if claim.supported)
                    judge_text = (
                        "abstained"
                        if judged.abstained
                        else f"{supported}/{len(judged.claims)} claims supported"
                    )
                except (openai.OpenAIError, ValueError) as exc:
                    # ValueError covers a judge reply that does not fit the schema. Only the
                    # class name is kept (an OpenAI message can hold part of a key)
                    result["error"] = type(exc).__name__
                    judge_text = f"ERROR {type(exc).__name__}"

        results.append(result)
        if question["answerable"]:
            outcome = f"hit@{rank}" if rank else "miss"
        else:
            outcome = "rejected" if not passing else "NOT rejected"
        similarity_text = "none" if best_similarity is None else f"{best_similarity:.3f}"
        print(f"{question['id']:5} {outcome:13} best_similarity={similarity_text}  {judge_text}")

        if rerank_on:
            time.sleep(args.rerank_pause)
    db.close()

    # 5. Aggregate metrics (retrieval metrics over the answerable questions only)
    ranks = [result["rank"] for result in results if result["answerable"]]
    metrics = {
        "answerable_questions": len(ranks),
        "unanswerable_questions": len(results) - len(ranks),
        "hit_at_1": hit_rate(ranks, 1) if ranks else None,
        "hit_at_3": hit_rate(ranks, 3) if ranks else None,
        "hit_at_5": hit_rate(ranks, 5) if ranks else None,
        "mrr_at_5": mean_reciprocal_rank(ranks, 5) if ranks else None,
        **rejection_metrics(results),
    }
    if args.answers:
        metrics.update(answer_metrics(results))

    run_settings = {
        "chunk_size": chunk_size,
        "longest_chunk_chars": longest_chunk,
        "longest_observed_overlap": longest_overlap,
        "chunk_count": len(chunk_rows),
        "rerank": rerank_on,
        "rerank_model": settings.RERANK_MODEL if rerank_on else None,
        "rerank_candidates_k": settings.RERANK_CANDIDATES_K if rerank_on else None,
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "retrieval_candidates_k": settings.RETRIEVAL_CANDIDATES_K,
        "relevance_threshold": settings.RELEVANCE_THRESHOLD,
        "answers": args.answers,
        "chat_model": settings.CHAT_MODEL if args.answers else None,
        "question_count": len(results),
    }
    RUNS_DIR.mkdir(exist_ok=True)
    run_path = RUNS_DIR / f"{args.name}.json"
    run_path.write_text(
        json.dumps(
            {
                "name": args.name,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "settings": run_settings,
                "metrics": metrics,
                "call_counts": counts,
                "questions": results,
            },
            indent=2,
        )
    )

    # 6. The table row and the call counts
    rerank_text = settings.RERANK_MODEL if rerank_on else "off"
    row = (
        f"| {args.name} | {chunk_size} | {rerank_text} | {format_number(metrics['hit_at_1'])} "
        f"| {format_number(metrics['hit_at_3'])} | {format_number(metrics['hit_at_5'])} "
        f"| {format_number(metrics['mrr_at_5'])} "
        f"| {format_number(metrics['unanswerable_rejected'])} "
        f"| {format_number(metrics['answerable_wrongly_rejected'])} "
        f"| {format_number(metrics.get('faithfulness'))} "
        f"| {format_number(metrics.get('citation_coverage'))} "
        f"| {format_number(metrics.get('abstention_on_unanswerable'))} |"
    )
    print()
    print(
        "| run | chunk size | rerank | hit@1 | hit@3 | hit@5 | MRR@5 | unanswerable rejected "
        "| answerable wrongly rejected | faithfulness | citation coverage | abstention |"
    )
    print(row)
    print()
    print(
        f"Calls: embed_query={counts['embed_query']} rerank={counts['rerank']} "
        f"answer={counts['answer']} judge={counts['judge']}  (saved {run_path})"
    )


if __name__ == "__main__":
    main()
