# Run with: docker compose exec api python -m scripts.run_embedding [options]
#   --dry-run      split the filings and print the numbers; no OpenAI call, nothing is written
#   --reembed      replace the chunks of the filings instead of skipping filings that have some
#   --ticker X     only the in-scope filings of this ticker
#   --filing-id N  only this in-scope filing (use with --reembed to replace one filing)
import argparse
import logging
import statistics
import sys
import time

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.database import SessionLocal
from app.exceptions import ConflictError
from app.rag import chunking
from app.rag.parsing import list_scope_filing_ids, load_filing_sections
from app.repositories import filings as filing_repository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--reembed", action="store_true")
parser.add_argument("--ticker")
parser.add_argument("--filing-id", type=int)
args = parser.parse_args()

# Without a key nothing can be embedded (a dry run needs none)
if not args.dry_run and not settings.OPENAI_API_KEY:
    print("embeddings disabled: OPENAI_API_KEY not set")
    sys.exit(1)

db = SessionLocal()
try:
    # Only filings inside the RAG scope, optionally narrowed to one ticker or one filing
    filing_ids = []
    for filing_id in list_scope_filing_ids(db):
        _, ticker = filing_repository.get_by_id_with_ticker(db, filing_id)
        if args.ticker and ticker != args.ticker.upper():
            continue
        if args.filing_id and filing_id != args.filing_id:
            continue
        filing_ids.append(filing_id)
    if not filing_ids:
        print("No in-scope filing matches these options")
        sys.exit(1)

    if args.dry_run:
        # Same splitter settings as the real run, no embedding call
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
        )
        # The encoding text-embedding-3-small uses. Downloaded once (1.7 MB) and then cached
        encoding = tiktoken.encoding_for_model(settings.EMBEDDING_MODEL)
        all_lengths = []
        total_tokens = 0
        print("filing ticker  FY section        chunks   min   med   max   tokens")
        for filing_id in filing_ids:
            filing, ticker = filing_repository.get_by_id_with_ticker(db, filing_id)
            for section in load_filing_sections(db, filing_id):
                chunks = splitter.split_documents([section])
                lengths = [len(chunk.page_content) for chunk in chunks]
                tokens = sum(len(encoding.encode_ordinary(chunk.page_content)) for chunk in chunks)
                all_lengths += lengths
                total_tokens += tokens
                print(
                    f"{filing_id:>6} {ticker:<6} {filing.fiscal_year} "
                    f"{section.metadata['section']:<13} {len(chunks):>6} {min(lengths):>5} "
                    f"{int(statistics.median(lengths)):>5} {max(lengths):>5} {tokens:>8}"
                )
        print(
            f"Total: {len(all_lengths)} chunks, {sum(all_lengths)} characters, "
            f"{total_tokens} estimated tokens ({encoding.name}), no API call made"
        )
    else:
        started = time.perf_counter()
        run = chunking.embed_filings(db, replace=args.reembed, filing_ids=filing_ids)
        print(f"Status: {run.status}")
        print(f"Message: {run.message}")
        print(f"Runtime: {time.perf_counter() - started:.1f} seconds")
except ConflictError as error:
    print(error.message)
    sys.exit(1)
finally:
    db.close()
