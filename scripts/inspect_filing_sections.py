# Run with: docker compose exec api python -m scripts.inspect_filing_sections [--show]
# Parses every in-scope 10-K (RAG_TICKERS, RAG_LOOKBACK_YEARS) and prints what was found.
# Read-only: it writes nothing.
import logging
import sys
import time

from app.database import SessionLocal
from app.rag.parsing import SECTIONS, list_scope_filing_ids, load_filing_sections
from app.repositories import filings as filing_repository

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")

show_text = "--show" in sys.argv

db = SessionLocal()
try:
    total_start = time.perf_counter()
    filing_ids = list_scope_filing_ids(db)
    print(f"{len(filing_ids)} in-scope 10-K filings")

    for filing_id in filing_ids:
        filing, ticker = filing_repository.get_by_id_with_ticker(db, filing_id)

        parse_start = time.perf_counter()
        documents = load_filing_sections(db, filing_id)
        parse_seconds = time.perf_counter() - parse_start

        print()
        print(f"{ticker} fiscal_year={filing.fiscal_year} {filing.accession_number}")
        by_section = {document.metadata["section"]: document for document in documents}
        for key in SECTIONS:
            document = by_section.get(key)
            if document is None:
                print(f"  {key}: missing (see the warning above)")
                continue
            text = document.page_content
            print(f"  {key}: {len(text)} chars")
            if show_text:
                print(f"    FIRST 300: {text[:300]!r}")
                print(f"    LAST 300:  {text[-300:]!r}")
        print(f"  parse time: {parse_seconds:.2f}s")

    print()
    print(f"Total runtime: {time.perf_counter() - total_start:.2f}s")
finally:
    db.close()
