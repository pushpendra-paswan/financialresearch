import logging
import re
from datetime import date, timedelta
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
from sqlalchemy.orm import Session

from app.config import settings
from app.exceptions import NotFoundError
from app.repositories import filings as filing_repository

logger = logging.getLogger(__name__)

# A section shorter than this is treated as not found (a table-of-contents row or a stray heading)
MIN_SECTION_CHARS = 2000

# The single source of truth for the sections we extract:
# key -> (label, start heading, end heading). 2.2 stores the key in document_chunks.section.
# The patterns match the text produced by Html2TextTransformer on real AAPL and NVDA 10-Ks:
# - a heading sits alone on its line, so "$" rejects cross-references inside a sentence and the
#   "|" in table-of-contents rows
# - the title is part of the pattern, so "Item 7" never matches "Item 7A"
# - \s+ lets a heading wrap over two lines (Item 7 does), and "\\?\." accepts "8." and "8\."
#   (html2text escapes a period that follows a number)
SECTIONS = {
    "risk_factors": (
        "Item 1A. Risk Factors",
        re.compile(r"^Item\s+1A\\?\.\s+Risk\s+Factors[ \t]*$", re.MULTILINE | re.IGNORECASE),
        re.compile(
            r"^Item\s+(?:1B|1C|2)\\?\.?\s+(?:Unresolved\s+Staff\s+Comments|Cybersecurity|Properties)"
            r"[ \t]*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    "mdna": (
        "Item 7. Management's Discussion and Analysis",
        re.compile(
            r"^Item\s+7\\?\.\s+Management.s\s+Discussion\s+and\s+Analysis\s+of\s+Financial"
            r"\s+Condition\s+and\s+Results\s+of\s+Operations[ \t]*$",
            re.MULTILINE | re.IGNORECASE,
        ),
        re.compile(
            r"^Item\s+(?:7A|8)\\?\.?\s+(?:Quantitative\s+and\s+Qualitative\s+Disclosures\s+about"
            r"\s+Market\s+Risk|Financial\s+Statements\s+and\s+Supplementary\s+Data)[ \t]*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
}


def load_filing_sections(db: Session, filing_id: int) -> list[Document]:
    # 1. Load the filing and its company's ticker
    row = filing_repository.get_by_id_with_ticker(db, filing_id)
    if row is None:
        raise NotFoundError("Filing not found")
    filing, ticker = row

    # 2. Validate: only 10-Ks, and only filings whose document is on disk
    if filing.form_type != "10-K":
        raise ValueError(f"Filing {filing.accession_number} is a {filing.form_type}, not a 10-K")
    if filing.raw_path is None:
        raise ValueError(f"Filing {filing.accession_number} has no downloaded document")

    # 3. Load the raw HTML from data/raw (no network)
    file_path = Path(settings.RAW_DATA_DIR) / filing.raw_path
    raw_documents = TextLoader(str(file_path), encoding="utf-8").load()

    # 4. Convert the HTML to text: block elements become lines, tables become "|" rows
    text = Html2TextTransformer().transform_documents(raw_documents)[0].page_content

    sections: list[Document] = []
    for key, (label, start_pattern, end_pattern) in SECTIONS.items():
        # 5. Every start heading is a candidate that runs to the first end heading after it.
        # The longest candidate wins, which skips the table of contents.
        best_content = None
        found_start = False
        for start_match in start_pattern.finditer(text):
            found_start = True
            end_match = end_pattern.search(text, start_match.end())
            if end_match is None:
                continue
            content = text[start_match.end() : end_match.start()].strip()
            if best_content is None or len(content) > len(best_content):
                best_content = content

        # A missing or too short section is left out with a warning, not an error
        if best_content is None:
            reason = "end heading not found" if found_start else "start heading not found"
        elif len(best_content) < MIN_SECTION_CHARS:
            reason = f"too short ({len(best_content)} chars)"
        else:
            reason = None
        if reason is not None:
            logger.warning(
                "Section not found: ticker=%s accession=%s section=%s reason=%s",
                ticker,
                filing.accession_number,
                key,
                reason,
            )
            continue

        # 6. One Document per section, with plain JSON metadata
        sections.append(
            Document(
                page_content=best_content,
                metadata={
                    "company_id": filing.company_id,
                    "ticker": ticker,
                    "filing_id": filing.id,
                    "form_type": filing.form_type,
                    "fiscal_year": filing.fiscal_year,
                    "section": key,
                },
            )
        )
        logger.info(
            "Parsed %s: ticker=%s accession=%s chars=%d",
            label,
            ticker,
            filing.accession_number,
            len(best_content),
        )

    # 7. Log the result
    logger.info(
        "Filing %s (%s): %d of %d sections found",
        filing.accession_number,
        ticker,
        len(sections),
        len(SECTIONS),
    )
    return sections


def list_scope_filing_ids(db: Session) -> list[int]:
    # The Phase 2 and 3 scope rule, written once: 10-Ks of RAG_TICKERS filed within
    # RAG_LOOKBACK_YEARS (365 days per year)
    tickers = [
        ticker.strip().upper() for ticker in settings.RAG_TICKERS.split(",") if ticker.strip()
    ]
    filed_since = date.today() - timedelta(days=365 * settings.RAG_LOOKBACK_YEARS)
    return filing_repository.list_in_scope_10k(db, tickers, filed_since)
