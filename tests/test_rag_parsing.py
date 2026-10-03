import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.config import settings
from app.exceptions import NotFoundError
from app.models.companies import Company
from app.models.filings import Filing
from app.rag import parsing
from app.rag.parsing import SECTIONS, list_scope_filing_ids, load_filing_sections
from app.repositories import companies as company_repository
from app.repositories import filings as filing_repository

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "filings"

APPLE_CIK = "0000320193"
NVIDIA_CIK = "0001045810"
MICROSOFT_CIK = "0000789019"

# Plain texts for the small inline HTML cases. They contain no "Item" so they can never look
# like a heading, and each one is different so a test can tell which one was returned
RISK_TEXT = "Supply chain disruptions could harm our results of operations. " * 10
MDNA_TEXT = "Revenue grew because customers bought more of our products. " * 10
MARKET_RISK_TEXT = "Interest rate changes could affect the value of our investments. " * 30

TOC_AND_HEADINGS = {
    "1A": "Item 1A. Risk Factors",
    "1B": "Item 1B. Unresolved Staff Comments",
    "7": "Item 7. Management&#8217;s Discussion and Analysis of Financial Condition and "
    "Results of Operations",
    "7A": "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Item 8. Financial Statements and Supplementary Data",
}


def div(text: str) -> str:
    return f"<div><span>{text}</span></div>"


def heading(item: str) -> str:
    return div(TOC_AND_HEADINGS[item])


def make_company(db: Session, ticker: str, cik: str) -> Company:
    return company_repository.create(db, ticker, cik, f"{ticker} Inc.", None)


def make_filing(
    db: Session,
    company: Company,
    accession_number: str,
    form_type: str,
    filed_on: date,
    raw_path: str | None,
) -> Filing:
    filing = filing_repository.create(
        db,
        company.id,
        accession_number,
        form_type,
        filed_on,
        report_date=filed_on,
        fiscal_year=filed_on.year,
        primary_document="document.htm",
    )
    # The repository's create does not take raw_path, the download step sets it
    filing.raw_path = raw_path
    db.flush()
    return filing


@pytest.fixture
def inline_filing(db: Session, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Builds a 10-K whose document is the given HTML body, stored in tmp_path
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(parsing, "MIN_SECTION_CHARS", 200)
    company = make_company(db, "AAPL", APPLE_CIK)

    def build(body: str) -> Filing:
        (tmp_path / "inline.htm").write_text(f"<html><body>{body}</body></html>", encoding="utf-8")
        return make_filing(
            db, company, "0000000000-26-000001", "10-K", date(2026, 1, 15), "inline.htm"
        )

    return build


def normalize(text: str) -> str:
    # The text converter wraps lines, so compare with all whitespace collapsed
    return " ".join(text.split())


# ---------- real filing excerpts ----------


@pytest.mark.parametrize(
    "file_name, ticker, cik, fiscal_year, expected",
    [
        (
            "aapl_10k_excerpt.htm",
            "AAPL",
            APPLE_CIK,
            2024,
            {
                "risk_factors": (
                    "The Company's business, reputation, results of operations, "
                    "financial condition",
                    "financial condition and stock price.",
                ),
                "mdna": (
                    "The following discussion should be read in conjunction with the consolidated",
                    "relative to the U.S. dollar.",
                ),
            },
        ),
        (
            "nvda_10k_excerpt.htm",
            "NVDA",
            NVIDIA_CIK,
            2025,
            {
                "risk_factors": (
                    "The following risk factors should be considered in addition to the other "
                    "information in this Annual Report on Form 10-K.",
                    "Failure to meet the evolving needs of our industry and markets may adversely "
                    "impact our financial results.",
                ),
                "mdna": (
                    "The following discussion and analysis of our financial condition and results "
                    "of operations should be read in conjunction with",
                    "No customer represented 10% or more of total revenue for fiscal year 2023.",
                ),
            },
        ),
    ],
)
def test_real_filing_excerpt(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    file_name: str,
    ticker: str,
    cik: str,
    fiscal_year: int,
    expected: dict[str, tuple[str, str]],
) -> None:
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(FIXTURES_DIR))
    company = make_company(db, ticker, cik)
    filing = make_filing(db, company, "0000000000-26-000099", "10-K", date(2026, 1, 15), file_name)
    filing.fiscal_year = fiscal_year

    documents = load_filing_sections(db, filing.id)

    # Exactly the two sections, in SECTIONS order
    assert [document.metadata["section"] for document in documents] == list(SECTIONS)

    for document in documents:
        key = document.metadata["section"]

        # Exactly the six metadata keys, all plain JSON values
        assert document.metadata == {
            "company_id": company.id,
            "ticker": ticker,
            "filing_id": filing.id,
            "form_type": "10-K",
            "fiscal_year": fiscal_year,
            "section": key,
        }
        json.dumps(document.metadata)

        # The section starts with its real first sentence and ends with its real last sentence
        text = normalize(document.page_content)
        first_phrase, last_phrase = expected[key]
        assert text.startswith(first_phrase)
        assert text.endswith(last_phrase)
        assert len(document.page_content) >= 3000

        # No table-of-contents rows ("Item 1A.| Risk Factors| 5") and no heading of any kind
        assert re.search(r"^Item\s+\w+\\?\.?\|", document.page_content, re.MULTILINE) is None
        _, start_pattern, end_pattern = SECTIONS[key]
        assert start_pattern.search(document.page_content) is None
        assert end_pattern.search(document.page_content) is None

    # The MD&A excerpt keeps one table, which is flattened to "|" rows
    assert "|" in documents[1].page_content


# ---------- small inline cases ----------


def test_table_of_contents_entry_loses_to_the_real_heading(db: Session, inline_filing) -> None:
    # A table of contents written as plain lines: its candidates are tiny and the real ones win
    toc = heading("1A") + heading("1B") + heading("7") + heading("7A")
    body = (
        toc
        + heading("1A")
        + div(RISK_TEXT)
        + heading("1B")
        + div("None.")
        + heading("7")
        + div(MDNA_TEXT)
        + heading("7A")
        + div("None.")
    )
    filing = inline_filing(body)

    documents = load_filing_sections(db, filing.id)

    assert [document.metadata["section"] for document in documents] == ["risk_factors", "mdna"]
    assert normalize(documents[0].page_content) == normalize(RISK_TEXT)
    assert normalize(documents[1].page_content) == normalize(MDNA_TEXT)


def test_item_7a_is_not_taken_as_item_7(db: Session, inline_filing) -> None:
    # The Item 7A text is longer than the MD&A text, so if "Item 7A" were taken as a start
    # heading it would win the longest-candidate rule
    body = (
        heading("7")
        + div(MDNA_TEXT)
        + heading("7A")
        + div(MARKET_RISK_TEXT)
        + heading("8")
        + div("None.")
    )
    filing = inline_filing(body)

    documents = load_filing_sections(db, filing.id)

    assert [document.metadata["section"] for document in documents] == ["mdna"]
    assert normalize(documents[0].page_content) == normalize(MDNA_TEXT)


@pytest.mark.parametrize(
    "mdna_html, reason",
    [
        # No Item 7 heading at all
        ("", "start heading not found"),
        # A heading, but fewer characters than MIN_SECTION_CHARS (200 in these tests)
        (heading("7") + div("Short text.") + heading("7A") + div("None."), "too short"),
        # A heading and enough text, but nothing that ends the section
        (heading("7") + div(MDNA_TEXT), "end heading not found"),
    ],
)
def test_missing_section_is_left_out_with_a_warning(
    db: Session,
    inline_filing,
    caplog: pytest.LogCaptureFixture,
    mdna_html: str,
    reason: str,
) -> None:
    body = heading("1A") + div(RISK_TEXT) + heading("1B") + div("None.") + mdna_html
    filing = inline_filing(body)
    # alembic's fileConfig can disable existing loggers in the test process (see Known issues)
    logging.getLogger("app.rag.parsing").disabled = False

    with caplog.at_level(logging.WARNING, logger="app.rag.parsing"):
        documents = load_filing_sections(db, filing.id)

    # Only the Risk Factors Document is returned, and nothing raised
    assert [document.metadata["section"] for document in documents] == ["risk_factors"]
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "ticker=AAPL" in warnings[0]
    assert "accession=0000000000-26-000001" in warnings[0]
    assert "section=mdna" in warnings[0]
    assert reason in warnings[0]


# ---------- errors ----------


def test_a_10q_is_rejected(db: Session) -> None:
    company = make_company(db, "AAPL", APPLE_CIK)
    filing = make_filing(db, company, "0000000000-26-000002", "10-Q", date(2026, 1, 15), "x.htm")

    with pytest.raises(ValueError, match="not a 10-K"):
        load_filing_sections(db, filing.id)


def test_a_filing_without_a_downloaded_document_is_rejected(db: Session) -> None:
    company = make_company(db, "AAPL", APPLE_CIK)
    filing = make_filing(db, company, "0000000000-26-000003", "10-K", date(2026, 1, 15), None)

    with pytest.raises(ValueError, match="no downloaded document"):
        load_filing_sections(db, filing.id)


def test_an_unknown_filing_id_is_not_found(db: Session) -> None:
    with pytest.raises(NotFoundError):
        load_filing_sections(db, 999_999)


# ---------- scope ----------


def test_scope_returns_only_in_scope_10ks(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA")
    monkeypatch.setattr(settings, "RAG_LOOKBACK_YEARS", 2)
    today = date.today()
    apple = make_company(db, "AAPL", APPLE_CIK)
    nvidia = make_company(db, "NVDA", NVIDIA_CIK)
    microsoft = make_company(db, "MSFT", MICROSOFT_CIK)

    # 10-Ks filed 100, 500 and 900 days ago (the lookback is 730 days)
    expected_ids = []
    number = 0
    for company in (apple, nvidia, microsoft):
        for days_ago in (100, 500, 900):
            number += 1
            filing = make_filing(
                db,
                company,
                f"0000000000-26-{number:06d}",
                "10-K",
                today - timedelta(days=days_ago),
                "x.htm",
            )
            if company is not microsoft and days_ago != 900:
                expected_ids.append(filing.id)

    # Same company and age, but a 10-Q, and a 10-K whose document was never downloaded
    make_filing(db, apple, "0000000000-26-000100", "10-Q", today - timedelta(days=100), "x.htm")
    make_filing(db, apple, "0000000000-26-000101", "10-K", today - timedelta(days=200), None)

    ids = list_scope_filing_ids(db)

    assert sorted(ids) == sorted(expected_ids)
    assert len(ids) == 4

    # The ticker setting is split and uppercased where it is used
    monkeypatch.setattr(settings, "RAG_TICKERS", " aapl , Nvda ")
    assert sorted(list_scope_filing_ids(db)) == sorted(expected_ids)
