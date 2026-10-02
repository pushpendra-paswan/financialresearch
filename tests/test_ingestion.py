from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.clients import sec
from app.config import settings
from app.exceptions import ConflictError
from app.models.companies import Company
from app.models.filings import Filing
from app.models.ingestion import IngestionRun
from app.repositories import companies as company_repository
from app.repositories import ingestion as ingestion_repository
from app.services import ingestion as ingestion_service

APPLE_CIK = "0000320193"
MICROSOFT_CIK = "0000789019"
BANK_CIK = "0000019617"


def days_ago(days: int) -> date:
    return date.today() - timedelta(days=days)


def make_submissions(
    rows: list[tuple[str, date, date | None, str, str]],
    industry: str | None = "Electronic Computers",
    older_pages: list[dict] | None = None,
) -> dict:
    # Builds a submissions file in the real structure: parallel lists inside filings.recent.
    # Each row is (accession_number, filing_date, report_date, form, primary_document).
    # Empty strings stand for missing values, as in the real file.
    return {
        "sicDescription": industry,
        "filings": {
            "recent": {
                "accessionNumber": [row[0] for row in rows],
                "filingDate": [row[1].isoformat() for row in rows],
                "reportDate": [row[2].isoformat() if row[2] else "" for row in rows],
                "form": [row[3] for row in rows],
                "primaryDocument": [row[4] for row in rows],
            },
            "files": older_pages or [],
        },
    }


def make_standard_rows(prefix: str) -> list[tuple[str, date, date | None, str, str]]:
    # The three-year window is 365 * 3 days. Dates are relative to today so the tests never go
    # stale. Only the first two rows should be stored.
    return [
        (f"{prefix}-26-000001", days_ago(100), days_ago(130), "10-K", f"{prefix}-10k.htm"),
        (f"{prefix}-26-000002", days_ago(30), days_ago(60), "10-Q", f"{prefix}-10q.htm"),
        # Older than the lookback window
        (f"{prefix}-20-000003", days_ago(365 * 3 + 30), days_ago(365 * 3 + 60), "10-K", "old.htm"),
        (f"{prefix}-26-000004", days_ago(10), None, "8-K", "eight.htm"),
        (f"{prefix}-26-000005", days_ago(50), days_ago(80), "10-K/A", "amendment.htm"),
        (f"{prefix}-26-000006", days_ago(5), days_ago(6), "4", "xslF345X06/form4.xml"),
        # A 10-Q with no primary document is skipped
        (f"{prefix}-26-000007", days_ago(20), days_ago(50), "10-Q", ""),
    ]


@pytest.fixture
def companies(db: Session) -> dict[str, Company]:
    apple = company_repository.create(db, "AAPL", APPLE_CIK, "Apple Inc.", "Nasdaq")
    microsoft = company_repository.create(db, "MSFT", MICROSOFT_CIK, "Microsoft Corp", "Nasdaq")
    db.commit()
    return {"AAPL": apple, "MSFT": microsoft}


@pytest.fixture
def fake_sec(monkeypatch: pytest.MonkeyPatch) -> dict:
    # Replaces the two SEC client functions the service uses, so nothing touches the network.
    # Tests fill in what the "SEC" returns and read what was requested.
    state: dict = {
        "submissions": {},  # cik -> submissions dict, or an exception to raise
        "pages": {},  # older page name -> page dict
        "failing_downloads": set(),  # accession numbers whose download raises
        "requested_pages": [],
        "downloads": [],  # (cik, accession_number) for every successful download
    }

    def fake_get_submissions(cik: str, page_name: str | None = None) -> dict:
        if page_name:
            state["requested_pages"].append(page_name)
            return state["pages"][page_name]
        result = state["submissions"][cik]
        if isinstance(result, Exception):
            raise result
        return result

    def fake_download(cik: str, accession_number: str, primary_document: str) -> str:
        if accession_number in state["failing_downloads"]:
            raise httpx.ConnectError("connection refused")
        state["downloads"].append((cik, accession_number))
        return f"sec/filings/{cik}/{accession_number}/{primary_document}"

    monkeypatch.setattr(sec, "get_submissions", fake_get_submissions)
    monkeypatch.setattr(sec, "download_filing_document", fake_download)
    return state


def stored_filings(db: Session, company: Company) -> list[Filing]:
    statement = select(Filing).where(Filing.company_id == company.id).order_by(Filing.filed_on)
    return list(db.execute(statement).scalars().all())


def count_rows(db: Session, model: type) -> int:
    return db.execute(select(func.count()).select_from(model)).scalar_one()


def test_first_run_stores_only_recent_10k_and_10q(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    fake_sec["submissions"][APPLE_CIK] = make_submissions(make_standard_rows("AAPL"))
    fake_sec["submissions"][MICROSOFT_CIK] = make_submissions(
        make_standard_rows("MSFT"), industry="Services-Prepackaged Software"
    )

    run = ingestion_service.ingest_filings(db)

    # Only the recent 10-K and the recent 10-Q are stored (oldest filed first)
    filings = stored_filings(db, companies["AAPL"])
    assert [filing.accession_number for filing in filings] == ["AAPL-26-000001", "AAPL-26-000002"]
    ten_k, ten_q = filings
    assert ten_k.form_type == "10-K"
    assert ten_k.filed_on == days_ago(100)
    assert ten_k.report_date == days_ago(130)
    assert ten_k.fiscal_year == days_ago(130).year
    assert ten_k.primary_document == "AAPL-10k.htm"
    assert ten_q.form_type == "10-Q"
    assert ten_q.fiscal_year == days_ago(60).year

    # The industry is the SIC description, per company
    assert companies["AAPL"].industry == "Electronic Computers"
    assert companies["MSFT"].industry == "Services-Prepackaged Software"

    # Both documents were downloaded and their relative paths stored
    assert ten_k.raw_path == f"sec/filings/{APPLE_CIK}/AAPL-26-000001/AAPL-10k.htm"
    assert ten_q.raw_path == f"sec/filings/{APPLE_CIK}/AAPL-26-000002/AAPL-10q.htm"
    assert count_rows(db, Filing) == 4

    # The run is recorded
    assert run.status == "success"
    assert run.finished_at is not None
    assert run.error is None
    assert run.message == (
        "Companies processed: 2, companies failed: 0, new filings: 4, "
        "documents downloaded: 4, documents failed: 0"
    )


def test_a_missing_report_date_gives_a_null_fiscal_year(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    rows = [("AAPL-26-000001", days_ago(10), None, "10-Q", "aapl.htm")]
    fake_sec["submissions"][APPLE_CIK] = make_submissions(rows)
    fake_sec["submissions"][MICROSOFT_CIK] = make_submissions([])

    ingestion_service.ingest_filings(db)

    filing = stored_filings(db, companies["AAPL"])[0]
    assert filing.report_date is None
    assert filing.fiscal_year is None


def test_second_run_creates_and_downloads_nothing(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    fake_sec["submissions"][APPLE_CIK] = make_submissions(make_standard_rows("AAPL"))
    fake_sec["submissions"][MICROSOFT_CIK] = make_submissions(make_standard_rows("MSFT"))
    first_run = ingestion_service.ingest_filings(db)
    downloads_after_first_run = len(fake_sec["downloads"])

    second_run = ingestion_service.ingest_filings(db)

    assert count_rows(db, Filing) == 4
    assert len(fake_sec["downloads"]) == downloads_after_first_run
    assert second_run.id != first_run.id
    assert count_rows(db, IngestionRun) == 2
    assert second_run.status == "success"
    assert second_run.message == (
        "Companies processed: 2, companies failed: 0, new filings: 0, "
        "documents downloaded: 0, documents failed: 0"
    )


def test_one_failing_company_does_not_stop_the_others(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    fake_sec["submissions"][APPLE_CIK] = httpx.ConnectError("connection refused")
    fake_sec["submissions"][MICROSOFT_CIK] = make_submissions(make_standard_rows("MSFT"))

    run = ingestion_service.ingest_filings(db)

    assert stored_filings(db, companies["AAPL"]) == []
    assert len(stored_filings(db, companies["MSFT"])) == 2
    assert run.status == "partial"
    assert run.finished_at is not None
    assert "companies failed: 1" in run.message
    assert "Failed companies: AAPL" in run.message


def test_malformed_submissions_data_fails_only_that_company(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    fake_sec["submissions"][APPLE_CIK] = {"sicDescription": "Electronic Computers"}
    fake_sec["submissions"][MICROSOFT_CIK] = make_submissions(make_standard_rows("MSFT"))

    run = ingestion_service.ingest_filings(db)

    assert run.status == "partial"
    assert len(stored_filings(db, companies["MSFT"])) == 2


def test_a_failed_download_is_retried_by_the_next_run(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    fake_sec["submissions"][APPLE_CIK] = make_submissions(make_standard_rows("AAPL"))
    fake_sec["submissions"][MICROSOFT_CIK] = make_submissions([])
    fake_sec["failing_downloads"] = {"AAPL-26-000002"}

    first_run = ingestion_service.ingest_filings(db)

    # The filing is stored, only its document is missing
    ten_k, ten_q = stored_filings(db, companies["AAPL"])
    assert ten_k.raw_path is not None
    assert ten_q.raw_path is None
    assert first_run.status == "partial"
    assert "documents failed: 1" in first_run.message

    # The download works now: the same filing gets its document, and nothing is duplicated
    fake_sec["failing_downloads"] = set()
    second_run = ingestion_service.ingest_filings(db)

    assert count_rows(db, Filing) == 2
    assert ten_q.raw_path == f"sec/filings/{APPLE_CIK}/AAPL-26-000002/AAPL-10q.htm"
    assert second_run.status == "success"
    assert "new filings: 0" in second_run.message
    assert "documents downloaded: 1" in second_run.message


def test_older_pages_are_fetched_only_when_recent_does_not_reach_the_window(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    # A company that files so often that "recent" covers only the last 100 days
    older_pages = [
        {"name": "page-1.json", "filingFrom": "x", "filingTo": days_ago(101).isoformat()},
        {"name": "page-2.json", "filingFrom": "x", "filingTo": days_ago(365 * 3 + 100).isoformat()},
    ]
    recent_rows = [("AAPL-26-000001", days_ago(10), days_ago(40), "10-Q", "aapl-q.htm")]
    fake_sec["submissions"][APPLE_CIK] = make_submissions(recent_rows, older_pages=older_pages)
    page_one = make_submissions(
        [
            ("AAPL-25-000001", days_ago(400), days_ago(430), "10-K", "aapl-k.htm"),
            ("AAPL-20-000001", days_ago(365 * 3 + 50), days_ago(365 * 3 + 80), "10-K", "old.htm"),
        ]
    )
    fake_sec["pages"]["page-1.json"] = page_one["filings"]["recent"]
    # Microsoft's recent block reaches back past the window, so no pages are fetched for it
    fake_sec["submissions"][MICROSOFT_CIK] = make_submissions(
        make_standard_rows("MSFT"), older_pages=older_pages
    )

    ingestion_service.ingest_filings(db)

    # page-1 overlaps the window and is fetched; page-2 ends before the window and is not
    assert fake_sec["requested_pages"] == ["page-1.json"]
    accession_numbers = [f.accession_number for f in stored_filings(db, companies["AAPL"])]
    assert accession_numbers == ["AAPL-25-000001", "AAPL-26-000001"]


def test_a_recent_running_run_blocks_a_new_one(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    running = ingestion_repository.create_run(db, ingestion_service.JOB_TYPE)
    running.started_at = datetime.now(UTC) - timedelta(minutes=10)
    db.commit()

    with pytest.raises(ConflictError, match="already in progress"):
        ingestion_service.ingest_filings(db)

    # Nothing was created and nothing was requested
    assert count_rows(db, IngestionRun) == 1
    assert fake_sec["downloads"] == []


def test_an_old_running_run_does_not_block(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    stale = ingestion_repository.create_run(db, ingestion_service.JOB_TYPE)
    stale.started_at = datetime.now(UTC) - timedelta(hours=3)
    db.commit()
    fake_sec["submissions"][APPLE_CIK] = make_submissions(make_standard_rows("AAPL"))
    fake_sec["submissions"][MICROSOFT_CIK] = make_submissions([])

    run = ingestion_service.ingest_filings(db)

    assert run.status == "success"
    assert count_rows(db, IngestionRun) == 2


def test_an_unexpected_error_fails_the_run_and_is_re_raised(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    fake_sec["submissions"][APPLE_CIK] = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        ingestion_service.ingest_filings(db)

    run = db.execute(select(IngestionRun)).scalar_one()
    assert run.status == "failed"
    assert "boom" in run.error
    assert run.finished_at is not None


def test_the_lookback_window_comes_from_settings(
    db: Session, companies: dict[str, Company], fake_sec: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "FILINGS_LOOKBACK_YEARS", 1)
    rows = [
        ("AAPL-26-000001", days_ago(100), days_ago(130), "10-K", "new.htm"),
        ("AAPL-24-000001", days_ago(500), days_ago(530), "10-K", "older.htm"),
    ]
    fake_sec["submissions"][APPLE_CIK] = make_submissions(rows)
    fake_sec["submissions"][MICROSOFT_CIK] = make_submissions([])

    ingestion_service.ingest_filings(db)

    assert [f.accession_number for f in stored_filings(db, companies["AAPL"])] == ["AAPL-26-000001"]


def test_celery_wiring() -> None:
    # No database needed: only the Celery app is inspected
    from app.workers import tasks  # noqa: F401  (importing registers the task)
    from app.workers.celery_app import celery_app

    assert "ingest_filings" in celery_app.tasks
    assert "app.workers.tasks" in celery_app.conf.include
    assert celery_app.conf.timezone == "UTC"

    entries = [
        entry
        for entry in celery_app.conf.beat_schedule.values()
        if entry["task"] == "ingest_filings"
    ]
    assert len(entries) == 1
    # Every day at 02:00 UTC
    assert entries[0]["schedule"].hour == {2}
    assert entries[0]["schedule"].minute == {0}
