from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.clients import sec
from app.config import settings
from app.exceptions import ConflictError
from app.models.companies import Company
from app.models.financials import FinancialFact
from app.models.ingestion import IngestionRun
from app.repositories import companies as company_repository
from app.repositories import financials as financial_repository
from app.repositories import ingestion as ingestion_repository
from app.schemas.financials import MetricName
from app.services import financials as financial_service

APPLE_CIK = "0000320193"
MICROSOFT_CIK = "0000789019"


def days_ago(days: int) -> date:
    return date.today() - timedelta(days=days)


# Three fiscal year ends, all inside the 6-year window, and the day each 10-K was filed
# (30 days after the year end). Dates are relative to today so the tests never go stale
YEAR_1_END = days_ago(1100)
YEAR_2_END = days_ago(735)
YEAR_3_END = days_ago(370)
YEAR_1_FILED = YEAR_1_END + timedelta(days=30)
YEAR_2_FILED = YEAR_2_END + timedelta(days=30)
YEAR_3_FILED = YEAR_3_END + timedelta(days=30)


def entry(
    end: date,
    val: float | None,
    accn: str,
    filed: date,
    form: str = "10-K",
    start: date | None = None,
) -> dict:
    # One entry of the real structure. start is left out for point-in-time values and val is
    # left out when val is None
    result = {
        "end": end.isoformat(),
        "accn": accn,
        "fy": filed.year,
        "fp": "FY",
        "form": form,
        "filed": filed.isoformat(),
    }
    if val is not None:
        result["val"] = val
    if start is not None:
        result["start"] = start.isoformat()
    return result


def annual(end: date, val: float | None, accn: str, filed: date, form: str = "10-K") -> dict:
    # A one-year duration entry (364 days)
    return entry(end, val, accn, filed, form, start=end - timedelta(days=364))


def concept_block(unit: str, entries: list[dict]) -> dict:
    return {"label": "x", "description": "x", "units": {unit: entries}}


def make_facts(us_gaap: dict, dei: dict | None = None) -> dict:
    # A company facts file in the real structure: facts -> taxonomy -> concept -> units -> unit
    return {"cik": 1, "entityName": "Test", "facts": {"dei": dei or {}, "us-gaap": us_gaap}}


def make_apple_facts() -> dict:
    revenues = [
        # Year 1: repeated as a comparative in the next two 10-Ks. The last one restates it
        annual(YEAR_1_END, 100, "ACC-1", YEAR_1_FILED),
        annual(YEAR_1_END, 100, "ACC-2", YEAR_2_FILED),
        annual(YEAR_1_END, 105, "ACC-3", YEAR_3_FILED),
        # Year 2: repeated once
        annual(YEAR_2_END, 200, "ACC-2", YEAR_2_FILED),
        annual(YEAR_2_END, 200, "ACC-3", YEAR_3_FILED),
        annual(YEAR_3_END, 300, "ACC-3", YEAR_3_FILED),
        # Excluded: a 10-Q (90 days), a 10-K/A for year 2 filed later with another value, a
        # 90-day duration inside a 10-K, and a period older than the 6-year window
        entry(YEAR_3_END, 70, "Q-1", YEAR_3_FILED, form="10-Q", start=YEAR_3_END - timedelta(90)),
        annual(YEAR_2_END, 999, "AMEND-1", YEAR_3_FILED + timedelta(days=5), form="10-K/A"),
        entry(YEAR_3_END, 77, "ACC-3", YEAR_3_FILED, start=YEAR_3_END - timedelta(days=90)),
        annual(days_ago(365 * 6 + 50), 11, "ACC-OLD", days_ago(365 * 6 + 20)),
        # Skipped: no "val"
        annual(days_ago(1830), None, "ACC-X", days_ago(1800)),
    ]
    # An older year tagged with the old revenue concept
    old_revenue = [annual(days_ago(1465), 50, "ACC-0", days_ago(1435))]
    assets = [
        # Point-in-time values have no start, and are repeated across filings
        entry(YEAR_1_END, 1000, "ACC-1", YEAR_1_FILED),
        entry(YEAR_1_END, 1000, "ACC-2", YEAR_2_FILED),
        entry(YEAR_2_END, 2000, "ACC-2", YEAR_2_FILED),
        entry(YEAR_2_END, 2000, "ACC-3", YEAR_3_FILED),
        entry(YEAR_3_END, 3000, "ACC-3", YEAR_3_FILED),
    ]
    eps = [
        annual(YEAR_2_END, 5.5, "ACC-2", YEAR_2_FILED),
        annual(YEAR_3_END, 6.08, "ACC-3", YEAR_3_FILED),
    ]
    return make_facts(
        {
            "Revenues": concept_block("USD", revenues),
            "SalesRevenueNet": concept_block("USD", old_revenue),
            "Assets": concept_block("USD", assets),
            "EarningsPerShareDiluted": concept_block("USD/shares", eps),
            # Not in the allowlist
            "SomeOtherConcept": concept_block(
                "USD", [annual(YEAR_3_END, 5, "ACC-3", YEAR_3_FILED)]
            ),
        },
        # A different taxonomy, ignored
        dei={
            "EntityPublicFloat": concept_block(
                "USD", [annual(YEAR_3_END, 9, "ACC-3", YEAR_3_FILED)]
            )
        },
    )


def make_microsoft_facts() -> dict:
    net_income = [annual(YEAR_3_END, 40, "MS-1", YEAR_3_FILED)]
    return make_facts({"NetIncomeLoss": concept_block("USD", net_income)})


@pytest.fixture
def companies(db: Session) -> dict[str, Company]:
    apple = company_repository.create(db, "AAPL", APPLE_CIK, "Apple Inc.", "Nasdaq")
    microsoft = company_repository.create(db, "MSFT", MICROSOFT_CIK, "Microsoft Corp", "Nasdaq")
    db.commit()
    return {"AAPL": apple, "MSFT": microsoft}


@pytest.fixture
def fake_sec(monkeypatch: pytest.MonkeyPatch) -> dict:
    # Replaces sec.get_company_facts so nothing touches the network. Tests fill in what the
    # "SEC" returns per CIK (a facts dict, or an exception to raise)
    state: dict = {"facts": {APPLE_CIK: make_apple_facts(), MICROSOFT_CIK: make_microsoft_facts()}}

    def fake_get_company_facts(cik: str) -> dict:
        result = state["facts"][cik]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(sec, "get_company_facts", fake_get_company_facts)
    return state


def count_facts(db: Session) -> int:
    return db.execute(select(func.count()).select_from(FinancialFact)).scalar_one()


def apple_facts_by_key(db: Session, company: Company) -> dict[tuple[str, date], FinancialFact]:
    rows = financial_repository.get_by_company(db, company.id)
    return {(row.concept, row.period_end): row for row in rows}


def test_first_run_stores_exactly_the_expected_rows(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    run = financial_service.ingest_financial_facts(db)

    rows = apple_facts_by_key(db, companies["AAPL"])
    assert set(rows) == {
        ("Revenues", YEAR_1_END),
        ("Revenues", YEAR_2_END),
        ("Revenues", YEAR_3_END),
        ("SalesRevenueNet", days_ago(1465)),
        ("Assets", YEAR_1_END),
        ("Assets", YEAR_2_END),
        ("Assets", YEAR_3_END),
        ("EarningsPerShareDiluted", YEAR_2_END),
        ("EarningsPerShareDiluted", YEAR_3_END),
    }

    # Year 1 was restated in the latest 10-K: the restated value and that filing win
    year_1 = rows[("Revenues", YEAR_1_END)]
    assert year_1.value == Decimal("105")
    assert year_1.accession_number == "ACC-3"
    assert year_1.filed_on == YEAR_3_FILED
    assert year_1.period_start == YEAR_1_END - timedelta(days=364)
    assert year_1.unit == "USD"
    assert year_1.form_type == "10-K"
    # The year of period_end, not the filing's fy
    assert year_1.fiscal_year == YEAR_1_END.year

    # The 10-K/A (999) did not replace year 2
    year_2 = rows[("Revenues", YEAR_2_END)]
    assert year_2.value == Decimal("200")
    assert year_2.accession_number == "ACC-3"

    # Point-in-time values have no start date, and the latest filing wins
    assets = rows[("Assets", YEAR_1_END)]
    assert assets.period_start is None
    assert assets.value == Decimal("1000")
    assert assets.accession_number == "ACC-2"
    assert assets.filed_on == YEAR_2_FILED

    # The decimal EPS is stored exactly, in its own unit
    eps = rows[("EarningsPerShareDiluted", YEAR_3_END)]
    assert eps.value == Decimal("6.08")
    assert eps.unit == "USD/shares"

    assert rows[("SalesRevenueNet", days_ago(1465))].value == Decimal("50")

    # Microsoft was ingested too
    assert len(financial_repository.get_by_company(db, companies["MSFT"].id)) == 1

    assert count_facts(db) == 10
    assert run.status == "success"
    assert run.job_type == "financial_facts"
    assert run.finished_at is not None
    assert run.error is None
    assert run.message == (
        "Companies processed: 2, companies failed: 0, companies without facts: 0, "
        "facts created: 10, facts updated: 0"
    )


def test_second_run_with_the_same_data_changes_nothing(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    financial_service.ingest_financial_facts(db)
    assert count_facts(db) == 10

    run = financial_service.ingest_financial_facts(db)

    # The point-in-time rows have a NULL start date, so duplicates there would show up here
    assert count_facts(db) == 10
    assert run.status == "success"
    assert "facts created: 0, facts updated: 0" in run.message


def test_a_newer_filing_updates_the_stored_row_in_place(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    financial_service.ingest_financial_facts(db)
    year_2_before = apple_facts_by_key(db, companies["AAPL"])[("Revenues", YEAR_2_END)]
    row_id = year_2_before.id

    newer_filed = days_ago(5)
    fake_sec["facts"][APPLE_CIK] = make_facts(
        {"Revenues": concept_block("USD", [annual(YEAR_2_END, 222, "ACC-NEW", newer_filed)])}
    )
    run = financial_service.ingest_financial_facts(db)

    assert count_facts(db) == 10
    assert "facts created: 0, facts updated: 1" in run.message
    year_2 = apple_facts_by_key(db, companies["AAPL"])[("Revenues", YEAR_2_END)]
    assert year_2.id == row_id
    assert year_2.value == Decimal("222")
    assert year_2.accession_number == "ACC-NEW"
    assert year_2.filed_on == newer_filed


def test_an_older_filing_does_not_overwrite_the_stored_row(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    financial_service.ingest_financial_facts(db)

    # Year 3 is stored from the filing of YEAR_3_FILED. This entry was filed 20 days earlier
    fake_sec["facts"][APPLE_CIK] = make_facts(
        {
            "Revenues": concept_block(
                "USD", [annual(YEAR_3_END, 1, "ACC-OLD", YEAR_3_FILED - timedelta(days=20))]
            )
        }
    )
    run = financial_service.ingest_financial_facts(db)

    year_3 = apple_facts_by_key(db, companies["AAPL"])[("Revenues", YEAR_3_END)]
    assert year_3.value == Decimal("300")
    assert year_3.accession_number == "ACC-3"
    assert "facts created: 0, facts updated: 0" in run.message
    assert count_facts(db) == 10


def test_the_unique_constraint_covers_a_null_period_start(
    db: Session, companies: dict[str, Company]
) -> None:
    company_id = companies["AAPL"].id

    def insert_balance_sheet_row() -> None:
        financial_repository.create(
            db,
            company_id,
            "Assets",
            "USD",
            None,
            YEAR_3_END,
            Decimal("1"),
            YEAR_3_END.year,
            "10-K",
            "ACC-1",
            YEAR_3_FILED,
        )

    insert_balance_sheet_row()
    with pytest.raises(IntegrityError):
        insert_balance_sheet_row()
    db.rollback()


def test_one_failing_company_does_not_stop_the_others(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    fake_sec["facts"][APPLE_CIK] = httpx.ConnectError("connection refused")

    run = financial_service.ingest_financial_facts(db)

    assert run.status == "partial"
    assert "Failed companies: AAPL" in run.message
    assert "companies failed: 1" in run.message
    # Microsoft was stored, Apple nothing
    assert financial_repository.get_by_company(db, companies["AAPL"].id) == []
    assert len(financial_repository.get_by_company(db, companies["MSFT"].id)) == 1


def test_a_company_without_us_gaap_facts_is_not_a_failure(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    fake_sec["facts"][APPLE_CIK] = make_facts({})

    run = financial_service.ingest_financial_facts(db)

    assert run.status == "success"
    assert "companies without facts: 1" in run.message
    assert "companies failed: 0" in run.message
    assert financial_repository.get_by_company(db, companies["AAPL"].id) == []
    assert len(financial_repository.get_by_company(db, companies["MSFT"].id)) == 1


def start_running_run(db: Session, job_type: str, started_ago: timedelta) -> None:
    run = ingestion_repository.create_run(db, job_type)
    run.started_at = datetime.now(UTC) - started_ago
    db.commit()


def test_a_recent_running_run_blocks_a_new_one(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    start_running_run(db, "financial_facts", timedelta(minutes=10))

    with pytest.raises(ConflictError) as error:
        financial_service.ingest_financial_facts(db)

    assert error.value.message == "A financial facts ingestion run is already in progress"
    # Nothing was created: only the existing run row, and no facts
    assert db.execute(select(func.count()).select_from(IngestionRun)).scalar_one() == 1
    assert count_facts(db) == 0


def test_a_stale_running_run_does_not_block(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    start_running_run(db, "financial_facts", timedelta(hours=3))

    run = financial_service.ingest_financial_facts(db)

    assert run.status == "success"


def test_a_running_filings_run_does_not_block(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    start_running_run(db, "ingest_filings", timedelta(minutes=10))

    run = financial_service.ingest_financial_facts(db)

    assert run.status == "success"


def test_an_unexpected_error_marks_the_run_failed_and_is_raised(
    db: Session, companies: dict[str, Company], fake_sec: dict
) -> None:
    fake_sec["facts"][APPLE_CIK] = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        financial_service.ingest_financial_facts(db)

    run = db.execute(select(IngestionRun)).scalar_one()
    assert run.status == "failed"
    assert "boom" in run.error
    assert run.finished_at is not None


def test_the_lookback_window_comes_from_settings(
    db: Session,
    companies: dict[str, Company],
    fake_sec: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 2-year window (730 days) keeps only year 3 (ended 370 days ago)
    monkeypatch.setattr(settings, "FINANCIALS_LOOKBACK_YEARS", 2)

    financial_service.ingest_financial_facts(db)

    rows = apple_facts_by_key(db, companies["AAPL"])
    assert {period_end for _, period_end in rows} == {YEAR_3_END}


def test_every_metric_has_an_entry_and_every_concept_is_used_once() -> None:
    assert set(financial_service.METRICS) == set(MetricName)

    all_concepts = [
        concept for _, _, concepts in financial_service.METRICS.values() for concept in concepts
    ]
    assert len(all_concepts) == len(set(all_concepts))


def test_celery_wiring() -> None:
    # No database needed: only the Celery app is inspected
    from app.workers import tasks  # noqa: F401  (importing registers the task)
    from app.workers.celery_app import celery_app

    assert "ingest_financial_facts" in celery_app.tasks

    schedule_entries = [
        schedule_entry
        for schedule_entry in celery_app.conf.beat_schedule.values()
        if schedule_entry["task"] == "ingest_financial_facts"
    ]
    assert len(schedule_entries) == 1
    # Every day at 03:00 UTC, one hour after the filings job
    assert schedule_entries[0]["schedule"].hour == {3}
    assert schedule_entries[0]["schedule"].minute == {0}
