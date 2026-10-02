import time
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.clients import prices as price_client
from app.config import settings
from app.exceptions import ConflictError
from app.models.companies import Company
from app.models.ingestion import IngestionRun
from app.models.prices import PriceBar
from app.repositories import companies as company_repository
from app.repositories import ingestion as ingestion_repository
from app.repositories import prices as price_repository
from app.services import prices as price_service

EASTERN = ZoneInfo("America/New_York")
# A fixed Wednesday, so the tests are deterministic and never go stale
TODAY = date(2026, 9, 30)
AFTER_CLOSE = datetime(2026, 9, 30, 17, 30, tzinfo=EASTERN)
BEFORE_CLOSE = datetime(2026, 9, 30, 12, 0, tzinfo=EASTERN)
BAR_COUNT = 20


def make_bars(last_day: date, count: int = BAR_COUNT) -> list[dict]:
    # `count` weekday bars ending on last_day (which must be a weekday), oldest first. Values
    # follow the position in the list, so every bar is different and always the same
    days: list[date] = []
    day = last_day
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    days.reverse()

    bars = []
    for position, trade_date in enumerate(days):
        bars.append(
            {
                "trade_date": trade_date,
                "open": Decimal(100 + position),
                "high": Decimal(103 + position),
                "low": Decimal(99 + position),
                "close": Decimal(f"{101 + position}.2500"),
                "adj_close": Decimal(f"{100 + position}.5000"),
                "volume": 1_000_000 + position,
            }
        )
    return bars


class FakeProvider(price_client.PriceProvider):
    # A second PriceProvider implementation: proves a new provider needs only a new class.
    # data maps a ticker to its bars, or to an exception to raise
    def __init__(self, data: dict[str, list[dict] | Exception]) -> None:
        self.data = data
        self.calls: list[tuple[str, date, date]] = []

    def get_daily_bars(self, ticker: str, start: date, end: date) -> list[dict]:
        self.calls.append((ticker, start, end))
        result = self.data[ticker]
        if isinstance(result, Exception):
            raise result
        return [bar for bar in result if start <= bar["trade_date"] <= end]


@pytest.fixture
def companies(db: Session) -> dict[str, Company]:
    apple = company_repository.create(db, "AAPL", "0000320193", "Apple Inc.", "Nasdaq")
    microsoft = company_repository.create(db, "MSFT", "0000789019", "Microsoft Corp", "Nasdaq")
    db.commit()
    return {"AAPL": apple, "MSFT": microsoft}


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    # No real waiting between companies. Records what was asked for
    recorded: list[float] = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


@pytest.fixture
def provider(
    monkeypatch: pytest.MonkeyPatch, companies: dict[str, Company], sleeps: list[float]
) -> FakeProvider:
    fake = FakeProvider({"AAPL": make_bars(TODAY), "MSFT": make_bars(TODAY)})
    monkeypatch.setattr(price_client, "get_price_provider", lambda: fake)
    return fake


def count_bars(db: Session) -> int:
    return db.execute(select(func.count()).select_from(PriceBar)).scalar_one()


def test_the_first_run_on_an_empty_table_is_the_backfill(
    db: Session, companies: dict[str, Company], provider: FakeProvider, sleeps: list[float]
) -> None:
    run = price_service.ingest_prices(db, now_eastern=AFTER_CLOSE)

    assert run.status == "success"
    assert run.finished_at is not None
    assert run.message == (
        "Companies processed: 2, companies failed: 0, "
        f"bars created: {2 * BAR_COUNT}, bars updated: 0"
    )
    assert count_bars(db) == 2 * BAR_COUNT

    # The provider got the whole window for every company
    expected_start = TODAY - timedelta(days=365 * settings.PRICES_LOOKBACK_YEARS)
    assert provider.calls == [("AAPL", expected_start, TODAY), ("MSFT", expected_start, TODAY)]
    # One pause per company
    assert sleeps == [price_service.PAUSE_BETWEEN_COMPANIES_SECONDS] * 2

    # Spot check the last bar of AAPL (position 19)
    stored = {
        bar.trade_date: bar for bar in price_repository.get_by_company(db, companies["AAPL"].id)
    }
    last = stored[TODAY]
    assert last.open == Decimal("119")
    assert last.high == Decimal("122")
    assert last.low == Decimal("118")
    assert last.close == Decimal("120.25")
    assert last.adj_close == Decimal("119.5")
    assert last.volume == 1_000_019


def test_a_second_run_with_identical_data_changes_nothing(
    db: Session, companies: dict[str, Company], provider: FakeProvider
) -> None:
    price_service.ingest_prices(db, now_eastern=AFTER_CLOSE)

    run = price_service.ingest_prices(db, now_eastern=AFTER_CLOSE)

    assert run.status == "success"
    assert "bars created: 0, bars updated: 0" in run.message
    assert count_bars(db) == 2 * BAR_COUNT


def test_a_dividend_style_correction_updates_every_row(
    db: Session, companies: dict[str, Company], provider: FakeProvider
) -> None:
    price_service.ingest_prices(db, now_eastern=AFTER_CLOSE)
    before = {
        bar.trade_date: bar.adj_close
        for bar in price_repository.get_by_company(db, companies["AAPL"].id)
    }

    # The same dates, every adj_close lowered by 1 percent
    for ticker in ("AAPL", "MSFT"):
        for bar in provider.data[ticker]:
            bar["adj_close"] = (bar["adj_close"] * Decimal("0.99")).quantize(Decimal("0.0001"))
    run = price_service.ingest_prices(db, now_eastern=AFTER_CLOSE)

    assert f"bars created: 0, bars updated: {2 * BAR_COUNT}" in run.message
    assert count_bars(db) == 2 * BAR_COUNT
    after = {
        bar.trade_date: bar.adj_close
        for bar in price_repository.get_by_company(db, companies["AAPL"].id)
    }
    assert after.keys() == before.keys()
    assert all(after[day] != before[day] for day in after)
    assert after[TODAY] == Decimal("118.3050")


def test_a_new_trading_day_creates_exactly_one_row(
    db: Session, companies: dict[str, Company], provider: FakeProvider
) -> None:
    # Yesterday (Tuesday) is the last day at first
    yesterday = TODAY - timedelta(days=1)
    provider.data["AAPL"] = make_bars(yesterday)
    provider.data["MSFT"] = make_bars(yesterday)
    price_service.ingest_prices(db, now_eastern=AFTER_CLOSE)
    assert count_bars(db) == 2 * BAR_COUNT

    # Today's bar appears for AAPL only. The other bars keep their values, so MSFT is unchanged
    new_bar = make_bars(TODAY)[-1]
    provider.data["AAPL"] = provider.data["AAPL"] + [new_bar]
    run = price_service.ingest_prices(db, now_eastern=AFTER_CLOSE)

    assert "bars created: 1, bars updated: 0" in run.message
    assert count_bars(db) == 2 * BAR_COUNT + 1


@pytest.mark.parametrize(
    ("now_eastern", "todays_bar_stored"), [(BEFORE_CLOSE, False), (AFTER_CLOSE, True)]
)
def test_todays_bar_is_stored_only_after_17_00_eastern(
    db: Session,
    companies: dict[str, Company],
    provider: FakeProvider,
    now_eastern: datetime,
    todays_bar_stored: bool,
) -> None:
    price_service.ingest_prices(db, now_eastern=now_eastern)

    dates = {bar.trade_date for bar in price_repository.get_by_company(db, companies["AAPL"].id)}
    assert (TODAY in dates) is todays_bar_stored
    # Yesterday's bar is stored either way
    assert (TODAY - timedelta(days=1)) in dates


def test_one_failing_company_does_not_stop_the_others(
    db: Session, companies: dict[str, Company], provider: FakeProvider
) -> None:
    provider.data["AAPL"] = price_client.PriceProviderError("AAPL: yfinance returned no data")

    run = price_service.ingest_prices(db, now_eastern=AFTER_CLOSE)

    assert run.status == "partial"
    assert "Companies processed: 1, companies failed: 1" in run.message
    assert run.message.endswith("Failed companies: AAPL")
    assert price_repository.get_by_company(db, companies["AAPL"].id) == []
    assert len(price_repository.get_by_company(db, companies["MSFT"].id)) == BAR_COUNT


def start_running_run(db: Session, job_type: str, started_ago: timedelta) -> None:
    run = ingestion_repository.create_run(db, job_type)
    run.started_at = datetime.now(UTC) - started_ago
    db.commit()


def test_a_recent_running_run_blocks_a_new_one(
    db: Session, companies: dict[str, Company], provider: FakeProvider
) -> None:
    start_running_run(db, "prices", timedelta(minutes=10))

    with pytest.raises(ConflictError) as error:
        price_service.ingest_prices(db, now_eastern=AFTER_CLOSE)

    assert error.value.message == "A price ingestion run is already in progress"
    # Nothing was created: only the existing run row, and no bars
    assert db.execute(select(func.count()).select_from(IngestionRun)).scalar_one() == 1
    assert count_bars(db) == 0
    assert provider.calls == []


def test_a_stale_running_run_does_not_block(
    db: Session, companies: dict[str, Company], provider: FakeProvider
) -> None:
    start_running_run(db, "prices", timedelta(hours=3))

    run = price_service.ingest_prices(db, now_eastern=AFTER_CLOSE)

    assert run.status == "success"


@pytest.mark.parametrize("other_job_type", ["ingest_filings", "financial_facts"])
def test_a_running_run_of_another_job_does_not_block(
    db: Session, companies: dict[str, Company], provider: FakeProvider, other_job_type: str
) -> None:
    start_running_run(db, other_job_type, timedelta(minutes=10))

    run = price_service.ingest_prices(db, now_eastern=AFTER_CLOSE)

    assert run.status == "success"


def test_an_unexpected_error_marks_the_run_failed_and_is_raised(
    db: Session, companies: dict[str, Company], provider: FakeProvider
) -> None:
    provider.data["AAPL"] = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        price_service.ingest_prices(db, now_eastern=AFTER_CLOSE)

    run = db.execute(select(IngestionRun)).scalar_one()
    assert run.job_type == "prices"
    assert run.status == "failed"
    assert run.error == "RuntimeError: boom"
    assert run.finished_at is not None


def test_a_second_row_for_the_same_company_and_day_is_rejected(
    db: Session, companies: dict[str, Company]
) -> None:
    bar = make_bars(TODAY, count=1)[0]
    args = [bar[field] for field in ("trade_date", "open", "high", "low", "close", "adj_close")]
    price_repository.create(db, companies["AAPL"].id, *args, bar["volume"])

    with pytest.raises(IntegrityError):
        price_repository.create(db, companies["AAPL"].id, *args, bar["volume"])
    db.rollback()


def test_celery_wiring() -> None:
    # No database needed: only the Celery app is inspected
    from app.workers import tasks  # noqa: F401  (importing registers the task)
    from app.workers.celery_app import celery_app

    assert "ingest_prices" in celery_app.tasks

    schedule_entries = [
        schedule_entry
        for schedule_entry in celery_app.conf.beat_schedule.values()
        if schedule_entry["task"] == "ingest_prices"
    ]
    assert len(schedule_entries) == 1
    # 23:00 UTC, Monday to Friday only
    assert schedule_entries[0]["schedule"].hour == {23}
    assert schedule_entries[0]["schedule"].minute == {0}
    assert schedule_entries[0]["schedule"].day_of_week == {1, 2, 3, 4, 5}
