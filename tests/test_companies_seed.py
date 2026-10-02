import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.clients import sec
from app.models.companies import Company
from app.services import companies as company_service

TICKERS = ["AAPL", "MSFT", "MA", "V", "AMZN", "GOOGL", "JNJ"]


def count_companies(db: Session) -> int:
    return db.execute(select(func.count()).select_from(Company)).scalar_one()


def test_first_run_creates_rows(
    db: Session, monkeypatch: pytest.MonkeyPatch, sec_rows: list[dict]
) -> None:
    monkeypatch.setattr(sec, "get_company_tickers", lambda: sec_rows)

    created, updated = company_service.seed_companies(db, TICKERS)

    assert (created, updated) == (7, 0)
    assert count_companies(db) == 7
    apple = db.execute(select(Company).where(Company.ticker == "AAPL")).scalar_one()
    assert apple.cik == "0000320193"
    assert apple.name == "Apple Inc."
    assert apple.exchange == "Nasdaq"


def test_second_run_creates_nothing(
    db: Session, monkeypatch: pytest.MonkeyPatch, sec_rows: list[dict]
) -> None:
    monkeypatch.setattr(sec, "get_company_tickers", lambda: sec_rows)
    company_service.seed_companies(db, TICKERS)

    created, updated = company_service.seed_companies(db, TICKERS)

    assert (created, updated) == (0, 7)
    assert count_companies(db) == 7


def test_second_run_refreshes_changed_name_and_ticker(
    db: Session, monkeypatch: pytest.MonkeyPatch, sec_rows: list[dict]
) -> None:
    monkeypatch.setattr(sec, "get_company_tickers", lambda: sec_rows)
    company_service.seed_companies(db, TICKERS)

    # The SEC now reports a new name and a new ticker for Visa's CIK
    changed_rows = [dict(row) for row in sec_rows]
    for row in changed_rows:
        if row["cik"] == "0001403161":
            row["name"] = "VISA HOLDINGS"
            row["ticker"] = "VV"
    monkeypatch.setattr(sec, "get_company_tickers", lambda: changed_rows)

    created, updated = company_service.seed_companies(db, ["AAPL", "VV"])

    assert (created, updated) == (0, 2)
    assert count_companies(db) == 7
    visa = db.execute(select(Company).where(Company.cik == "0001403161")).scalar_one()
    assert visa.ticker == "VV"
    assert visa.name == "VISA HOLDINGS"


def test_unknown_ticker_raises_and_writes_nothing(
    db: Session, monkeypatch: pytest.MonkeyPatch, sec_rows: list[dict]
) -> None:
    monkeypatch.setattr(sec, "get_company_tickers", lambda: sec_rows)

    with pytest.raises(ValueError) as error:
        company_service.seed_companies(db, ["AAPL", "NOPE", "MSFT", "ALSONOPE"])

    # All missing tickers are named, and the valid ones were not written
    assert "NOPE" in str(error.value)
    assert "ALSONOPE" in str(error.value)
    assert count_companies(db) == 0


def test_two_tickers_with_same_cik_raise_and_write_nothing(
    db: Session, monkeypatch: pytest.MonkeyPatch, sec_rows: list[dict]
) -> None:
    monkeypatch.setattr(sec, "get_company_tickers", lambda: sec_rows)

    with pytest.raises(ValueError) as error:
        company_service.seed_companies(db, ["AAPL", "GOOGL", "GOOG"])

    assert "GOOGL" in str(error.value)
    assert "GOOG" in str(error.value)
    assert count_companies(db) == 0
