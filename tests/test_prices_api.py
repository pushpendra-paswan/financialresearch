from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.repositories import companies as company_repository
from app.repositories import prices as price_repository

RegisterOrg = Callable[[str, str], dict[str, str]]

# Bars exist for every calendar day from 400 days ago until today, so the windows are easy to
# count. Dates are relative to today so the tests never go stale
SEEDED_DAYS = 400


@pytest.fixture
def headers(seeded: None, register_org: RegisterOrg, db: Session) -> dict[str, str]:
    # AAPL has bars; MSFT has none. The close of a bar is 100 + its age in days, so the oldest
    # bars have the highest close, and the volume is an int
    apple = company_repository.get_by_ticker(db, "AAPL")
    today = date.today()
    for age in range(SEEDED_DAYS + 1):
        price_repository.create(
            db,
            apple.id,
            today - timedelta(days=age),
            Decimal("189.3000"),
            Decimal("190.1235"),
            Decimal("188.0000"),
            Decimal(100 + age),
            Decimal("99.5000"),
            50_000_000 + age,
        )
    db.commit()
    return register_org("Acme", "admin@acme.com")


def test_a_token_is_required(client: TestClient, headers: dict[str, str]) -> None:
    assert client.get("/companies/AAPL/prices").status_code == 401


def test_viewer_can_read(client: TestClient, headers: dict[str, str]) -> None:
    client.post(
        "/users",
        json={"email": "viewer@acme.com", "password": "correct-horse-battery", "role": "viewer"},
        headers=headers,
    )
    login = client.post(
        "/auth/login", json={"email": "viewer@acme.com", "password": "correct-horse-battery"}
    )
    viewer_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    response = client.get("/companies/AAPL/prices", headers=viewer_headers)

    assert response.status_code == 200


def test_days_limits_the_window_and_bars_are_oldest_first(
    client: TestClient, headers: dict[str, str]
) -> None:
    response = client.get("/companies/AAPL/prices?days=30", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["name"] == "Apple Inc."
    dates = [bar["trade_date"] for bar in body["bars"]]
    today = date.today()
    # Today and the 30 days before it
    assert len(dates) == 31
    assert dates == sorted(dates)
    assert dates[0] == (today - timedelta(days=30)).isoformat()
    assert dates[-1] == today.isoformat()


def test_the_default_is_about_one_year(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/companies/AAPL/prices", headers=headers)

    assert response.status_code == 200
    assert len(response.json()["bars"]) == 366


def test_the_maximum_of_1825_days_works(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/companies/AAPL/prices?days=1825", headers=headers)

    assert response.status_code == 200
    # Everything that was seeded
    assert len(response.json()["bars"]) == SEEDED_DAYS + 1


@pytest.mark.parametrize("days", [0, 1826, -5, "abc"])
def test_days_outside_the_range_is_rejected(
    client: TestClient, headers: dict[str, str], days: int | str
) -> None:
    assert client.get(f"/companies/AAPL/prices?days={days}", headers=headers).status_code == 422


def test_the_ticker_is_case_insensitive(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/companies/aapl/prices?days=5", headers=headers)

    assert response.status_code == 200
    assert response.json()["ticker"] == "AAPL"


def test_an_unknown_ticker_returns_404(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/companies/NOPE/prices", headers=headers)

    assert response.status_code == 404
    assert response.json() == {"detail": "Company not found"}


def test_a_company_without_bars_returns_an_empty_list(
    client: TestClient, headers: dict[str, str]
) -> None:
    response = client.get("/companies/MSFT/prices", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"ticker": "MSFT", "name": "MICROSOFT CORP", "bars": []}


def test_another_organization_sees_the_same_data(
    client: TestClient, headers: dict[str, str], register_org: RegisterOrg
) -> None:
    other_headers = register_org("Other Corp", "admin@other.com")

    first = client.get("/companies/AAPL/prices?days=30", headers=headers)
    second = client.get("/companies/AAPL/prices?days=30", headers=other_headers)

    assert second.status_code == 200
    assert second.json() == first.json()


def test_prices_and_volume_are_json_numbers(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/companies/AAPL/prices?days=5", headers=headers)

    bar = response.json()["bars"][-1]
    for field in ("open", "high", "low", "close", "adj_close"):
        assert isinstance(bar[field], float)
    assert isinstance(bar["volume"], int)
    assert bar["open"] == 189.3
    assert bar["high"] == 190.1235
    # The newest bar is today (age 0)
    assert bar["close"] == 100.0
    assert bar["volume"] == 50_000_000
