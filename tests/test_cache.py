import logging
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.orm import Session

from app.config import settings
from app.redis_client import redis_client
from app.repositories import companies as company_repository
from app.repositories import prices as price_repository

RegisterOrg = Callable[[str, str], dict[str, str]]


@pytest.fixture
def headers(seeded: None, register_org: RegisterOrg, db: Session) -> dict[str, str]:
    # AAPL gets 100 daily bars ending today
    apple = company_repository.get_by_ticker(db, "AAPL")
    for age in range(100):
        price_repository.create(
            db,
            apple.id,
            date.today() - timedelta(days=age),
            Decimal("189.3000"),
            Decimal("190.1235"),
            Decimal("188.0000"),
            Decimal(100 + age),
            Decimal("99.5000"),
            50_000_000 + age,
        )
    db.commit()
    return register_org("Acme", "admin@acme.com")


def test_company_detail_is_cached(client: TestClient, headers: dict[str, str], db: Session) -> None:
    first = client.get("/companies/aapl", headers=headers)

    assert first.status_code == 200
    assert 1 <= redis_client.ttl("company:AAPL") <= settings.CACHE_TTL_SECONDS
    # Identical body: the JSON round trip of dates and numbers loses nothing
    second = client.get("/companies/AAPL", headers=headers)
    assert second.json() == first.json()

    # A change in the database is invisible until the entry is gone (TTL only, no invalidation)
    company_repository.get_by_ticker(db, "AAPL").name = "Changed Name"
    db.commit()
    assert client.get("/companies/AAPL", headers=headers).json()["name"] == first.json()["name"]

    redis_client.delete("company:AAPL")
    assert client.get("/companies/AAPL", headers=headers).json()["name"] == "Changed Name"


def test_company_list_keys(client: TestClient, headers: dict[str, str]) -> None:
    uncached = client.get("/companies?search=aapl", headers=headers)
    client.get("/companies?search=%20AAPL%20", headers=headers)
    client.get("/companies?search=aapl&page=2", headers=headers)
    client.get("/companies?search=aapl&page_size=5", headers=headers)
    client.get("/companies", headers=headers)

    keys = sorted(key for key in redis_client.keys("companies:list:*"))
    # "aapl" and " AAPL " share a key; page, page_size and the empty search are separate
    assert keys == [
        "companies:list::1:20",
        "companies:list:aapl:1:20",
        "companies:list:aapl:1:5",
        "companies:list:aapl:2:20",
    ]
    cached = client.get("/companies?search=aapl", headers=headers)
    assert cached.json() == uncached.json()


def test_prices_are_cached_per_window(client: TestClient, headers: dict[str, str]) -> None:
    uncached = client.get("/companies/aapl/prices?days=30", headers=headers)
    client.get("/companies/AAPL/prices?days=60", headers=headers)

    assert sorted(redis_client.keys("prices:*")) == ["prices:AAPL:30", "prices:AAPL:60"]
    assert 1 <= redis_client.ttl("prices:AAPL:30") <= settings.CACHE_TTL_SECONDS
    cached = client.get("/companies/AAPL/prices?days=30", headers=headers)
    assert cached.json() == uncached.json()
    assert len(uncached.json()["bars"]) == 31
    assert len(client.get("/companies/AAPL/prices?days=60", headers=headers).json()["bars"]) == 61


def test_not_found_is_never_cached(client: TestClient, headers: dict[str, str]) -> None:
    for _ in range(2):
        assert client.get("/companies/NOPE", headers=headers).status_code == 404
        assert client.get("/companies/NOPE/prices", headers=headers).status_code == 404

    assert [key for key in redis_client.keys("*") if not key.startswith("ratelimit:")] == []


def test_cache_is_shared_between_organizations(
    client: TestClient, headers: dict[str, str], register_org: RegisterOrg
) -> None:
    other_headers = register_org("Globex", "admin@globex.com")

    first = client.get("/companies/AAPL", headers=headers)
    second = client.get("/companies/AAPL", headers=other_headers)

    assert second.status_code == 200
    assert second.json() == first.json()


def test_private_data_is_never_cached(client: TestClient, people: dict[str, dict]) -> None:
    headers = people["analyst"]["headers"]
    watchlist = client.post("/watchlists", json={"name": "Tech"}, headers=headers).json()
    alert = client.post(
        "/alerts",
        json={"ticker": "AAPL", "alert_type": "price_above", "threshold": "200"},
        headers=headers,
    ).json()

    assert client.get("/watchlists", headers=headers).status_code == 200
    assert client.get(f"/watchlists/{watchlist['id']}", headers=headers).status_code == 200
    assert client.get("/alerts", headers=headers).status_code == 200
    assert client.get(f"/alerts/{alert['id']}", headers=headers).status_code == 200
    assert client.get("/notifications", headers=headers).status_code == 200
    assert client.get("/users", headers=headers).status_code == 200

    keys = redis_client.keys("*")
    assert keys != []
    assert all(key.startswith("ratelimit:") for key in keys)


def test_redis_outage_is_a_cache_miss(
    client: TestClient,
    headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    expected_list = client.get("/companies", headers=headers).json()
    expected_prices = client.get("/companies/aapl/prices", headers=headers).json()
    redis_client.flushdb()

    def broken(*args: object, **kwargs: object) -> None:
        raise RedisConnectionError("redis is down")

    monkeypatch.setattr(redis_client, "get", broken)
    monkeypatch.setattr(redis_client, "set", broken)
    # alembic's fileConfig can disable existing loggers in the test process (see Known issues)
    logging.getLogger("app.redis_client").disabled = False

    with caplog.at_level(logging.WARNING, logger="app.redis_client"):
        list_response = client.get("/companies", headers=headers)
        prices_response = client.get("/companies/aapl/prices", headers=headers)

    assert list_response.status_code == 200
    assert list_response.json() == expected_list
    assert prices_response.status_code == 200
    assert prices_response.json() == expected_prices
    assert "Cache read failed" in caplog.text
    assert "Cache write failed" in caplog.text


def test_corrupted_entry_is_a_miss_and_is_replaced(
    client: TestClient, headers: dict[str, str]
) -> None:
    expected = client.get("/companies/AAPL", headers=headers).json()
    redis_client.set("company:AAPL", "not json")

    response = client.get("/companies/AAPL", headers=headers)

    assert response.status_code == 200
    assert response.json() == expected
    assert '"ticker":"AAPL"' in redis_client.get("company:AAPL")


def test_ttl_comes_from_the_setting(
    client: TestClient, headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "CACHE_TTL_SECONDS", 5)

    client.get("/companies/AAPL", headers=headers)

    assert 1 <= redis_client.ttl("company:AAPL") <= 5
