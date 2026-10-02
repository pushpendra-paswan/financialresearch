import logging
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from app.config import settings
from app.main import app
from app.redis_client import redis_client

AUTH_KEY = "ratelimit:auth:testclient"
WRONG_LOGIN = {"email": "nobody@acme.com", "password": "wrong-password-1"}


@pytest.fixture
def limited_users(people: dict[str, dict], monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    # The people fixture made requests that were counted: start every counter from zero
    redis_client.flushdb()
    monkeypatch.setattr(settings, "RATE_LIMIT_API_PER_MINUTE", 3)
    return people


def test_auth_limit_blocks_the_fourth_attempt(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 3)

    for _ in range(3):
        assert client.post("/auth/login", json=WRONG_LOGIN).status_code == 401
    response = client.post("/auth/login", json=WRONG_LOGIN)

    assert response.status_code == 429
    assert 1 <= int(response.headers["Retry-After"]) <= 60
    assert response.json() == {
        "detail": f"Too many requests. Try again in {response.headers['Retry-After']} seconds."
    }


def test_login_and_register_share_one_counter(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 3)
    body = {
        "organization_name": "Acme",
        "email": "admin@acme.com",
        "password": "correct-horse-battery",
    }

    assert client.post("/auth/login", json=WRONG_LOGIN).status_code == 401
    assert client.post("/auth/register", json=body).status_code == 201
    assert client.post("/auth/login", json=WRONG_LOGIN).status_code == 401

    assert client.post("/auth/register", json=body).status_code == 429


def test_another_ip_is_not_limited(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 1)
    other_client = TestClient(app, client=("10.0.0.2", 50000))

    assert client.post("/auth/login", json=WRONG_LOGIN).status_code == 401
    assert client.post("/auth/login", json=WRONG_LOGIN).status_code == 429

    assert other_client.post("/auth/login", json=WRONG_LOGIN).status_code == 401
    assert redis_client.exists("ratelimit:auth:10.0.0.2") == 1


def test_window_expiry_is_set_once_and_not_extended(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 10)

    client.post("/auth/login", json=WRONG_LOGIN)
    assert 1 <= redis_client.ttl(AUTH_KEY) <= 60

    # Shorten the window by hand: a second request must not push it back to 60
    redis_client.expire(AUTH_KEY, 30)
    client.post("/auth/login", json=WRONG_LOGIN)

    assert 1 <= redis_client.ttl(AUTH_KEY) <= 30


def test_user_limit_blocks_the_fourth_request(
    client: TestClient, limited_users: dict[str, dict]
) -> None:
    headers = limited_users["admin"]["headers"]

    for _ in range(3):
        assert client.get("/companies", headers=headers).status_code == 200
    response = client.get("/companies", headers=headers)

    assert response.status_code == 429
    assert 1 <= int(response.headers["Retry-After"]) <= 60
    assert response.json()["detail"].startswith("Too many requests. Try again in ")


def test_other_users_are_not_limited(client: TestClient, limited_users: dict[str, dict]) -> None:
    for _ in range(3):
        client.get("/companies", headers=limited_users["admin"]["headers"])
    assert client.get("/companies", headers=limited_users["admin"]["headers"]).status_code == 429

    # Same organization, and another organization
    assert client.get("/companies", headers=limited_users["analyst"]["headers"]).status_code == 200
    assert client.get("/companies", headers=limited_users["outsider"]["headers"]).status_code == 200


def test_routers_share_the_per_user_counter(
    client: TestClient, limited_users: dict[str, dict]
) -> None:
    headers = limited_users["colleague"]["headers"]

    assert client.get("/auth/me", headers=headers).status_code == 200
    assert client.get("/watchlists", headers=headers).status_code == 200
    assert client.get("/notifications", headers=headers).status_code == 200

    assert client.get("/companies", headers=headers).status_code == 429
    assert client.get("/users", headers=headers).status_code == 429
    assert client.get("/alerts", headers=headers).status_code == 429


def test_requests_without_a_valid_token_create_no_user_key(
    client: TestClient, limited_users: dict[str, dict]
) -> None:
    assert client.get("/companies").status_code == 401
    bad_headers = {"Authorization": "Bearer not-a-real-token"}
    assert client.get("/companies", headers=bad_headers).status_code == 401

    assert redis_client.keys("ratelimit:user:*") == []


def test_health_endpoints_are_never_limited(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 1)
    monkeypatch.setattr(settings, "RATE_LIMIT_API_PER_MINUTE", 1)

    for _ in range(10):
        assert client.get("/health").status_code == 200
        assert client.get("/health/ready").status_code == 200
    assert redis_client.keys("ratelimit:*") == []


def test_redis_outage_fails_open(
    client: TestClient,
    register_org: Callable[[str, str], dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    headers = register_org("Acme", "admin@acme.com")

    def broken_pipeline(*args: object, **kwargs: object) -> None:
        raise RedisConnectionError("redis is down")

    monkeypatch.setattr(redis_client, "pipeline", broken_pipeline)
    # alembic's fileConfig can disable existing loggers in the test process (see Known issues)
    logging.getLogger("app.redis_client").disabled = False

    with caplog.at_level(logging.WARNING, logger="app.redis_client"):
        response = client.get("/auth/me", headers=headers)

    assert response.status_code == 200
    assert "Rate limit check failed" in caplog.text
