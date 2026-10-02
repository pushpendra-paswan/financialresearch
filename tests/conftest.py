import os
from collections.abc import Callable, Generator
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

# --- Test database safety ---
# This block must run before any "app" module is imported, because app.config reads
# DATABASE_URL when it is first imported. Replacing it here means the app, Alembic and the
# tests can never touch the development database.
test_database_url = os.environ.get("TEST_DATABASE_URL")
if not test_database_url:
    pytest.exit("TEST_DATABASE_URL is not set. Add it to .env (see .env.example).", returncode=2)

test_database_name = make_url(test_database_url).database
if not test_database_name or not test_database_name.endswith("_test"):
    pytest.exit(
        f"Refusing to run: TEST_DATABASE_URL points at database '{test_database_name}', "
        "but the name must end with '_test' so the development database is never touched.",
        returncode=2,
    )

os.environ["DATABASE_URL"] = test_database_url

# These imports must come after the environment override above
from alembic.config import Config
from fastapi.testclient import TestClient

from alembic import command
from app.clients import sec
from app.config import settings
from app.database import engine
from app.dependencies import get_db
from app.main import app
from app.services import companies as company_service


@pytest.fixture(scope="session", autouse=True)
def test_database() -> None:
    # Connect to the server's default "postgres" database. CREATE DATABASE cannot run inside
    # a transaction, so this connection uses autocommit.
    server_url = make_url(test_database_url).set(database="postgres")
    server_engine = create_engine(server_url, isolation_level="AUTOCOMMIT")
    with server_engine.connect() as connection:
        exists = connection.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": test_database_name},
        ).scalar()
        # The database is reused between runs, so only create it the first time
        if not exists:
            connection.execute(text(f'CREATE DATABASE "{test_database_name}"'))
    server_engine.dispose()

    # Bring the schema up to date. alembic/env.py reads DATABASE_URL, which now points at the
    # test database. This is a no-op when it is already at head.
    command.upgrade(Config("alembic.ini"), "head")


@pytest.fixture
def db() -> Generator[Session, None, None]:
    # Open one connection and start an outer transaction that is never committed
    connection = engine.connect()
    outer_transaction = connection.begin()

    # "create_savepoint" turns each db.commit() in our services into a savepoint release
    # instead of a real commit, so services can commit as in production
    session = Session(bind=connection, join_transaction_mode="create_savepoint")

    yield session

    # Rolling back the outer transaction discards everything the test wrote,
    # so no data is ever persisted between tests
    session.close()
    outer_transaction.rollback()
    connection.close()


@pytest.fixture
def client(db: Session) -> Generator[TestClient, None, None]:
    # Every request in the test uses the same session as the test itself,
    # so the test sees what the endpoint wrote (and everything is rolled back at the end)
    def override_get_db() -> Generator[Session, None, None]:
        yield db

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def register_org(client: TestClient) -> Callable[[str, str], dict[str, str]]:
    # Registers a new organization through the API and returns the admin's auth headers,
    # so tests can set up several organizations in one line each
    def register(organization_name: str, email: str) -> dict[str, str]:
        response = client.post(
            "/auth/register",
            json={
                "organization_name": organization_name,
                "email": email,
                "password": "correct-horse-battery",
            },
        )
        assert response.status_code == 201
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    return register


# A small file in the same structure as the real SEC company_tickers_exchange.json
SEC_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "company_tickers_exchange.json"


@pytest.fixture
def sec_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    # The parsed fixture, produced by the real client code with the download mocked,
    # so no test ever calls the SEC
    def fake_get(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(
            200, content=SEC_FIXTURE_PATH.read_bytes(), request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(tmp_path))
    return sec.get_company_tickers()


@pytest.fixture
def seeded(db: Session, monkeypatch: pytest.MonkeyPatch, sec_rows: list[dict]) -> None:
    # Seeds the catalog through the service with the SEC client mocked. These are the tickers
    # from the fixture file that tests use.
    monkeypatch.setattr(sec, "get_company_tickers", lambda: sec_rows)
    company_service.seed_companies(db, ["AAPL", "MSFT", "MA", "V", "AMZN", "GOOGL", "JNJ"])


@pytest.fixture
def people(
    client: TestClient, seeded: None, register_org: Callable[[str, str], dict[str, str]]
) -> dict[str, dict]:
    # The users the alert and notification tests need, each as {"headers", "org_id", "user_id"}:
    #   admin, analyst, viewer: three roles of organization "Acme"
    #   colleague: a second analyst of Acme (same organization, a different person)
    #   outsider: the admin of another organization, "Globex"
    password = "correct-horse-battery"
    admin_headers = register_org("Acme", "admin@acme.com")
    outsider_headers = register_org("Globex", "admin@globex.com")
    headers_by_name = {"admin": admin_headers, "outsider": outsider_headers}

    for name, role in [("analyst", "analyst"), ("viewer", "viewer"), ("colleague", "analyst")]:
        email = f"{name}@acme.com"
        created = client.post(
            "/users",
            json={"email": email, "password": password, "role": role},
            headers=admin_headers,
        )
        assert created.status_code == 201
        login = client.post("/auth/login", json={"email": email, "password": password})
        headers_by_name[name] = {"Authorization": f"Bearer {login.json()['access_token']}"}

    result = {}
    for name, headers in headers_by_name.items():
        me = client.get("/auth/me", headers=headers).json()
        result[name] = {"headers": headers, "org_id": me["org_id"], "user_id": me["id"]}
    return result
