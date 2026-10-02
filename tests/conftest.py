import os
from collections.abc import Generator

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
from app.database import engine
from app.dependencies import get_db
from app.main import app


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
