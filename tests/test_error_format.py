from collections.abc import Callable

from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.main import app
from app.services import auth as auth_service

RegisterOrg = Callable[[str, str], dict[str, str]]


class FakeDriverError(Exception):
    # Stands in for psycopg's error: the handler only reads the sqlstate attribute
    def __init__(self, sqlstate: str) -> None:
        super().__init__(f"driver error {sqlstate}: secret table details")
        self.sqlstate = sqlstate


# Throwaway routes that exist only for this test module
@app.get("/test-only/crash")
def raise_runtime_error() -> None:
    raise RuntimeError("secret internal detail")


@app.get("/test-only/unique-violation")
def raise_unique_violation() -> None:
    raise IntegrityError("INSERT ...", {}, FakeDriverError("23505"))


@app.get("/test-only/foreign-key-violation")
def raise_foreign_key_violation() -> None:
    raise IntegrityError("INSERT ...", {}, FakeDriverError("23503"))


def test_unhandled_exception_returns_generic_500(db: Session) -> None:
    # raise_server_exceptions=False, otherwise Starlette re-raises the exception into the test
    crash_client = TestClient(app, raise_server_exceptions=False)

    response = crash_client.get("/test-only/crash")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "secret internal detail" not in response.text


def test_unique_violation_returns_409(client: TestClient) -> None:
    response = client.get("/test-only/unique-violation")

    assert response.status_code == 409
    assert response.json() == {"detail": "This resource already exists"}


def test_other_integrity_error_returns_generic_500(db: Session) -> None:
    crash_client = TestClient(app, raise_server_exceptions=False)

    response = crash_client.get("/test-only/foreign-key-violation")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "23503" not in response.text
    assert "secret table details" not in response.text


def test_invalid_register_body_gives_a_string_detail(client: TestClient) -> None:
    response = client.post(
        "/auth/register",
        json={"organization_name": "Acme", "email": "not-an-email", "password": "short"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, str)
    assert "email:" in detail
    assert "password: String should have at least 8 characters" in detail
    assert "; " in detail
    # The submitted password must never be echoed back
    assert "short" not in detail.replace("String should have at least 8 characters", "")


def test_missing_body_field_gives_a_string_detail(client: TestClient) -> None:
    response = client.post("/auth/register", json={"email": "a@b.com", "password": "long-enough-1"})

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)
    assert "organization_name: Field required" in response.json()["detail"]


def test_missing_body_entirely_uses_just_the_message(client: TestClient) -> None:
    response = client.post("/auth/register")

    assert response.status_code == 422
    assert response.json() == {"detail": "Field required"}


def test_invalid_query_parameter_gives_a_string_detail(
    client: TestClient, register_org: RegisterOrg
) -> None:
    headers = register_org("Acme", "admin@acme.com")

    response = client.get("/companies?page=0", headers=headers)

    assert response.status_code == 422
    assert response.json() == {"detail": "page: Input should be greater than or equal to 1"}


def test_password_is_never_in_a_422_response(client: TestClient) -> None:
    response = client.post(
        "/auth/register",
        json={"organization_name": "Acme", "email": "bad", "password": "my-secret-pw!"},
    )

    assert response.status_code == 422
    assert "my-secret-pw!" not in response.text


def test_unknown_route_keeps_the_default_404(client: TestClient) -> None:
    response = client.get("/nope")

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_wrong_method_keeps_the_default_405(client: TestClient) -> None:
    response = client.post("/health")

    assert response.status_code == 405
    assert response.json() == {"detail": "Method Not Allowed"}


def test_concurrent_duplicate_registration_returns_409(
    client: TestClient, db: Session, monkeypatch
) -> None:
    body = {
        "organization_name": "Acme",
        "email": "race@acme.com",
        "password": "correct-horse-battery",
    }
    assert client.post("/auth/register", json=body).status_code == 201

    # Simulate the race: the service's "email exists" check misses the first user, so the
    # insert reaches the database and hits the unique constraint
    monkeypatch.setattr(auth_service.user_repository, "get_by_email", lambda db, email: None)
    response = client.post("/auth/register", json=body)

    assert response.status_code == 409
    assert response.json() == {"detail": "This resource already exists"}

    # In production get_db closes the failed session. Here the test shares one session, so it
    # rolls back the failed savepoint itself; the first registration is kept.
    db.rollback()
    monkeypatch.undo()
    login = client.post(
        "/auth/login", json={"email": "race@acme.com", "password": "correct-horse-battery"}
    )
    assert login.status_code == 200
