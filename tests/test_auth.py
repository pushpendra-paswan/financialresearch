from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.audit import AuditLog
from app.models.organizations import Organization
from app.models.users import User
from app.security import create_access_token

PASSWORD = "correct-horse-battery"


def register_payload(email: str = "admin@acme.com", password: str = PASSWORD) -> dict[str, str]:
    return {"organization_name": "Acme", "email": email, "password": password}


def test_register_creates_organization_and_admin(client: TestClient, db: Session) -> None:
    response = client.post("/auth/register", json=register_payload())

    assert response.status_code == 201
    assert response.json()["token_type"] == "bearer"
    token = response.json()["access_token"]

    # The token works and the caller is an admin
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "admin@acme.com"
    assert me.json()["role"] == "admin"

    # The organization exists and the audit row belongs to it
    organization = db.execute(
        select(Organization).where(Organization.id == me.json()["org_id"])
    ).scalar_one()
    assert organization.name == "Acme"
    audit_actions = (
        db.execute(select(AuditLog.action).where(AuditLog.org_id == organization.id))
        .scalars()
        .all()
    )
    assert audit_actions == ["auth.register"]


@pytest.mark.parametrize("second_email", ["admin@acme.com", "ADMIN@Acme.COM"])
def test_register_duplicate_email_returns_409(client: TestClient, second_email: str) -> None:
    client.post("/auth/register", json=register_payload())

    response = client.post("/auth/register", json=register_payload(email=second_email))

    assert response.status_code == 409


def test_register_stores_email_lowercase(client: TestClient, db: Session) -> None:
    client.post("/auth/register", json=register_payload(email="Admin@Acme.COM"))

    stored_email = db.execute(select(User.email)).scalar_one()
    assert stored_email == "admin@acme.com"


def test_register_short_password_returns_422(client: TestClient) -> None:
    response = client.post("/auth/register", json=register_payload(password="short"))

    assert response.status_code == 422


def test_register_password_over_72_bytes_returns_422(client: TestClient) -> None:
    # 73 ASCII bytes. The character count alone would pass, so this checks bytes
    response = client.post("/auth/register", json=register_payload(password="a" * 73))
    assert response.status_code == 422

    # 40 characters but 80 bytes in UTF-8
    response = client.post("/auth/register", json=register_payload(password="é" * 40))
    assert response.status_code == 422

    # Exactly 72 bytes is allowed
    response = client.post("/auth/register", json=register_payload(password="a" * 72))
    assert response.status_code == 201


def test_login_succeeds_and_writes_audit_row(client: TestClient, db: Session) -> None:
    client.post("/auth/register", json=register_payload())

    # Login is case-insensitive on the email
    response = client.post("/auth/login", json={"email": "Admin@acme.com", "password": PASSWORD})

    assert response.status_code == 200
    token = response.json()["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200

    actions = db.execute(select(AuditLog.action).order_by(AuditLog.id)).scalars().all()
    assert actions == ["auth.register", "auth.login"]


def test_login_wrong_password_and_unknown_email_give_identical_401(client: TestClient) -> None:
    client.post("/auth/register", json=register_payload())

    wrong_password = client.post(
        "/auth/login", json={"email": "admin@acme.com", "password": "not-the-password"}
    )
    unknown_email = client.post(
        "/auth/login", json={"email": "nobody@acme.com", "password": PASSWORD}
    )

    assert wrong_password.status_code == 401
    assert unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json()
    assert wrong_password.json() == {"detail": "Invalid email or password"}


def test_login_password_over_72_bytes_returns_422(client: TestClient) -> None:
    response = client.post("/auth/login", json={"email": "admin@acme.com", "password": "a" * 73})

    assert response.status_code == 422


def build_bad_tokens() -> dict[str, str]:
    # One bad token per way a token can be wrong
    expired_payload = {"sub": "1", "exp": datetime.now(UTC) - timedelta(minutes=1)}
    return {
        "garbage": "this-is-not-a-jwt",
        "wrong_key": jwt.encode(
            {"sub": "1", "exp": datetime.now(UTC) + timedelta(minutes=5)},
            "some-other-secret-key-that-is-long-enough",
            algorithm=settings.JWT_ALGORITHM,
        ),
        "expired": jwt.encode(
            expired_payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        ),
        "unknown_user": create_access_token(999_999_999),
    }


@pytest.mark.parametrize("path", ["/auth/me", "/users"])
def test_protected_routes_reject_missing_token(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("path", ["/auth/me", "/users"])
@pytest.mark.parametrize("case", ["garbage", "wrong_key", "expired", "unknown_user"])
def test_protected_routes_reject_bad_tokens(client: TestClient, path: str, case: str) -> None:
    token = build_bad_tokens()[case]

    response = client.get(path, headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_password_is_hashed_and_never_returned(client: TestClient, db: Session) -> None:
    register = client.post("/auth/register", json=register_payload())
    headers = {"Authorization": f"Bearer {register.json()['access_token']}"}

    stored_hash = db.execute(select(User.hashed_password)).scalar_one()
    assert stored_hash != PASSWORD
    assert stored_hash.startswith("$2")

    # No response that returns users may contain the hash
    responses = [
        register,
        client.get("/auth/me", headers=headers),
        client.get("/users", headers=headers),
    ]
    for response in responses:
        assert "hashed_password" not in response.text
        assert stored_hash not in response.text
