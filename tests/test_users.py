from collections.abc import Callable

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.audit import AuditLog

PASSWORD = "correct-horse-battery"
RegisterOrg = Callable[[str, str], dict[str, str]]


def create_user(
    client: TestClient, headers: dict[str, str], email: str, role: str = "analyst"
) -> dict:
    # Helper for set-up steps. The test itself asserts on the responses that matter.
    response = client.post(
        "/users", json={"email": email, "password": PASSWORD, "role": role}, headers=headers
    )
    assert response.status_code == 201
    return response.json()


def test_admin_creates_analyst(client: TestClient, db: Session, register_org: RegisterOrg) -> None:
    admin_headers = register_org("Acme", "admin@acme.com")
    admin = client.get("/auth/me", headers=admin_headers).json()

    response = client.post(
        "/users",
        json={"email": "Analyst@Acme.com", "password": PASSWORD, "role": "analyst"},
        headers=admin_headers,
    )

    assert response.status_code == 201
    new_user = response.json()
    assert new_user["org_id"] == admin["org_id"]
    assert new_user["role"] == "analyst"
    assert new_user["email"] == "analyst@acme.com"

    # The audit row names the admin as the actor and the new user as the entity
    audit_row = db.execute(select(AuditLog).where(AuditLog.action == "user.create")).scalar_one()
    assert audit_row.org_id == admin["org_id"]
    assert audit_row.user_id == admin["id"]
    assert audit_row.entity_id == new_user["id"]

    # The new user can log in
    login = client.post("/auth/login", json={"email": "analyst@acme.com", "password": PASSWORD})
    assert login.status_code == 200


def test_analyst_cannot_create_users(client: TestClient, register_org: RegisterOrg) -> None:
    admin_headers = register_org("Acme", "admin@acme.com")
    create_user(client, admin_headers, "analyst@acme.com")
    login = client.post("/auth/login", json={"email": "analyst@acme.com", "password": PASSWORD})
    analyst_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    response = client.post(
        "/users",
        json={"email": "other@acme.com", "password": PASSWORD, "role": "viewer"},
        headers=analyst_headers,
    )

    assert response.status_code == 403


def test_invalid_role_returns_422(client: TestClient, register_org: RegisterOrg) -> None:
    admin_headers = register_org("Acme", "admin@acme.com")

    response = client.post(
        "/users",
        json={"email": "boss@acme.com", "password": PASSWORD, "role": "superuser"},
        headers=admin_headers,
    )

    assert response.status_code == 422


def test_email_in_another_organization_returns_409(
    client: TestClient, register_org: RegisterOrg
) -> None:
    register_org("Acme", "admin@acme.com")
    headers_b = register_org("Globex", "admin@globex.com")

    response = client.post(
        "/users",
        json={"email": "admin@acme.com", "password": PASSWORD, "role": "viewer"},
        headers=headers_b,
    )

    assert response.status_code == 409


def test_list_users_returns_only_own_organization(
    client: TestClient, register_org: RegisterOrg
) -> None:
    headers_a = register_org("Acme", "admin@acme.com")
    headers_b = register_org("Globex", "admin@globex.com")
    create_user(client, headers_a, "analyst@acme.com")
    create_user(client, headers_b, "analyst@globex.com")

    list_a = client.get("/users", headers=headers_a)
    list_b = client.get("/users", headers=headers_b)

    assert list_a.status_code == 200
    assert sorted(user["email"] for user in list_a.json()) == [
        "admin@acme.com",
        "analyst@acme.com",
    ]
    assert sorted(user["email"] for user in list_b.json()) == [
        "admin@globex.com",
        "analyst@globex.com",
    ]
    # Every returned user belongs to the same organization as the caller
    assert len({user["org_id"] for user in list_a.json()}) == 1


def test_get_user_in_another_organization_returns_404(
    client: TestClient, register_org: RegisterOrg
) -> None:
    headers_a = register_org("Acme", "admin@acme.com")
    headers_b = register_org("Globex", "admin@globex.com")
    analyst_a = create_user(client, headers_a, "analyst@acme.com")
    analyst_b = create_user(client, headers_b, "analyst@globex.com")

    other_org = client.get(f"/users/{analyst_a['id']}", headers=headers_b)
    missing = client.get("/users/999999999", headers=headers_b)
    own_org = client.get(f"/users/{analyst_b['id']}", headers=headers_b)

    # Another organization's user looks exactly like a user that does not exist
    assert other_org.status_code == 404
    assert missing.status_code == 404
    assert other_org.json() == missing.json()
    assert other_org.json() == {"detail": "User not found"}

    assert own_org.status_code == 200
    assert own_org.json()["email"] == "analyst@globex.com"
