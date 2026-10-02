from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.audit import AuditLog
from app.models.companies import Company
from app.models.watchlists import WatchlistItem

PASSWORD = "correct-horse-battery"
RegisterOrg = Callable[[str, str], dict[str, str]]
Headers = dict[str, str]


# --- Set-up helpers (the tests themselves assert on the responses that matter) ---


def login_as(client: TestClient, admin_headers: Headers, email: str, role: str) -> Headers:
    # The admin creates a user with the given role, then that user logs in
    created = client.post(
        "/users", json={"email": email, "password": PASSWORD, "role": role}, headers=admin_headers
    )
    assert created.status_code == 201
    login = client.post("/auth/login", json={"email": email, "password": PASSWORD})
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def make_watchlist(client: TestClient, headers: Headers, name: str) -> dict:
    response = client.post("/watchlists", json={"name": name}, headers=headers)
    assert response.status_code == 201
    return response.json()


def add_company(client: TestClient, headers: Headers, watchlist_id: int, ticker: str) -> dict:
    response = client.post(
        f"/watchlists/{watchlist_id}/companies", json={"ticker": ticker}, headers=headers
    )
    assert response.status_code == 201
    return response.json()


def item_tickers(db: Session, watchlist_id: int) -> list[str]:
    # Reads the items straight from the database, not through the API
    statement = (
        select(Company.ticker)
        .join(WatchlistItem, WatchlistItem.company_id == Company.id)
        .where(WatchlistItem.watchlist_id == watchlist_id)
        .order_by(Company.ticker)
    )
    return list(db.execute(statement).scalars().all())


@pytest.fixture
def admin_headers(seeded: None, register_org: RegisterOrg) -> Headers:
    return register_org("Acme", "admin@acme.com")


@pytest.fixture
def other_admin_headers(admin_headers: Headers, register_org: RegisterOrg) -> Headers:
    # A second organization. It depends on admin_headers so the catalog is seeded first.
    return register_org("Globex", "admin@globex.com")


# --- Authentication and roles ---


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/watchlists", {"name": "Tech"}),
        ("GET", "/watchlists", None),
        ("GET", "/watchlists/1", None),
        ("PATCH", "/watchlists/1", {"name": "Tech"}),
        ("DELETE", "/watchlists/1", None),
        ("POST", "/watchlists/1/companies", {"ticker": "AAPL"}),
        ("DELETE", "/watchlists/1/companies/AAPL", None),
    ],
)
def test_endpoints_require_a_token(
    client: TestClient, method: str, path: str, body: dict | None
) -> None:
    response = client.request(method, path, json=body)

    assert response.status_code == 401


@pytest.mark.parametrize("role", ["admin", "analyst"])
def test_admin_and_analyst_can_create(
    client: TestClient, admin_headers: Headers, role: str
) -> None:
    headers = (
        admin_headers
        if role == "admin"
        else login_as(client, admin_headers, "analyst@acme.com", "analyst")
    )

    response = client.post("/watchlists", json={"name": "Tech"}, headers=headers)

    assert response.status_code == 201


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/watchlists", {"name": "New"}),
        ("PATCH", "/watchlists/{id}", {"name": "Renamed"}),
        ("DELETE", "/watchlists/{id}", None),
        ("POST", "/watchlists/{id}/companies", {"ticker": "MSFT"}),
        ("DELETE", "/watchlists/{id}/companies/AAPL", None),
    ],
)
def test_viewer_cannot_change_anything(
    client: TestClient,
    db: Session,
    admin_headers: Headers,
    method: str,
    path: str,
    body: dict | None,
) -> None:
    watchlist = make_watchlist(client, admin_headers, "Tech")
    add_company(client, admin_headers, watchlist["id"], "AAPL")
    viewer_headers = login_as(client, admin_headers, "viewer@acme.com", "viewer")

    response = client.request(
        method, path.format(id=watchlist["id"]), json=body, headers=viewer_headers
    )

    assert response.status_code == 403
    # Nothing changed
    assert client.get(f"/watchlists/{watchlist['id']}", headers=admin_headers).json()["name"] == (
        "Tech"
    )
    assert item_tickers(db, watchlist["id"]) == ["AAPL"]


def test_viewer_can_list_and_open_watchlists(client: TestClient, admin_headers: Headers) -> None:
    watchlist = make_watchlist(client, admin_headers, "Tech")
    viewer_headers = login_as(client, admin_headers, "viewer@acme.com", "viewer")

    listing = client.get("/watchlists", headers=viewer_headers)
    detail = client.get(f"/watchlists/{watchlist['id']}", headers=viewer_headers)

    assert listing.status_code == 200
    assert [item["name"] for item in listing.json()] == ["Tech"]
    assert detail.status_code == 200


def test_analyst_can_change_a_watchlist_created_by_an_admin(
    client: TestClient, db: Session, admin_headers: Headers
) -> None:
    # Watchlists are shared team resources, not owned by their creator
    watchlist = make_watchlist(client, admin_headers, "Tech")
    analyst_headers = login_as(client, admin_headers, "analyst@acme.com", "analyst")
    path = f"/watchlists/{watchlist['id']}"

    renamed = client.patch(path, json={"name": "Software"}, headers=analyst_headers)
    added = client.post(f"{path}/companies", json={"ticker": "MSFT"}, headers=analyst_headers)
    removed = client.delete(f"{path}/companies/MSFT", headers=analyst_headers)
    deleted = client.delete(path, headers=analyst_headers)

    assert renamed.status_code == 200
    assert added.status_code == 201
    assert removed.status_code == 204
    assert deleted.status_code == 204


# --- Create and names ---


def test_create_returns_the_watchlist_and_writes_an_audit_row(
    client: TestClient, db: Session, admin_headers: Headers
) -> None:
    admin = client.get("/auth/me", headers=admin_headers).json()

    response = client.post("/watchlists", json={"name": "  Tech  "}, headers=admin_headers)

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Tech"
    assert body["created_by"] == admin["id"]
    assert body["item_count"] == 0
    assert set(body) == {"id", "name", "created_by", "created_at", "item_count"}

    audit_row = db.execute(
        select(AuditLog).where(AuditLog.action == "watchlist.create")
    ).scalar_one()
    assert audit_row.org_id == admin["org_id"]
    assert audit_row.user_id == admin["id"]
    assert audit_row.entity_id == body["id"]


@pytest.mark.parametrize("name", ["", "   ", "a" * 101])
def test_create_rejects_invalid_names(
    client: TestClient, admin_headers: Headers, name: str
) -> None:
    response = client.post("/watchlists", json={"name": name}, headers=admin_headers)

    assert response.status_code == 422


def test_create_accepts_a_100_character_name(client: TestClient, admin_headers: Headers) -> None:
    response = client.post("/watchlists", json={"name": "a" * 100}, headers=admin_headers)

    assert response.status_code == 201


@pytest.mark.parametrize("second_name", ["Tech", "tech", "TECH"])
def test_duplicate_name_in_the_same_organization_returns_409(
    client: TestClient, admin_headers: Headers, second_name: str
) -> None:
    make_watchlist(client, admin_headers, "Tech")

    response = client.post("/watchlists", json={"name": second_name}, headers=admin_headers)

    assert response.status_code == 409
    assert response.json() == {"detail": "A watchlist with this name already exists"}


def test_same_name_in_another_organization_is_allowed(
    client: TestClient, admin_headers: Headers, other_admin_headers: Headers
) -> None:
    make_watchlist(client, admin_headers, "Tech")

    response = client.post("/watchlists", json={"name": "Tech"}, headers=other_admin_headers)

    assert response.status_code == 201


# --- List and detail ---


def test_list_shows_only_the_callers_organization_with_counts_ordered_by_name(
    client: TestClient, admin_headers: Headers, other_admin_headers: Headers
) -> None:
    tech = make_watchlist(client, admin_headers, "Tech")
    banks = make_watchlist(client, admin_headers, "banks")
    add_company(client, admin_headers, tech["id"], "AAPL")
    add_company(client, admin_headers, tech["id"], "MSFT")
    add_company(client, admin_headers, banks["id"], "JNJ")
    energy = make_watchlist(client, other_admin_headers, "energy")
    make_watchlist(client, other_admin_headers, "Auto")
    add_company(client, other_admin_headers, energy["id"], "V")

    response = client.get("/watchlists", headers=admin_headers)
    other_response = client.get("/watchlists", headers=other_admin_headers)

    # Ordered by name ignoring case, so "banks" comes before "Tech"
    assert [(item["name"], item["item_count"]) for item in response.json()] == [
        ("banks", 1),
        ("Tech", 2),
    ]
    assert [(item["name"], item["item_count"]) for item in other_response.json()] == [
        ("Auto", 0),
        ("energy", 1),
    ]


def test_detail_returns_companies_sorted_by_ticker(
    client: TestClient, admin_headers: Headers
) -> None:
    watchlist = make_watchlist(client, admin_headers, "Tech")
    for ticker in ["MSFT", "AAPL", "MA"]:
        add_company(client, admin_headers, watchlist["id"], ticker)

    response = client.get(f"/watchlists/{watchlist['id']}", headers=admin_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Tech"
    assert [company["ticker"] for company in body["companies"]] == ["AAPL", "MA", "MSFT"]
    # Companies use the same shape as the catalog
    assert body["companies"][0]["cik"] == "0000320193"


# --- Rename ---


def test_rename_works_and_writes_an_audit_row(
    client: TestClient, db: Session, admin_headers: Headers
) -> None:
    watchlist = make_watchlist(client, admin_headers, "Tech")
    add_company(client, admin_headers, watchlist["id"], "AAPL")

    response = client.patch(
        f"/watchlists/{watchlist['id']}", json={"name": "  Software  "}, headers=admin_headers
    )

    assert response.status_code == 200
    assert response.json()["name"] == "Software"
    assert response.json()["item_count"] == 1
    audit_row = db.execute(
        select(AuditLog).where(AuditLog.action == "watchlist.rename")
    ).scalar_one()
    assert audit_row.entity_id == watchlist["id"]


def test_rename_to_another_watchlists_name_returns_409(
    client: TestClient, admin_headers: Headers
) -> None:
    tech = make_watchlist(client, admin_headers, "Tech")
    make_watchlist(client, admin_headers, "Banks")

    response = client.patch(
        f"/watchlists/{tech['id']}", json={"name": "banks"}, headers=admin_headers
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "A watchlist with this name already exists"}


@pytest.mark.parametrize("new_name", ["Tech", "tech", "TECH"])
def test_rename_to_its_own_name_is_allowed(
    client: TestClient, admin_headers: Headers, new_name: str
) -> None:
    # Same name, or only a change of letter case
    watchlist = make_watchlist(client, admin_headers, "Tech")

    response = client.patch(
        f"/watchlists/{watchlist['id']}", json={"name": new_name}, headers=admin_headers
    )

    assert response.status_code == 200
    assert response.json()["name"] == new_name


# --- Delete ---


def test_delete_removes_the_watchlist_and_its_items_but_not_the_companies(
    client: TestClient, db: Session, admin_headers: Headers, other_admin_headers: Headers
) -> None:
    watchlist = make_watchlist(client, admin_headers, "Tech")
    add_company(client, admin_headers, watchlist["id"], "AAPL")
    add_company(client, admin_headers, watchlist["id"], "MSFT")
    other = make_watchlist(client, other_admin_headers, "Tech")
    add_company(client, other_admin_headers, other["id"], "AAPL")
    companies_before = db.execute(select(func.count()).select_from(Company)).scalar_one()

    response = client.delete(f"/watchlists/{watchlist['id']}", headers=admin_headers)

    assert response.status_code == 204
    assert client.get(f"/watchlists/{watchlist['id']}", headers=admin_headers).status_code == 404
    # The items are gone (checked in the table, not through the API)
    remaining_items = db.execute(
        select(func.count())
        .select_from(WatchlistItem)
        .where(WatchlistItem.watchlist_id == watchlist["id"])
    ).scalar_one()
    assert remaining_items == 0
    # The catalog is untouched
    companies_after = db.execute(select(func.count()).select_from(Company)).scalar_one()
    assert companies_after == companies_before
    # The audit row exists
    audit_row = db.execute(
        select(AuditLog).where(AuditLog.action == "watchlist.delete")
    ).scalar_one()
    assert audit_row.entity_id == watchlist["id"]
    # The other organization's watchlist with the same name is untouched
    other_detail = client.get(f"/watchlists/{other['id']}", headers=other_admin_headers)
    assert other_detail.status_code == 200
    assert [company["ticker"] for company in other_detail.json()["companies"]] == ["AAPL"]


# --- Items ---


def test_add_company_by_lowercase_ticker_returns_the_updated_detail(
    client: TestClient, db: Session, admin_headers: Headers
) -> None:
    watchlist = make_watchlist(client, admin_headers, "Tech")

    response = client.post(
        f"/watchlists/{watchlist['id']}/companies", json={"ticker": " aapl "}, headers=admin_headers
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == watchlist["id"]
    assert [company["ticker"] for company in body["companies"]] == ["AAPL"]
    assert body["companies"][0]["name"] == "Apple Inc."
    audit_row = db.execute(
        select(AuditLog).where(AuditLog.action == "watchlist.add_company")
    ).scalar_one()
    assert audit_row.entity_id == watchlist["id"]


def test_adding_the_same_company_twice_returns_409(
    client: TestClient, db: Session, admin_headers: Headers
) -> None:
    watchlist = make_watchlist(client, admin_headers, "Tech")
    add_company(client, admin_headers, watchlist["id"], "AAPL")

    response = client.post(
        f"/watchlists/{watchlist['id']}/companies", json={"ticker": "aapl"}, headers=admin_headers
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Company is already in this watchlist"}
    assert item_tickers(db, watchlist["id"]) == ["AAPL"]


def test_adding_an_unknown_ticker_returns_404(client: TestClient, admin_headers: Headers) -> None:
    watchlist = make_watchlist(client, admin_headers, "Tech")

    response = client.post(
        f"/watchlists/{watchlist['id']}/companies", json={"ticker": "NOPE"}, headers=admin_headers
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Company not found"}


@pytest.mark.parametrize("ticker", ["", "   ", "A" * 16])
def test_adding_an_invalid_ticker_returns_422(
    client: TestClient, admin_headers: Headers, ticker: str
) -> None:
    watchlist = make_watchlist(client, admin_headers, "Tech")

    response = client.post(
        f"/watchlists/{watchlist['id']}/companies", json={"ticker": ticker}, headers=admin_headers
    )

    assert response.status_code == 422


def test_the_same_company_can_be_in_many_watchlists_and_organizations(
    client: TestClient, db: Session, admin_headers: Headers, other_admin_headers: Headers
) -> None:
    tech = make_watchlist(client, admin_headers, "Tech")
    favorites = make_watchlist(client, admin_headers, "Favorites")
    other = make_watchlist(client, other_admin_headers, "Tech")

    add_company(client, admin_headers, tech["id"], "AAPL")
    add_company(client, admin_headers, favorites["id"], "AAPL")
    add_company(client, other_admin_headers, other["id"], "AAPL")

    assert item_tickers(db, tech["id"]) == ["AAPL"]
    assert item_tickers(db, favorites["id"]) == ["AAPL"]
    assert item_tickers(db, other["id"]) == ["AAPL"]


def test_remove_company(client: TestClient, db: Session, admin_headers: Headers) -> None:
    watchlist = make_watchlist(client, admin_headers, "Tech")
    add_company(client, admin_headers, watchlist["id"], "AAPL")
    add_company(client, admin_headers, watchlist["id"], "MSFT")

    response = client.delete(f"/watchlists/{watchlist['id']}/companies/msft", headers=admin_headers)

    assert response.status_code == 204
    assert item_tickers(db, watchlist["id"]) == ["AAPL"]
    audit_row = db.execute(
        select(AuditLog).where(AuditLog.action == "watchlist.remove_company")
    ).scalar_one()
    assert audit_row.entity_id == watchlist["id"]


def test_removing_a_company_that_is_not_in_the_watchlist_returns_404(
    client: TestClient, admin_headers: Headers
) -> None:
    watchlist = make_watchlist(client, admin_headers, "Tech")

    response = client.delete(f"/watchlists/{watchlist['id']}/companies/AAPL", headers=admin_headers)

    assert response.status_code == 404
    assert response.json() == {"detail": "Company is not in this watchlist"}


def test_removing_an_unknown_ticker_returns_404(client: TestClient, admin_headers: Headers) -> None:
    watchlist = make_watchlist(client, admin_headers, "Tech")

    response = client.delete(f"/watchlists/{watchlist['id']}/companies/NOPE", headers=admin_headers)

    assert response.status_code == 404
    assert response.json() == {"detail": "Company not found"}


# --- Cross-organization isolation ---


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/watchlists/{id}", None),
        ("PATCH", "/watchlists/{id}", {"name": "Hijacked"}),
        ("DELETE", "/watchlists/{id}", None),
        ("POST", "/watchlists/{id}/companies", {"ticker": "MSFT"}),
        ("DELETE", "/watchlists/{id}/companies/AAPL", None),
    ],
)
def test_another_organization_gets_the_same_404_as_a_missing_watchlist(
    client: TestClient,
    db: Session,
    admin_headers: Headers,
    other_admin_headers: Headers,
    method: str,
    path: str,
    body: dict | None,
) -> None:
    # Organization A owns a watchlist with one company
    watchlist = make_watchlist(client, admin_headers, "Tech")
    add_company(client, admin_headers, watchlist["id"], "AAPL")

    # Organization B uses A's watchlist id, and then an id that does not exist
    cross_org = client.request(
        method, path.format(id=watchlist["id"]), json=body, headers=other_admin_headers
    )
    missing = client.request(method, path.format(id=999999), json=body, headers=other_admin_headers)

    assert cross_org.status_code == 404
    assert cross_org.status_code == missing.status_code
    assert cross_org.json() == missing.json() == {"detail": "Watchlist not found"}

    # A's watchlist is unchanged, checked directly in the database
    assert item_tickers(db, watchlist["id"]) == ["AAPL"]
    still_there = client.get(f"/watchlists/{watchlist['id']}", headers=admin_headers)
    assert still_there.status_code == 200
    assert still_there.json()["name"] == "Tech"
