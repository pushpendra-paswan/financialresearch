from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

RegisterOrg = Callable[[str, str], dict[str, str]]


@pytest.fixture
def headers(seeded: None, register_org: RegisterOrg) -> dict[str, str]:
    return register_org("Acme", "admin@acme.com")


def test_endpoints_require_a_token(client: TestClient, seeded: None) -> None:
    assert client.get("/companies").status_code == 401
    assert client.get("/companies/AAPL").status_code == 401


def test_viewer_can_list_and_open_a_company(client: TestClient, headers: dict[str, str]) -> None:
    client.post(
        "/users",
        json={"email": "viewer@acme.com", "password": "correct-horse-battery", "role": "viewer"},
        headers=headers,
    )
    login = client.post(
        "/auth/login", json={"email": "viewer@acme.com", "password": "correct-horse-battery"}
    )
    viewer_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    listing = client.get("/companies", headers=viewer_headers)
    detail = client.get("/companies/AAPL", headers=viewer_headers)

    assert listing.status_code == 200
    assert listing.json()["total"] == 7
    assert detail.status_code == 200


def test_pagination(client: TestClient, headers: dict[str, str]) -> None:
    page_one = client.get("/companies?page_size=3", headers=headers).json()
    page_two = client.get("/companies?page=2&page_size=3", headers=headers).json()
    beyond_end = client.get("/companies?page=10&page_size=3", headers=headers).json()

    assert len(page_one["items"]) == 3
    assert page_one["total"] == 7
    assert page_one["page"] == 1
    assert page_one["page_size"] == 3
    assert len(page_two["items"]) == 3
    # Default order is by ticker, so the pages must not overlap
    page_one_tickers = [item["ticker"] for item in page_one["items"]]
    page_two_tickers = [item["ticker"] for item in page_two["items"]]
    assert page_one_tickers == ["AAPL", "AMZN", "GOOGL"]
    assert page_two_tickers == ["JNJ", "MA", "MSFT"]
    # A page past the end is empty but still reports the total
    assert beyond_end["items"] == []
    assert beyond_end["total"] == 7


@pytest.mark.parametrize("query", ["page=0", "page_size=0", "page_size=101", "search=" + "a" * 51])
def test_invalid_query_values_return_422(
    client: TestClient, headers: dict[str, str], query: str
) -> None:
    assert client.get(f"/companies?{query}", headers=headers).status_code == 422


def test_search_by_lowercase_ticker(client: TestClient, headers: dict[str, str]) -> None:
    body = client.get("/companies?search=aapl", headers=headers).json()

    assert body["total"] == 1
    assert body["items"][0]["name"] == "Apple Inc."
    assert body["items"][0]["cik"] == "0000320193"


def test_search_by_name_fragment(client: TestClient, headers: dict[str, str]) -> None:
    body = client.get("/companies?search=microsoft", headers=headers).json()

    assert [item["ticker"] for item in body["items"]] == ["MSFT"]


def test_search_with_no_match(client: TestClient, headers: dict[str, str]) -> None:
    body = client.get("/companies?search=zzzz", headers=headers).json()

    assert body["total"] == 0
    assert body["items"] == []


def test_search_treats_percent_literally(client: TestClient, headers: dict[str, str]) -> None:
    # "%" must not act as a wildcard that matches everything
    body = client.get("/companies?search=%25", headers=headers).json()

    assert body["total"] == 0
    assert body["items"] == []


def test_blank_search_means_no_filter(client: TestClient, headers: dict[str, str]) -> None:
    body = client.get("/companies?search=%20%20", headers=headers).json()

    assert body["total"] == 7


def test_search_ranks_exact_ticker_first(client: TestClient, headers: dict[str, str]) -> None:
    body = client.get("/companies?search=MA", headers=headers).json()
    tickers = [item["ticker"] for item in body["items"]]

    # Amazon (AMZN) also matches because "AMAZON" contains "ma", and sorts before MA by ticker,
    # but the exact ticker match must come first
    assert tickers[0] == "MA"
    assert "AMZN" in tickers
    assert body["total"] == len(tickers)


def test_get_company_by_ticker_is_case_insensitive(
    client: TestClient, headers: dict[str, str]
) -> None:
    response = client.get("/companies/aapl", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["cik"] == "0000320193"
    assert body["exchange"] == "Nasdaq"
    # The industry is filled by the filing ingestion, so it is null for a freshly seeded company
    assert body["industry"] is None


def test_unknown_ticker_returns_404(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/companies/NOPE", headers=headers)

    assert response.status_code == 404
    assert response.json() == {"detail": "Company not found"}


def test_companies_are_shared_between_organizations(
    client: TestClient, headers: dict[str, str], register_org: RegisterOrg
) -> None:
    other_headers = register_org("Globex", "admin@globex.com")

    first = client.get("/companies", headers=headers).json()
    second = client.get("/companies", headers=other_headers).json()

    assert first["total"] == second["total"] == 7
    assert first["items"] == second["items"]
