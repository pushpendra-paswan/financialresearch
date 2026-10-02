from collections.abc import Callable
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.repositories import companies as company_repository
from app.repositories import filings as filing_repository

RegisterOrg = Callable[[str, str], dict[str, str]]


@pytest.fixture
def headers(seeded: None, register_org: RegisterOrg, db: Session) -> dict[str, str]:
    # Seeds filings for AAPL (the newest is a 10-Q, and only one document is downloaded)
    # and none for MSFT
    apple = company_repository.get_by_ticker(db, "AAPL")
    filings = [
        ("0000320193-24-000001", "10-K", date(2024, 11, 1), date(2024, 9, 28), True),
        ("0000320193-25-000001", "10-Q", date(2025, 2, 1), date(2024, 12, 28), True),
        ("0000320193-25-000002", "10-Q", date(2025, 5, 2), date(2025, 3, 29), False),
        ("0000320193-25-000003", "10-K", date(2025, 10, 31), date(2025, 9, 27), True),
        ("0000320193-26-000001", "10-Q", date(2026, 1, 30), None, False),
    ]
    for accession_number, form_type, filed_on, report_date, downloaded in filings:
        filing = filing_repository.create(
            db,
            apple.id,
            accession_number,
            form_type,
            filed_on,
            report_date,
            report_date.year if report_date else None,
            "aapl.htm",
        )
        if downloaded:
            filing.raw_path = f"sec/filings/0000320193/{accession_number}/aapl.htm"
    db.commit()
    return register_org("Acme", "admin@acme.com")


def test_a_token_is_required(client: TestClient, headers: dict[str, str]) -> None:
    assert client.get("/companies/AAPL/filings").status_code == 401


def test_viewer_can_list_filings(client: TestClient, headers: dict[str, str]) -> None:
    client.post(
        "/users",
        json={"email": "viewer@acme.com", "password": "correct-horse-battery", "role": "viewer"},
        headers=headers,
    )
    login = client.post(
        "/auth/login", json={"email": "viewer@acme.com", "password": "correct-horse-battery"}
    )
    viewer_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    response = client.get("/companies/AAPL/filings", headers=viewer_headers)

    assert response.status_code == 200
    assert response.json()["total"] == 5


def test_filings_are_newest_first(client: TestClient, headers: dict[str, str]) -> None:
    body = client.get("/companies/AAPL/filings", headers=headers).json()

    assert [item["filed_on"] for item in body["items"]] == [
        "2026-01-30",
        "2025-10-31",
        "2025-05-02",
        "2025-02-01",
        "2024-11-01",
    ]
    assert body["page"] == 1
    assert body["page_size"] == 20
    newest = body["items"][0]
    assert newest["accession_number"] == "0000320193-26-000001"
    assert newest["form_type"] == "10-Q"
    assert newest["report_date"] is None
    assert newest["fiscal_year"] is None
    assert newest["primary_document"] == "aapl.htm"


def test_form_type_filter(client: TestClient, headers: dict[str, str]) -> None:
    ten_k = client.get("/companies/AAPL/filings?form_type=10-K", headers=headers).json()
    ten_q = client.get("/companies/AAPL/filings?form_type=10-Q", headers=headers).json()

    assert ten_k["total"] == 2
    assert {item["form_type"] for item in ten_k["items"]} == {"10-K"}
    assert ten_q["total"] == 3
    assert {item["form_type"] for item in ten_q["items"]} == {"10-Q"}


@pytest.mark.parametrize(
    "query",
    ["form_type=8-K", "form_type=10-K/A", "form_type=", "page=0", "page_size=0", "page_size=101"],
)
def test_invalid_query_values_return_422(
    client: TestClient, headers: dict[str, str], query: str
) -> None:
    assert client.get(f"/companies/AAPL/filings?{query}", headers=headers).status_code == 422


def test_pagination(client: TestClient, headers: dict[str, str]) -> None:
    page_one = client.get("/companies/AAPL/filings?page_size=2", headers=headers).json()
    page_two = client.get("/companies/AAPL/filings?page=2&page_size=2", headers=headers).json()
    beyond_end = client.get("/companies/AAPL/filings?page=4&page_size=2", headers=headers).json()

    assert page_one["total"] == 5
    assert len(page_one["items"]) == 2
    assert page_two["total"] == 5
    assert page_two["page"] == 2
    assert [item["filed_on"] for item in page_two["items"]] == ["2025-05-02", "2025-02-01"]
    assert beyond_end["items"] == []
    assert beyond_end["total"] == 5


def test_ticker_is_case_insensitive(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/companies/aapl/filings", headers=headers)

    assert response.status_code == 200
    assert response.json()["total"] == 5


def test_unknown_ticker_returns_404(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/companies/NOPE/filings", headers=headers)

    assert response.status_code == 404
    assert response.json() == {"detail": "Company not found"}


def test_a_company_without_filings_returns_an_empty_page(
    client: TestClient, headers: dict[str, str]
) -> None:
    body = client.get("/companies/MSFT/filings", headers=headers).json()

    assert body["items"] == []
    assert body["total"] == 0


def test_another_organization_sees_the_same_filings(
    client: TestClient, headers: dict[str, str], register_org: RegisterOrg
) -> None:
    # Filings are shared public data, so every organization reads the same rows
    other_headers = register_org("Other Org", "admin@other.com")

    mine = client.get("/companies/AAPL/filings", headers=headers).json()
    theirs = client.get("/companies/AAPL/filings", headers=other_headers).json()

    assert theirs["total"] == 5
    assert theirs == mine


def test_response_has_document_downloaded_but_not_raw_path(
    client: TestClient, headers: dict[str, str]
) -> None:
    body = client.get("/companies/AAPL/filings", headers=headers).json()

    downloaded_by_accession = {
        item["accession_number"]: item["document_downloaded"] for item in body["items"]
    }
    assert downloaded_by_accession == {
        "0000320193-26-000001": False,
        "0000320193-25-000003": True,
        "0000320193-25-000002": False,
        "0000320193-25-000001": True,
        "0000320193-24-000001": True,
    }
    for item in body["items"]:
        assert "raw_path" not in item
        assert set(item) == {
            "id",
            "accession_number",
            "form_type",
            "filed_on",
            "report_date",
            "fiscal_year",
            "primary_document",
            "document_downloaded",
        }
