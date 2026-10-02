from collections.abc import Callable
from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.repositories import companies as company_repository
from app.repositories import financials as financial_repository
from app.schemas.financials import MetricName

RegisterOrg = Callable[[str, str], dict[str, str]]


def add_fact(
    db: Session,
    ticker: str,
    concept: str,
    period_end: date,
    value: str,
    unit: str = "USD",
    with_start: bool = True,
) -> None:
    # Seeds one row directly through the repository
    company = company_repository.get_by_ticker(db, ticker)
    period_start = date(period_end.year - 1, period_end.month, 1) if with_start else None
    financial_repository.create(
        db,
        company.id,
        concept,
        unit,
        period_start,
        period_end,
        Decimal(value),
        period_end.year,
        "10-K",
        f"ACC-{period_end.year}",
        date(period_end.year, 12, 20),
    )


@pytest.fixture
def headers(seeded: None, register_org: RegisterOrg, db: Session) -> dict[str, str]:
    # AAPL: 7 years of Revenues (2018 to 2024), where 2018 and 2019 use the old tag. 2024 also
    # has a RevenueFromContract row. MSFT has no facts at all
    for year in range(2018, 2025):
        concept = "SalesRevenueNet" if year < 2020 else "Revenues"
        add_fact(db, "AAPL", concept, date(year, 9, 30), f"{year}000000000")
    add_fact(
        db, "AAPL", "RevenueFromContractWithCustomerExcludingAssessedTax", date(2024, 9, 30), "1"
    )
    add_fact(
        db, "AAPL", "RevenueFromContractWithCustomerExcludingAssessedTax", date(2023, 9, 30), "2"
    )
    add_fact(db, "AAPL", "Assets", date(2024, 9, 30), "364980000000", with_start=False)
    add_fact(db, "AAPL", "EarningsPerShareDiluted", date(2024, 9, 30), "6.08", unit="USD/shares")
    db.commit()
    return register_org("Acme", "admin@acme.com")


def test_a_token_is_required(client: TestClient, headers: dict[str, str]) -> None:
    assert client.get("/companies/AAPL/financials").status_code == 401


def test_viewer_can_read(client: TestClient, headers: dict[str, str]) -> None:
    client.post(
        "/users",
        json={"email": "viewer@acme.com", "password": "correct-horse-battery", "role": "viewer"},
        headers=headers,
    )
    login = client.post(
        "/auth/login", json={"email": "viewer@acme.com", "password": "correct-horse-battery"}
    )
    viewer_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    response = client.get("/companies/AAPL/financials?metric=revenue", headers=viewer_headers)

    assert response.status_code == 200


def test_revenue_returns_the_latest_years_oldest_first(
    client: TestClient, headers: dict[str, str]
) -> None:
    # The company has 7 periods; the default is the latest 5
    response = client.get("/companies/AAPL/financials?metric=revenue&years=5", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["name"] == "Apple Inc."
    assert len(body["metrics"]) == 1
    series = body["metrics"][0]
    assert series["metric"] == "revenue"
    assert series["label"] == "Revenue"
    assert series["unit"] == "USD"
    assert [point["fiscal_year"] for point in series["points"]] == [2020, 2021, 2022, 2023, 2024]
    assert series["points"][-1]["period_end"] == "2024-09-30"
    assert series["points"][-1]["period_start"] == "2023-09-01"
    assert series["points"][-1]["filed_on"] == "2024-12-20"
    assert set(series["points"][0]) == {
        "fiscal_year",
        "period_start",
        "period_end",
        "value",
        "concept",
        "accession_number",
        "filed_on",
    }


def test_years_limits_to_the_latest_periods(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/companies/AAPL/financials?metric=revenue&years=2", headers=headers)

    points = response.json()["metrics"][0]["points"]
    assert [point["fiscal_year"] for point in points] == [2023, 2024]


def test_default_years_is_five(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/companies/AAPL/financials?metric=revenue", headers=headers)

    assert len(response.json()["metrics"][0]["points"]) == 5


def test_the_first_concept_in_the_priority_list_wins(
    client: TestClient, headers: dict[str, str]
) -> None:
    response = client.get("/companies/AAPL/financials?metric=revenue&years=10", headers=headers)

    points = {point["fiscal_year"]: point for point in response.json()["metrics"][0]["points"]}
    # 2024 has Revenues (2024000000000) and a RevenueFromContract row (1): Revenues wins
    assert points[2024]["concept"] == "Revenues"
    assert points[2024]["value"] == 2024000000000
    # 2018 has only SalesRevenueNet, and the concept field says so
    assert points[2018]["concept"] == "SalesRevenueNet"
    assert points[2018]["value"] == 2018000000000
    assert len(points) == 7


def test_without_metric_all_nine_are_returned_in_enum_order(
    client: TestClient, headers: dict[str, str]
) -> None:
    response = client.get("/companies/AAPL/financials", headers=headers)

    metrics = response.json()["metrics"]
    assert [series["metric"] for series in metrics] == [name.value for name in MetricName]
    points_by_metric = {series["metric"]: series["points"] for series in metrics}
    assert len(points_by_metric["revenue"]) == 5
    assert len(points_by_metric["total_assets"]) == 1
    # A balance sheet value has no start date
    assert points_by_metric["total_assets"][0]["period_start"] is None
    assert points_by_metric["eps_diluted"][0]["value"] == 6.08
    # Metrics without data have an empty list
    assert points_by_metric["net_income"] == []
    assert points_by_metric["operating_cash_flow"] == []


def test_a_company_without_facts_returns_empty_series(
    client: TestClient, headers: dict[str, str]
) -> None:
    response = client.get("/companies/MSFT/financials", headers=headers)

    assert response.status_code == 200
    assert all(series["points"] == [] for series in response.json()["metrics"])


@pytest.mark.parametrize("query", ["metric=profit", "years=0", "years=11", "years=abc"])
def test_invalid_query_values_return_422(
    client: TestClient, headers: dict[str, str], query: str
) -> None:
    assert client.get(f"/companies/AAPL/financials?{query}", headers=headers).status_code == 422


def test_the_ticker_is_case_insensitive(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/companies/aapl/financials?metric=revenue", headers=headers)

    assert response.status_code == 200
    assert response.json()["ticker"] == "AAPL"


def test_an_unknown_ticker_returns_404(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/companies/NOPE/financials", headers=headers)

    assert response.status_code == 404
    assert response.json() == {"detail": "Company not found"}


def test_another_organization_sees_the_same_data(
    client: TestClient, headers: dict[str, str], register_org: RegisterOrg
) -> None:
    other_headers = register_org("Globex", "admin@globex.com")

    mine = client.get("/companies/AAPL/financials", headers=headers)
    theirs = client.get("/companies/AAPL/financials", headers=other_headers)

    assert theirs.status_code == 200
    assert theirs.json() == mine.json()


def test_values_are_json_numbers(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/companies/AAPL/financials?metric=total_assets", headers=headers)

    value = response.json()["metrics"][0]["points"][0]["value"]
    assert isinstance(value, int | float)
    assert value == 364980000000
