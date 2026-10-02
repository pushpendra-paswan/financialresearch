from datetime import date, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.alerts import Alert
from app.models.audit import AuditLog
from app.models.notifications import Notification
from app.repositories import alerts as alert_repository
from app.repositories import companies as company_repository
from app.repositories import notifications as notification_repository

People = dict[str, dict]
TODAY = date.today()


# --- Set-up helpers (the tests themselves assert on the responses that matter) ---


def seed_alert(
    db: Session,
    person: dict,
    ticker: str = "AAPL",
    alert_type: str = "price_above",
    threshold: str = "100",
    watch_from: date = TODAY,
    active: bool = True,
) -> Alert:
    # Creates an alert straight through the repository (no API, no limit check)
    company = company_repository.get_by_ticker(db, ticker)
    return alert_repository.create(
        db,
        person["org_id"],
        person["user_id"],
        company.id,
        alert_type,
        Decimal(threshold),
        watch_from,
        active,
    )


def seed_notification(db: Session, alert: Alert, trade_date: date) -> Notification:
    return notification_repository.create(
        db,
        alert.org_id,
        alert.user_id,
        alert.id,
        alert.company_id,
        trade_date,
        Decimal("101"),
        "test notification",
    )


def count_rows(db: Session, model: type, *filters: object) -> int:
    return db.execute(select(func.count()).select_from(model).where(*filters)).scalar_one()


def audit_actions(db: Session, action: str, entity_id: int) -> int:
    return count_rows(db, AuditLog, AuditLog.action == action, AuditLog.entity_id == entity_id)


# --- Authentication ---


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/alerts", {"ticker": "AAPL", "alert_type": "price_above", "threshold": 100}),
        ("GET", "/alerts", None),
        ("GET", "/alerts/1", None),
        ("PATCH", "/alerts/1", {"active": False}),
        ("DELETE", "/alerts/1", None),
    ],
)
def test_endpoints_require_a_token(
    client: TestClient, method: str, path: str, body: dict | None
) -> None:
    response = client.request(method, path, json=body)

    assert response.status_code == 401


# --- Create ---


@pytest.mark.parametrize("role", ["admin", "analyst", "viewer"])
def test_every_role_can_create_an_alert(
    client: TestClient, db: Session, people: People, role: str
) -> None:
    person = people[role]

    response = client.post(
        "/alerts",
        json={"ticker": "AAPL", "alert_type": "price_above", "threshold": "150.25"},
        headers=person["headers"],
    )

    assert response.status_code == 201
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["company_name"]
    assert body["alert_type"] == "price_above"
    assert body["threshold"] == 150.25
    assert body["active"] is True
    assert body["watch_from"] == TODAY.isoformat()
    assert body["created_at"]
    # Neither the organization nor the owner is exposed
    assert "org_id" not in body
    assert "user_id" not in body
    assert audit_actions(db, "alert.create", body["id"]) == 1


def test_a_lowercase_and_padded_ticker_works(client: TestClient, people: People) -> None:
    response = client.post(
        "/alerts",
        json={"ticker": "  aapl ", "alert_type": "price_below", "threshold": 10},
        headers=people["viewer"]["headers"],
    )

    assert response.status_code == 201
    assert response.json()["ticker"] == "AAPL"


def test_an_unknown_ticker_returns_404(client: TestClient, people: People) -> None:
    response = client.post(
        "/alerts",
        json={"ticker": "ZZZZ", "alert_type": "price_above", "threshold": 10},
        headers=people["viewer"]["headers"],
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Company not found"}


@pytest.mark.parametrize(
    "body",
    [
        {"ticker": "AAPL", "alert_type": "price_above", "threshold": 0},
        {"ticker": "AAPL", "alert_type": "price_above", "threshold": -5},
        {"ticker": "AAPL", "alert_type": "price_above", "threshold": "1.23456"},
        {"ticker": "AAPL", "alert_type": "volume_above", "threshold": 10},
        {"ticker": "", "alert_type": "price_above", "threshold": 10},
        {"ticker": "   ", "alert_type": "price_above", "threshold": 10},
        {"ticker": "A" * 16, "alert_type": "price_above", "threshold": 10},
    ],
)
def test_invalid_bodies_return_422(client: TestClient, people: People, body: dict) -> None:
    response = client.post("/alerts", json=body, headers=people["viewer"]["headers"])

    assert response.status_code == 422


def test_the_same_alert_twice_returns_409_but_another_user_can_have_it(
    client: TestClient, people: People
) -> None:
    body = {"ticker": "AAPL", "alert_type": "price_above", "threshold": 100}

    first = client.post("/alerts", json=body, headers=people["viewer"]["headers"])
    second = client.post("/alerts", json=body, headers=people["viewer"]["headers"])
    # Same organization, different user
    colleague = client.post("/alerts", json=body, headers=people["colleague"]["headers"])
    # Other organization
    outsider = client.post("/alerts", json=body, headers=people["outsider"]["headers"])

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json() == {"detail": "You already have this alert"}
    assert colleague.status_code == 201
    assert outsider.status_code == 201


def test_a_different_type_or_threshold_is_not_a_duplicate(
    client: TestClient, people: People
) -> None:
    headers = people["viewer"]["headers"]
    base = {"ticker": "AAPL", "alert_type": "price_above", "threshold": 100}
    client.post("/alerts", json=base, headers=headers)

    other_type = client.post("/alerts", json={**base, "alert_type": "price_below"}, headers=headers)
    other_threshold = client.post("/alerts", json={**base, "threshold": 101}, headers=headers)
    other_company = client.post("/alerts", json={**base, "ticker": "MSFT"}, headers=headers)

    assert other_type.status_code == 201
    assert other_threshold.status_code == 201
    assert other_company.status_code == 201


def test_the_51st_alert_of_a_user_returns_409(
    client: TestClient, db: Session, people: People
) -> None:
    viewer = people["viewer"]
    for number in range(1, 51):
        seed_alert(db, viewer, threshold=str(number))

    response = client.post(
        "/alerts",
        json={"ticker": "AAPL", "alert_type": "price_above", "threshold": 500},
        headers=viewer["headers"],
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Alert limit reached"}
    # The limit is per user: a colleague can still create one
    other = client.post(
        "/alerts",
        json={"ticker": "AAPL", "alert_type": "price_above", "threshold": 500},
        headers=people["colleague"]["headers"],
    )
    assert other.status_code == 201


# --- List ---


def test_list_returns_only_the_callers_alerts_newest_first(
    client: TestClient, db: Session, people: People
) -> None:
    first = seed_alert(db, people["viewer"], "AAPL", threshold="1")
    second = seed_alert(db, people["viewer"], "MSFT", threshold="2")
    third = seed_alert(db, people["viewer"], "AAPL", threshold="3")
    seed_alert(db, people["colleague"], "AAPL", threshold="4")
    seed_alert(db, people["outsider"], "AAPL", threshold="5")

    response = client.get("/alerts", headers=people["viewer"]["headers"])

    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [third.id, second.id, first.id]
    # The admin of the same organization sees none of them (alerts are personal)
    admin_response = client.get("/alerts", headers=people["admin"]["headers"])
    assert admin_response.json() == []


def test_list_active_filter(client: TestClient, db: Session, people: People) -> None:
    active = seed_alert(db, people["viewer"], threshold="1", active=True)
    inactive = seed_alert(db, people["viewer"], threshold="2", active=False)
    headers = people["viewer"]["headers"]

    all_alerts = client.get("/alerts", headers=headers).json()
    only_active = client.get("/alerts?active=true", headers=headers).json()
    only_inactive = client.get("/alerts?active=false", headers=headers).json()

    assert {item["id"] for item in all_alerts} == {active.id, inactive.id}
    assert [item["id"] for item in only_active] == [active.id]
    assert [item["id"] for item in only_inactive] == [inactive.id]


# --- Get ---


def test_get_own_alert(client: TestClient, db: Session, people: People) -> None:
    alert = seed_alert(db, people["viewer"], threshold="12.5")

    response = client.get(f"/alerts/{alert.id}", headers=people["viewer"]["headers"])

    assert response.status_code == 200
    assert response.json()["id"] == alert.id
    assert response.json()["threshold"] == 12.5


def test_other_peoples_and_missing_alerts_return_the_identical_404(
    client: TestClient, db: Session, people: People
) -> None:
    alert = seed_alert(db, people["viewer"])
    headers = people["viewer"]["headers"]
    # The admin is in the SAME organization as the owner; the outsider is in another one
    same_org = client.get(f"/alerts/{alert.id}", headers=people["admin"]["headers"])
    other_org = client.get(f"/alerts/{alert.id}", headers=people["outsider"]["headers"])
    missing = client.get("/alerts/999999", headers=headers)

    assert same_org.status_code == other_org.status_code == missing.status_code == 404
    assert same_org.json() == other_org.json() == missing.json() == {"detail": "Alert not found"}


# --- Patch ---


def test_changing_the_threshold_resets_watch_from(
    client: TestClient, db: Session, people: People
) -> None:
    old_day = TODAY - timedelta(days=10)
    alert = seed_alert(db, people["viewer"], threshold="100", watch_from=old_day)

    response = client.patch(
        f"/alerts/{alert.id}", json={"threshold": "120.5"}, headers=people["viewer"]["headers"]
    )

    assert response.status_code == 200
    assert response.json()["threshold"] == 120.5
    assert response.json()["watch_from"] == TODAY.isoformat()
    assert audit_actions(db, "alert.update", alert.id) == 1


def test_patching_the_threshold_to_the_same_value_changes_nothing(
    client: TestClient, db: Session, people: People
) -> None:
    old_day = TODAY - timedelta(days=10)
    alert = seed_alert(db, people["viewer"], threshold="100", watch_from=old_day)

    response = client.patch(
        f"/alerts/{alert.id}", json={"threshold": 100}, headers=people["viewer"]["headers"]
    )

    assert response.status_code == 200
    assert response.json()["watch_from"] == old_day.isoformat()


def test_activating_an_inactive_alert_resets_watch_from(
    client: TestClient, db: Session, people: People
) -> None:
    old_day = TODAY - timedelta(days=10)
    alert = seed_alert(db, people["viewer"], watch_from=old_day, active=False)

    response = client.patch(
        f"/alerts/{alert.id}", json={"active": True}, headers=people["viewer"]["headers"]
    )

    assert response.status_code == 200
    assert response.json()["active"] is True
    assert response.json()["watch_from"] == TODAY.isoformat()


def test_deactivating_keeps_watch_from(client: TestClient, db: Session, people: People) -> None:
    old_day = TODAY - timedelta(days=10)
    alert = seed_alert(db, people["viewer"], watch_from=old_day, active=True)

    response = client.patch(
        f"/alerts/{alert.id}", json={"active": False}, headers=people["viewer"]["headers"]
    )

    assert response.status_code == 200
    assert response.json()["active"] is False
    assert response.json()["watch_from"] == old_day.isoformat()


def test_patching_active_to_its_current_value_changes_nothing(
    client: TestClient, db: Session, people: People
) -> None:
    old_day = TODAY - timedelta(days=10)
    active_alert = seed_alert(db, people["viewer"], threshold="1", watch_from=old_day, active=True)
    inactive_alert = seed_alert(
        db, people["viewer"], threshold="2", watch_from=old_day, active=False
    )
    headers = people["viewer"]["headers"]

    still_active = client.patch(
        f"/alerts/{active_alert.id}", json={"active": True}, headers=headers
    )
    still_inactive = client.patch(
        f"/alerts/{inactive_alert.id}", json={"active": False}, headers=headers
    )

    assert still_active.json()["active"] is True
    assert still_active.json()["watch_from"] == old_day.isoformat()
    assert still_inactive.json()["active"] is False
    assert still_inactive.json()["watch_from"] == old_day.isoformat()


@pytest.mark.parametrize("body", [{}, {"threshold": 0}, {"threshold": -1}, {"active": "maybe"}])
def test_invalid_patch_bodies_return_422(
    client: TestClient, db: Session, people: People, body: dict
) -> None:
    alert = seed_alert(db, people["viewer"])

    response = client.patch(f"/alerts/{alert.id}", json=body, headers=people["viewer"]["headers"])

    assert response.status_code == 422


def test_patching_another_users_alert_returns_404_and_changes_nothing(
    client: TestClient, db: Session, people: People
) -> None:
    alert = seed_alert(db, people["viewer"], threshold="100")

    same_org = client.patch(
        f"/alerts/{alert.id}", json={"threshold": 1}, headers=people["colleague"]["headers"]
    )
    other_org = client.patch(
        f"/alerts/{alert.id}", json={"threshold": 1}, headers=people["outsider"]["headers"]
    )

    assert same_org.status_code == other_org.status_code == 404
    assert same_org.json() == other_org.json() == {"detail": "Alert not found"}
    db.expire_all()
    assert db.get(Alert, alert.id).threshold == Decimal("100")
    assert audit_actions(db, "alert.update", alert.id) == 0


# --- Delete ---


def test_delete_removes_the_alert_and_its_notifications_only(
    client: TestClient, db: Session, people: People
) -> None:
    doomed = seed_alert(db, people["viewer"], threshold="1")
    kept_own = seed_alert(db, people["viewer"], threshold="2")
    colleague_alert = seed_alert(db, people["colleague"], threshold="3")
    seed_notification(db, doomed, TODAY)
    seed_notification(db, doomed, TODAY - timedelta(days=1))
    seed_notification(db, kept_own, TODAY)
    seed_notification(db, colleague_alert, TODAY)

    response = client.delete(f"/alerts/{doomed.id}", headers=people["viewer"]["headers"])

    assert response.status_code == 204
    assert count_rows(db, Alert, Alert.id == doomed.id) == 0
    assert count_rows(db, Notification, Notification.alert_id == doomed.id) == 0
    assert audit_actions(db, "alert.delete", doomed.id) == 1
    # Everything else is untouched
    assert count_rows(db, Alert, Alert.id.in_([kept_own.id, colleague_alert.id])) == 2
    assert count_rows(db, Notification) == 2


def test_deleting_another_users_alert_returns_404_and_keeps_it(
    client: TestClient, db: Session, people: People
) -> None:
    alert = seed_alert(db, people["viewer"])
    seed_notification(db, alert, TODAY)

    same_org = client.delete(f"/alerts/{alert.id}", headers=people["colleague"]["headers"])
    other_org = client.delete(f"/alerts/{alert.id}", headers=people["outsider"]["headers"])
    missing = client.delete("/alerts/999999", headers=people["viewer"]["headers"])

    assert same_org.status_code == other_org.status_code == missing.status_code == 404
    assert same_org.json() == other_org.json() == missing.json() == {"detail": "Alert not found"}
    assert count_rows(db, Alert, Alert.id == alert.id) == 1
    assert count_rows(db, Notification, Notification.alert_id == alert.id) == 1
    assert audit_actions(db, "alert.delete", alert.id) == 0
