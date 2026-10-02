from datetime import date, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.alerts import Alert
from app.models.notifications import Notification
from app.repositories import alerts as alert_repository
from app.repositories import companies as company_repository
from app.repositories import notifications as notification_repository

People = dict[str, dict]
TODAY = date.today()


# --- Set-up helpers (the tests themselves assert on the responses that matter) ---


def seed_alert(db: Session, person: dict, ticker: str = "AAPL") -> Alert:
    company = company_repository.get_by_ticker(db, ticker)
    return alert_repository.create(
        db, person["org_id"], person["user_id"], company.id, "price_above", Decimal("100"), TODAY
    )


def seed_notifications(
    db: Session, person: dict, count: int, read_count: int = 0, ticker: str = "AAPL"
) -> list[Notification]:
    # `count` notifications for one new alert of the person, oldest day first. The first
    # `read_count` of them are already read. Notifications of one alert need different days
    alert = seed_alert(db, person, ticker)
    notifications = []
    for number in range(count):
        notification = notification_repository.create(
            db,
            alert.org_id,
            alert.user_id,
            alert.id,
            alert.company_id,
            TODAY - timedelta(days=count - number),
            Decimal("101.5"),
            f"{ticker} crossed above 100 ({number})",
        )
        notification.is_read = number < read_count
        notifications.append(notification)
    db.flush()
    return notifications


# --- Authentication and roles ---


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/notifications"),
        ("POST", "/notifications/1/read"),
        ("POST", "/notifications/read-all"),
    ],
)
def test_endpoints_require_a_token(client: TestClient, method: str, path: str) -> None:
    response = client.request(method, path)

    assert response.status_code == 401


def test_a_viewer_can_list_and_mark_their_own_notifications(
    client: TestClient, db: Session, people: People
) -> None:
    notifications = seed_notifications(db, people["viewer"], count=2)
    headers = people["viewer"]["headers"]

    listed = client.get("/notifications", headers=headers)
    marked = client.post(f"/notifications/{notifications[0].id}/read", headers=headers)

    assert listed.status_code == 200
    assert listed.json()["total"] == 2
    assert marked.status_code == 200


# --- List ---


def test_list_is_newest_first(client: TestClient, db: Session, people: People) -> None:
    notifications = seed_notifications(db, people["viewer"], count=3)

    response = client.get("/notifications", headers=people["viewer"]["headers"])

    ids = [item["id"] for item in response.json()["items"]]
    assert ids == [notification.id for notification in reversed(notifications)]


def test_list_shows_only_the_callers_notifications(
    client: TestClient, db: Session, people: People
) -> None:
    mine = seed_notifications(db, people["viewer"], count=2)
    seed_notifications(db, people["colleague"], count=3)
    seed_notifications(db, people["outsider"], count=4)

    response = client.get("/notifications", headers=people["viewer"]["headers"])

    body = response.json()
    assert body["total"] == 2
    assert {item["id"] for item in body["items"]} == {notification.id for notification in mine}
    assert body["unread_count"] == 2
    # The admin of the same organization has none of their own
    admin_body = client.get("/notifications", headers=people["admin"]["headers"]).json()
    assert admin_body["items"] == []
    assert admin_body["total"] == 0
    assert admin_body["unread_count"] == 0


def test_unread_only_and_unread_count(client: TestClient, db: Session, people: People) -> None:
    # 5 notifications, the 2 oldest are already read
    notifications = seed_notifications(db, people["viewer"], count=5, read_count=2)
    seed_notifications(db, people["colleague"], count=3)
    headers = people["viewer"]["headers"]

    everything = client.get("/notifications", headers=headers).json()
    unread = client.get("/notifications?unread_only=true", headers=headers).json()
    # unread_count counts ALL the caller's unread notifications, whatever the page or the filter
    small_page = client.get("/notifications?page=2&page_size=2", headers=headers).json()
    small_unread_page = client.get(
        "/notifications?unread_only=true&page=2&page_size=2", headers=headers
    ).json()

    assert everything["total"] == 5
    assert everything["unread_count"] == 3
    assert unread["total"] == 3
    assert {item["id"] for item in unread["items"]} == {n.id for n in notifications[2:]}
    assert all(item["is_read"] is False for item in unread["items"])
    assert unread["unread_count"] == 3
    assert len(small_page["items"]) == 2
    assert small_page["unread_count"] == 3
    assert len(small_unread_page["items"]) == 1
    assert small_unread_page["total"] == 3
    assert small_unread_page["unread_count"] == 3


def test_pagination(client: TestClient, db: Session, people: People) -> None:
    notifications = seed_notifications(db, people["viewer"], count=5)
    newest_first = [notification.id for notification in reversed(notifications)]
    headers = people["viewer"]["headers"]

    page_1 = client.get("/notifications?page=1&page_size=2", headers=headers).json()
    page_2 = client.get("/notifications?page=2&page_size=2", headers=headers).json()
    page_3 = client.get("/notifications?page=3&page_size=2", headers=headers).json()
    beyond = client.get("/notifications?page=4&page_size=2", headers=headers).json()

    assert [item["id"] for item in page_1["items"]] == newest_first[0:2]
    assert [item["id"] for item in page_2["items"]] == newest_first[2:4]
    assert [item["id"] for item in page_3["items"]] == newest_first[4:5]
    assert beyond["items"] == []
    for page in (page_1, page_2, page_3, beyond):
        assert page["total"] == 5
        assert page["page_size"] == 2
    assert (page_1["page"], page_2["page"], beyond["page"]) == (1, 2, 4)


@pytest.mark.parametrize("query", ["page=0", "page_size=0", "page_size=101"])
def test_invalid_paging_returns_422(client: TestClient, people: People, query: str) -> None:
    response = client.get(f"/notifications?{query}", headers=people["viewer"]["headers"])

    assert response.status_code == 422


def test_response_fields(client: TestClient, db: Session, people: People) -> None:
    seed_notifications(db, people["viewer"], count=1, ticker="MSFT")

    item = client.get("/notifications", headers=people["viewer"]["headers"]).json()["items"][0]

    assert item["ticker"] == "MSFT"
    assert item["trade_date"] == (TODAY - timedelta(days=1)).isoformat()
    # A JSON number, not a string
    assert item["trigger_value"] == 101.5
    assert isinstance(item["trigger_value"], float)
    assert item["message"].startswith("MSFT")
    assert item["is_read"] is False
    assert item["alert_id"]
    assert item["created_at"]
    assert "org_id" not in item
    assert "user_id" not in item


# --- Mark read ---


def test_mark_read_sets_is_read_and_is_idempotent(
    client: TestClient, db: Session, people: People
) -> None:
    notification = seed_notifications(db, people["viewer"], count=1)[0]
    headers = people["viewer"]["headers"]

    first = client.post(f"/notifications/{notification.id}/read", headers=headers)
    second = client.post(f"/notifications/{notification.id}/read", headers=headers)

    assert first.status_code == second.status_code == 200
    assert first.json()["is_read"] is True
    assert second.json()["is_read"] is True
    assert client.get("/notifications", headers=headers).json()["unread_count"] == 0


def test_mark_read_of_other_peoples_or_missing_notifications_returns_the_identical_404(
    client: TestClient, db: Session, people: People
) -> None:
    notification = seed_notifications(db, people["viewer"], count=1)[0]

    # The colleague is in the SAME organization as the owner; the outsider is in another one
    same_org = client.post(
        f"/notifications/{notification.id}/read", headers=people["colleague"]["headers"]
    )
    other_org = client.post(
        f"/notifications/{notification.id}/read", headers=people["outsider"]["headers"]
    )
    missing = client.post("/notifications/999999/read", headers=people["viewer"]["headers"])

    assert same_org.status_code == other_org.status_code == missing.status_code == 404
    expected = {"detail": "Notification not found"}
    assert same_org.json() == other_org.json() == missing.json() == expected
    # The owner's notification is still unread
    db.expire_all()
    stored = db.execute(select(Notification.is_read).where(Notification.id == notification.id))
    assert stored.scalar_one() is False


# --- Read all ---


def test_read_all_marks_only_the_callers_unread_notifications(
    client: TestClient, db: Session, people: People
) -> None:
    # 4 notifications, 1 already read, so 3 change
    seed_notifications(db, people["viewer"], count=4, read_count=1)
    colleague_notifications = seed_notifications(db, people["colleague"], count=2)
    outsider_notifications = seed_notifications(db, people["outsider"], count=2)
    headers = people["viewer"]["headers"]

    response = client.post("/notifications/read-all", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"updated": 3}
    assert client.get("/notifications", headers=headers).json()["unread_count"] == 0
    # Nobody else's notifications changed
    db.expire_all()
    others = [*colleague_notifications, *outsider_notifications]
    assert [notification.is_read for notification in others] == [False] * 4
    assert (
        client.get("/notifications", headers=people["colleague"]["headers"]).json()["unread_count"]
        == 2
    )


def test_read_all_with_nothing_unread_returns_zero(
    client: TestClient, db: Session, people: People
) -> None:
    seed_notifications(db, people["viewer"], count=2, read_count=2)
    headers = people["viewer"]["headers"]

    with_nothing_unread = client.post("/notifications/read-all", headers=headers)
    # A user without any notification at all
    without_any = client.post("/notifications/read-all", headers=people["admin"]["headers"])

    assert with_nothing_unread.json() == {"updated": 0}
    assert without_any.json() == {"updated": 0}
