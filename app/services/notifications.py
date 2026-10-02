from sqlalchemy.orm import Session

from app.exceptions import NotFoundError
from app.repositories import notifications as notification_repository
from app.schemas.notifications import (
    NotificationListResponse,
    NotificationResponse,
    ReadAllResponse,
)

# Reading notifications writes no audit rows: it is not a change to shared data


def list_notifications(
    db: Session, org_id: int, user_id: int, unread_only: bool, page: int, page_size: int
) -> NotificationListResponse:
    offset = (page - 1) * page_size
    notifications, total = notification_repository.list_by_user(
        db, org_id, user_id, unread_only, page_size, offset
    )

    # All of the user's unread notifications, whatever the filter and the page
    unread_count = notification_repository.count_unread(db, org_id, user_id)

    return NotificationListResponse(
        items=[NotificationResponse.model_validate(item) for item in notifications],
        total=total,
        page=page,
        page_size=page_size,
        unread_count=unread_count,
    )


def mark_read(db: Session, org_id: int, user_id: int, notification_id: int) -> NotificationResponse:
    notification = notification_repository.get_by_id(db, org_id, user_id, notification_id)

    # Same message for a missing notification and for someone else's
    if notification is None:
        raise NotFoundError("Notification not found")

    # Marking an already-read notification is fine (idempotent)
    notification.is_read = True
    db.commit()
    return NotificationResponse.model_validate(notification)


def mark_all_read(db: Session, org_id: int, user_id: int) -> ReadAllResponse:
    updated = notification_repository.mark_all_read(db, org_id, user_id)
    db.commit()
    return ReadAllResponse(updated=updated)
