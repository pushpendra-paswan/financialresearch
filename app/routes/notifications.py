from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.dependencies import get_current_user, get_db
from app.models.users import User
from app.schemas.notifications import (
    NotificationListResponse,
    NotificationResponse,
    ReadAllResponse,
)
from app.services import notifications as notification_service

# Notifications belong to the alert's owner; any role can read and mark their own
router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get(
    "",
    response_model=NotificationListResponse,
    description=(
        "The caller's own notifications, newest first. unread_count is the number of ALL the "
        "caller's unread notifications, independent of unread_only and of the page."
    ),
)
def list_notifications(
    unread_only: bool = False,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> NotificationListResponse:
    return notification_service.list_notifications(
        db, current_user.org_id, current_user.id, unread_only, page, page_size
    )


@router.post("/read-all", response_model=ReadAllResponse)
def mark_all_read(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> ReadAllResponse:
    return notification_service.mark_all_read(db, current_user.org_id, current_user.id)


@router.post("/{notification_id}/read", response_model=NotificationResponse)
def mark_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> NotificationResponse:
    return notification_service.mark_read(db, current_user.org_id, current_user.id, notification_id)
