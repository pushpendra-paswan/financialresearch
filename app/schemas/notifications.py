from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class NotificationResponse(BaseModel):
    # Built from a Notification model; ticker is a property of the model.
    # org_id and user_id are deliberately not included
    model_config = ConfigDict(from_attributes=True)

    id: int
    alert_id: int
    ticker: str
    message: str
    trade_date: date
    trigger_value: float
    is_read: bool
    created_at: datetime


class NotificationListResponse(BaseModel):
    items: list[NotificationResponse]
    total: int
    page: int
    page_size: int
    # All of the user's unread notifications, independent of the filter and the page
    unread_count: int


class ReadAllResponse(BaseModel):
    updated: int
