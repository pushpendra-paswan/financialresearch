from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app.dependencies import get_current_user, get_db
from app.models.users import User
from app.schemas.alerts import AlertCreate, AlertResponse, AlertUpdate
from app.services import alerts as alert_service

# Alerts are personal, and a personal alert changes no shared organization data, so every
# role (admin, analyst, viewer) can manage their own: get_current_user, not require_editor
router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.post(
    "",
    response_model=AlertResponse,
    status_code=201,
    description=(
        "Creates a personal price alert. An alert fires on the day the daily close CROSSES the "
        "threshold (price_above, price_below) or moves by at least the threshold percent versus "
        "the previous close (daily_change_pct). An alert whose condition is already true when it "
        "is created does not fire until the price crosses again. Only bars from today (UTC) on "
        "are considered."
    ),
)
def create_alert(
    data: AlertCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> AlertResponse:
    return alert_service.create_alert(db, current_user.org_id, current_user.id, data)


@router.get(
    "",
    response_model=list[AlertResponse],
    description="The caller's own alerts, newest first. Optional filter: active=true or false.",
)
def list_alerts(
    active: bool | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[AlertResponse]:
    return alert_service.list_alerts(db, current_user.org_id, current_user.id, active)


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert(
    alert_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> AlertResponse:
    return alert_service.get_alert(db, current_user.org_id, current_user.id, alert_id)


@router.patch(
    "/{alert_id}",
    response_model=AlertResponse,
    description=(
        "Changes the threshold, the active flag or both. Changing the threshold or switching "
        "an inactive alert to active restarts watching from today, so old moves never fire it."
    ),
)
def update_alert(
    alert_id: int,
    data: AlertUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AlertResponse:
    return alert_service.update_alert(db, current_user.org_id, current_user.id, alert_id, data)


@router.delete(
    "/{alert_id}",
    status_code=204,
    description="Deletes the alert and all of its notifications.",
)
def delete_alert(
    alert_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> Response:
    alert_service.delete_alert(db, current_user.org_id, current_user.id, alert_id)
    return Response(status_code=204)
