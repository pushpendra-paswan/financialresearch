from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from app.dependencies import get_current_user, get_db, require_editor
from app.models.users import User
from app.schemas.reports import ReportDetail, ReportListResponse
from app.services import reports as report_service

# Reports are organization data: every role reads them, admin and analyst delete them. There is no
# POST: a report is created only by the research agent, after the user approved its text
router = APIRouter(prefix="/reports", tags=["reports"])


@router.get(
    "",
    response_model=ReportListResponse,
    description="The organization's reports, newest first, with their company tickers.",
)
def list_reports(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ReportListResponse:
    return report_service.list_reports(db, current_user.org_id, page, page_size)


@router.get(
    "/{report_id}",
    response_model=ReportDetail,
    description=(
        "One report with its companies, its stored citations (the passages as they were when "
        "the report was saved) and the data tool calls it used. 404 for a missing report and "
        "another organization's report (identical bodies)."
    ),
)
def get_report(
    report_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> ReportDetail:
    return report_service.get_report(db, current_user.org_id, report_id)


@router.delete("/{report_id}", status_code=204)
def delete_report(
    report_id: int, current_user: User = Depends(require_editor), db: Session = Depends(get_db)
) -> Response:
    report_service.delete_report(db, current_user.org_id, current_user.id, report_id)
    return Response(status_code=204)
