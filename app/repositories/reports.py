# Reports are private to the organization: every function that reads them takes org_id and filters
# by it. Companies, citations and the creator's email are read through the report's org_id too.
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.companies import Company
from app.models.reports import Report, ReportCitation, ReportCompany
from app.models.users import User


def create(
    db: Session,
    org_id: int,
    user_id: int,
    agent_run_id: int | None,
    title: str,
    content: str,
    data_sources: list[dict],
) -> Report:
    report = Report(
        org_id=org_id,
        user_id=user_id,
        agent_run_id=agent_run_id,
        title=title,
        content=content,
        data_sources=data_sources,
    )
    db.add(report)
    db.flush()
    return report


def add_companies(db: Session, report_id: int, company_ids: list[int]) -> None:
    # The service created the report in this transaction, so no org_id check is needed here
    for company_id in company_ids:
        db.add(ReportCompany(report_id=report_id, company_id=company_id))
    db.flush()


def create_citations(db: Session, citations: list[ReportCitation]) -> None:
    db.add_all(citations)
    db.flush()


def get_by_id(db: Session, org_id: int, report_id: int) -> tuple[Report, str] | None:
    # The report with its creator's email
    statement = (
        select(Report, User.email)
        .join(User, User.id == Report.user_id)
        .where(Report.org_id == org_id, Report.id == report_id)
    )
    row = db.execute(statement).first()
    return (row[0], row[1]) if row is not None else None


def list_by_org(
    db: Session, org_id: int, limit: int, offset: int
) -> tuple[list[tuple[Report, str]], int]:
    # Newest first; the page of reports with the creator's email, and the total
    total = db.execute(
        select(func.count()).select_from(Report).where(Report.org_id == org_id)
    ).scalar_one()
    statement = (
        select(Report, User.email)
        .join(User, User.id == Report.user_id)
        .where(Report.org_id == org_id)
        .order_by(Report.created_at.desc(), Report.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return [(report, email) for report, email in db.execute(statement).all()], total


def list_tickers(db: Session, org_id: int, report_ids: list[int]) -> dict[int, list[str]]:
    # {report id: tickers sorted} for a whole page in ONE query (the join goes through reports,
    # so the org_id filter applies to the links as well)
    statement = (
        select(ReportCompany.report_id, Company.ticker)
        .join(Report, Report.id == ReportCompany.report_id)
        .join(Company, Company.id == ReportCompany.company_id)
        .where(Report.org_id == org_id, ReportCompany.report_id.in_(report_ids))
        .order_by(Company.ticker)
    )
    tickers: dict[int, list[str]] = {report_id: [] for report_id in report_ids}
    for report_id, ticker in db.execute(statement).all():
        tickers[report_id].append(ticker)
    return tickers


def list_companies(db: Session, org_id: int, report_id: int) -> list[Company]:
    statement = (
        select(Company)
        .join(ReportCompany, ReportCompany.company_id == Company.id)
        .join(Report, Report.id == ReportCompany.report_id)
        .where(Report.org_id == org_id, Report.id == report_id)
        .order_by(Company.ticker)
    )
    return list(db.execute(statement).scalars().all())


def list_citations(db: Session, org_id: int, report_id: int) -> list[ReportCitation]:
    statement = (
        select(ReportCitation)
        .join(Report, Report.id == ReportCitation.report_id)
        .where(Report.org_id == org_id, Report.id == report_id)
        .order_by(ReportCitation.number)
    )
    return list(db.execute(statement).scalars().all())


def delete(db: Session, report: Report) -> None:
    # The service loaded the report with get_by_id(org_id). The database cascade (ON DELETE
    # CASCADE) removes its companies and citations
    db.delete(report)
    db.flush()
