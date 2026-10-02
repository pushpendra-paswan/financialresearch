# Filings are shared public data, so no function in this file takes an org_id.
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.filings import Filing


def get_accession_numbers(db: Session, company_id: int) -> set[str]:
    statement = select(Filing.accession_number).where(Filing.company_id == company_id)
    return set(db.execute(statement).scalars().all())


def create(
    db: Session,
    company_id: int,
    accession_number: str,
    form_type: str,
    filed_on: date,
    report_date: date | None,
    fiscal_year: int | None,
    primary_document: str,
) -> Filing:
    filing = Filing(
        company_id=company_id,
        accession_number=accession_number,
        form_type=form_type,
        filed_on=filed_on,
        report_date=report_date,
        fiscal_year=fiscal_year,
        primary_document=primary_document,
    )
    db.add(filing)
    db.flush()
    return filing


def list_without_document(db: Session, company_id: int) -> list[Filing]:
    # Filings whose document has not been downloaded yet (raw_path is null)
    statement = (
        select(Filing)
        .where(Filing.company_id == company_id, Filing.raw_path.is_(None))
        .order_by(Filing.filed_on.desc(), Filing.id.desc())
    )
    return list(db.execute(statement).scalars().all())


def list_by_company(
    db: Session, company_id: int, form_type: str | None, limit: int, offset: int
) -> tuple[list[Filing], int]:
    # The filter is built once and used for both the page of rows and the total count
    filters = [Filing.company_id == company_id]
    if form_type:
        filters.append(Filing.form_type == form_type)

    count_statement = select(func.count()).select_from(Filing).where(*filters)
    total = db.execute(count_statement).scalar_one()

    # Newest first. The id breaks ties between filings made on the same day
    page_statement = (
        select(Filing)
        .where(*filters)
        .order_by(Filing.filed_on.desc(), Filing.id.desc())
        .limit(limit)
        .offset(offset)
    )
    filings = list(db.execute(page_statement).scalars().all())
    return filings, total
