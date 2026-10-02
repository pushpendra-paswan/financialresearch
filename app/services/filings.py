from sqlalchemy.orm import Session

from app.exceptions import NotFoundError
from app.repositories import companies as company_repository
from app.repositories import filings as filing_repository
from app.schemas.filings import FilingListResponse, FilingResponse


def list_filings(
    db: Session, ticker: str, form_type: str | None, page: int, page_size: int
) -> FilingListResponse:
    company = company_repository.get_by_ticker(db, ticker.upper())
    if company is None:
        raise NotFoundError("Company not found")

    offset = (page - 1) * page_size
    filings, total = filing_repository.list_by_company(db, company.id, form_type, page_size, offset)

    # raw_path is not exposed, only whether the document has been downloaded
    items = [
        FilingResponse(
            id=filing.id,
            accession_number=filing.accession_number,
            form_type=filing.form_type,
            filed_on=filing.filed_on,
            report_date=filing.report_date,
            fiscal_year=filing.fiscal_year,
            primary_document=filing.primary_document,
            document_downloaded=filing.raw_path is not None,
        )
        for filing in filings
    ]
    return FilingListResponse(items=items, total=total, page=page, page_size=page_size)
