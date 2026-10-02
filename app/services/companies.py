import logging

from sqlalchemy.orm import Session

from app.clients import sec
from app.exceptions import NotFoundError
from app.models.companies import Company
from app.repositories import companies as company_repository
from app.schemas.companies import CompanyListResponse, CompanyResponse

logger = logging.getLogger(__name__)


def list_companies(
    db: Session, search: str | None, page: int, page_size: int
) -> CompanyListResponse:
    # Text that is empty after stripping means "no filter"
    search = (search or "").strip() or None

    offset = (page - 1) * page_size
    companies, total = company_repository.search_companies(db, search, page_size, offset)

    return CompanyListResponse(
        items=[CompanyResponse.model_validate(company) for company in companies],
        total=total,
        page=page,
        page_size=page_size,
    )


def get_company(db: Session, ticker: str) -> Company:
    company = company_repository.get_by_ticker(db, ticker.upper())
    if company is None:
        raise NotFoundError("Company not found")
    return company


def seed_companies(db: Session, tickers: list[str]) -> tuple[int, int]:
    # 1. Download the SEC ticker file and index it by ticker
    sec_rows = sec.get_company_tickers()
    sec_by_ticker = {row["ticker"]: row for row in sec_rows}

    # 2. Validate everything before writing anything. These are command-line failures,
    # so they raise ValueError (not an HTTP error).
    missing = [ticker for ticker in tickers if ticker not in sec_by_ticker]
    if missing:
        raise ValueError(f"Tickers not found in the SEC data: {', '.join(missing)}")

    tickers_by_cik: dict[str, list[str]] = {}
    for ticker in tickers:
        tickers_by_cik.setdefault(sec_by_ticker[ticker]["cik"], []).append(ticker)
    for cik, cik_tickers in tickers_by_cik.items():
        if len(cik_tickers) > 1:
            raise ValueError(
                f"Tickers {', '.join(cik_tickers)} belong to the same company (CIK {cik}). "
                "Seed only one ticker per company."
            )

    # 3. Look up each company by CIK, not by ticker: a CIK never changes, a ticker can
    created = 0
    updated = 0
    for ticker in tickers:
        row = sec_by_ticker[ticker]
        company = company_repository.get_by_cik(db, row["cik"])
        if company:
            company.ticker = row["ticker"]
            company.name = row["name"]
            company.exchange = row["exchange"]
            updated += 1
        else:
            company_repository.create(db, row["ticker"], row["cik"], row["name"], row["exchange"])
            created += 1

    # 4. One commit for the whole run. There is no audit row: audit_logs needs an
    # organization, and this is a system action.
    db.commit()
    logger.info("Seeded companies: %d created, %d updated", created, updated)
    return created, updated
