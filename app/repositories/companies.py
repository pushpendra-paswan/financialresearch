# Companies are shared public data, so no function in this file takes an org_id.
from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session

from app.models.companies import Company


def get_by_ticker(db: Session, ticker: str) -> Company | None:
    statement = select(Company).where(Company.ticker == ticker)
    return db.execute(statement).scalar_one_or_none()


def get_by_cik(db: Session, cik: str) -> Company | None:
    statement = select(Company).where(Company.cik == cik)
    return db.execute(statement).scalar_one_or_none()


def create(db: Session, ticker: str, cik: str, name: str, exchange: str | None) -> Company:
    company = Company(ticker=ticker, cik=cik, name=name, exchange=exchange)
    db.add(company)
    db.flush()
    return company


def search_companies(
    db: Session, search: str | None, limit: int, offset: int
) -> tuple[list[Company], int]:
    # The filter is built once and used for both the page of rows and the total count
    filters = []
    if search:
        # icontains is a case-insensitive substring match. autoescape=True makes "%" and "_"
        # typed by a user match themselves instead of acting as wildcards.
        filters.append(
            or_(
                Company.ticker.icontains(search, autoescape=True),
                Company.name.icontains(search, autoescape=True),
            )
        )

    count_statement = select(func.count()).select_from(Company).where(*filters)
    total = db.execute(count_statement).scalar_one()

    # An exact ticker match (ignoring case) comes first, then everything by ticker.
    # Searching "MA" therefore puts Mastercard before names that merely contain "ma".
    exact_ticker_first = case((func.upper(Company.ticker) == (search or "").upper(), 0), else_=1)
    page_statement = (
        select(Company)
        .where(*filters)
        .order_by(exact_ticker_first, Company.ticker)
        .limit(limit)
        .offset(offset)
    )
    companies = list(db.execute(page_statement).scalars().all())
    return companies, total
