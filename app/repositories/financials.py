# Financial facts are shared public data, so no function in this file takes an org_id.
from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.financials import FinancialFact


def get_by_company(db: Session, company_id: int) -> list[FinancialFact]:
    statement = select(FinancialFact).where(FinancialFact.company_id == company_id)
    return list(db.execute(statement).scalars().all())


def create(
    db: Session,
    company_id: int,
    concept: str,
    unit: str,
    period_start: date | None,
    period_end: date,
    value: Decimal,
    fiscal_year: int,
    form_type: str,
    accession_number: str,
    filed_on: date,
) -> FinancialFact:
    fact = FinancialFact(
        company_id=company_id,
        concept=concept,
        unit=unit,
        period_start=period_start,
        period_end=period_end,
        value=value,
        fiscal_year=fiscal_year,
        form_type=form_type,
        accession_number=accession_number,
        filed_on=filed_on,
    )
    db.add(fact)
    db.flush()
    return fact


def list_by_concepts(
    db: Session, company_id: int, concepts: list[str], unit: str
) -> list[FinancialFact]:
    # Newest period first
    statement = (
        select(FinancialFact)
        .where(
            FinancialFact.company_id == company_id,
            FinancialFact.concept.in_(concepts),
            FinancialFact.unit == unit,
        )
        .order_by(FinancialFact.period_end.desc(), FinancialFact.id)
    )
    return list(db.execute(statement).scalars().all())
