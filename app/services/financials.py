import json
import logging
from datetime import date, timedelta
from decimal import Decimal

import httpx
from sqlalchemy.orm import Session

from app.clients import sec
from app.config import settings
from app.exceptions import NotFoundError
from app.models.financials import FinancialFact
from app.models.ingestion import IngestionRun
from app.repositories import companies as company_repository
from app.repositories import financials as financial_repository
from app.schemas.financials import FinancialPoint, FinancialsResponse, MetricName, MetricSeries
from app.services import ingestion as ingestion_service

logger = logging.getLogger(__name__)

# The single source of truth for ingestion and the API: metric -> (label, unit, us-gaap concepts in
# PRIORITY order). Companies tag the same thing differently (and can switch tags over the years),
# so for one period the first concept in the list that has a value wins. A concept must appear
# under one metric only.
METRICS: dict[MetricName, tuple[str, str, list[str]]] = {
    MetricName.revenue: (
        "Revenue",
        "USD",
        [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
        ],
    ),
    MetricName.net_income: ("Net income", "USD", ["NetIncomeLoss"]),
    MetricName.operating_income: ("Operating income", "USD", ["OperatingIncomeLoss"]),
    MetricName.gross_profit: ("Gross profit", "USD", ["GrossProfit"]),
    MetricName.total_assets: ("Total assets", "USD", ["Assets"]),
    MetricName.total_liabilities: ("Total liabilities", "USD", ["Liabilities"]),
    MetricName.shareholders_equity: ("Shareholders' equity", "USD", ["StockholdersEquity"]),
    MetricName.eps_diluted: ("Diluted EPS", "USD/shares", ["EarningsPerShareDiluted"]),
    MetricName.operating_cash_flow: (
        "Operating cash flow",
        "USD",
        ["NetCashProvidedByUsedInOperatingActivities"],
    ),
}

JOB_TYPE = "financial_facts"
# A duration value counts as annual when the period is this many days long (52 and 53 week years)
MIN_PERIOD_DAYS = 350
MAX_PERIOD_DAYS = 380


def ingest_financial_facts(db: Session) -> IngestionRun:
    # 1-2. Guard against an overlapping facts run (a filings run does not block this) and record
    # this one
    run = ingestion_service.start_run(db, JOB_TYPE, "financial facts")

    # The one broad except of this job: whatever goes wrong, the run row must not stay
    # "running" forever. It records the failure and re-raises.
    try:
        # 3. Only periods that ended on or after this date are stored. timedelta (not
        # date.replace) avoids the error for a leap day
        cutoff = date.today() - timedelta(days=365 * settings.FINANCIALS_LOOKBACK_YEARS)

        companies_processed = 0
        companies_without_facts = 0
        failed_tickers: list[str] = []
        facts_created = 0
        facts_updated = 0

        for company in company_repository.list_all(db):
            # Read these now: a rollback below expires the company object
            company_id = company.id
            ticker = company.ticker
            cik = company.cik

            # 4. Download the company's facts and store the annual values we want. A failure
            # here skips only this company
            try:
                # a. The raw file is saved by the client before parsing
                facts = sec.get_company_facts(cik)

                # b. A company without us-gaap data (a new holding company, for example) is not
                # a failure
                us_gaap = facts.get("facts", {}).get("us-gaap", {})
                if not us_gaap:
                    logger.warning("%s: no us-gaap facts", ticker)
                    companies_without_facts += 1
                    continue

                # c. Keep the wanted entries. The same period appears in several filings (a
                # 10-K repeats last year's numbers, and companies restate), so for each period
                # only the entry with the latest filed date is kept
                best: dict[tuple, dict] = {}
                for _, unit, concepts in METRICS.values():
                    for concept in concepts:
                        for entry in us_gaap.get(concept, {}).get("units", {}).get(unit, []):
                            # 10-Q, 10-K/A and every other form are ignored
                            if entry.get("form") != "10-K":
                                continue
                            if any(entry.get(f) is None for f in ("end", "val", "accn", "filed")):
                                continue
                            period_end = date.fromisoformat(entry["end"])
                            if period_end < cutoff:
                                continue

                            # Balance sheet values have no start date. Income statement and cash
                            # flow values are kept only when the period is about one year
                            # (10-Ks also hold quarterly durations)
                            period_start = None
                            if entry.get("start") is not None:
                                period_start = date.fromisoformat(entry["start"])
                                period_days = (period_end - period_start).days
                                if not MIN_PERIOD_DAYS <= period_days <= MAX_PERIOD_DAYS:
                                    continue

                            filed_on = date.fromisoformat(entry["filed"])
                            key = (concept, unit, period_start, period_end)
                            if key in best and best[key]["filed_on"] >= filed_on:
                                continue
                            best[key] = {
                                "value": entry["val"],
                                "accession_number": entry["accn"],
                                "filed_on": filed_on,
                            }

                # d. Compare with what is stored. New period: create. Newer filing for a stored
                # period (a restatement): update in place. Otherwise: nothing
                stored: dict[tuple, FinancialFact] = {
                    (fact.concept, fact.unit, fact.period_start, fact.period_end): fact
                    for fact in financial_repository.get_by_company(db, company_id)
                }
                company_created = 0
                company_updated = 0
                for key, entry in best.items():
                    concept, unit, period_start, period_end = key
                    # Decimal(str(...)) avoids binary float noise (6.08 stays 6.08)
                    value = Decimal(str(entry["value"]))
                    if key not in stored:
                        financial_repository.create(
                            db,
                            company_id,
                            concept,
                            unit,
                            period_start,
                            period_end,
                            value,
                            period_end.year,
                            "10-K",
                            entry["accession_number"],
                            entry["filed_on"],
                        )
                        company_created += 1
                    elif entry["filed_on"] > stored[key].filed_on:
                        stored_fact = stored[key]
                        stored_fact.value = value
                        stored_fact.accession_number = entry["accession_number"]
                        stored_fact.filed_on = entry["filed_on"]
                        stored_fact.fiscal_year = period_end.year
                        company_updated += 1

                # e. One commit per company, so a later failure never loses finished companies
                db.commit()
                companies_processed += 1
                facts_created += company_created
                facts_updated += company_updated
                logger.info(
                    "%s: %d facts created, %d updated", ticker, company_created, company_updated
                )
            except (httpx.HTTPError, OSError, json.JSONDecodeError) as error:
                db.rollback()
                logger.error("Financial facts ingestion failed for %s: %s", ticker, error)
                failed_tickers.append(ticker)
                continue

        # 5. Finish the run. "partial" means at least one company failed
        message = (
            f"Companies processed: {companies_processed}, companies failed: {len(failed_tickers)}, "
            f"companies without facts: {companies_without_facts}, "
            f"facts created: {facts_created}, facts updated: {facts_updated}"
        )
        if failed_tickers:
            message += f". Failed companies: {', '.join(failed_tickers)}"
        ingestion_service.finish_run(db, run, len(failed_tickers), message)
        logger.info("Financial facts ingestion finished (%s): %s", run.status, run.message)
        return run
    except Exception as error:
        # 6. Record the failure and re-raise it
        ingestion_service.fail_run(db, run, error)
        logger.exception("Financial facts ingestion failed")
        raise


def get_financials(
    db: Session, ticker: str, metric: MetricName | None, years: int
) -> FinancialsResponse:
    company = company_repository.get_by_ticker(db, ticker.upper())
    if company is None:
        raise NotFoundError("Company not found")

    # All nine metrics in enum order, unless one was requested
    metric_names = [metric] if metric else list(MetricName)
    series = []
    for metric_name in metric_names:
        label, unit, concepts = METRICS[metric_name]
        rows = financial_repository.list_by_concepts(db, company.id, concepts, unit)

        # One row per period_end: walk the concepts in priority order, the first concept that
        # has a row for a period wins
        chosen: dict[date, FinancialFact] = {}
        for concept in concepts:
            for row in rows:
                if row.concept == concept and row.period_end not in chosen:
                    chosen[row.period_end] = row

        # The latest `years` periods, returned oldest first
        latest_periods = sorted(chosen, reverse=True)[:years]
        points = [
            FinancialPoint(
                fiscal_year=chosen[period_end].fiscal_year,
                period_start=chosen[period_end].period_start,
                period_end=period_end,
                # float is exact enough here: the largest values (hundreds of billions) are far
                # below the 2**53 limit of exact whole numbers in a float
                value=float(chosen[period_end].value),
                concept=chosen[period_end].concept,
                accession_number=chosen[period_end].accession_number,
                filed_on=chosen[period_end].filed_on,
            )
            for period_end in reversed(latest_periods)
        ]
        series.append(MetricSeries(metric=metric_name, label=label, unit=unit, points=points))

    return FinancialsResponse(ticker=company.ticker, name=company.name, metrics=series)
