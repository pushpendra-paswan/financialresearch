import logging
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.clients import prices as price_client
from app.config import settings
from app.exceptions import NotFoundError
from app.models.ingestion import IngestionRun
from app.repositories import companies as company_repository
from app.repositories import prices as price_repository
from app.schemas.prices import PriceBarResponse, PriceHistoryResponse
from app.services import ingestion as ingestion_service

logger = logging.getLogger(__name__)

JOB_TYPE = "prices"
# Pause between companies, to be polite to the data source
PAUSE_BETWEEN_COMPANIES_SECONDS = 0.5
# A bar dated today is stored only from this hour (US Eastern) on. The US market closes at 16:00,
# so by 17:00 the day's bar is final
STORE_TODAYS_BAR_FROM_HOUR = 17
# The bar fields that are compared with the stored row, and copied over when they differ
BAR_FIELDS = ("open", "high", "low", "close", "adj_close", "volume")


def ingest_prices(db: Session, now_eastern: datetime | None = None) -> IngestionRun:
    # 1. Guard against an overlapping prices run and record this one
    run = ingestion_service.start_run(db, JOB_TYPE, "price")

    # The one broad except of this job: whatever goes wrong, the run row must not stay
    # "running" forever. It records the failure and re-raises.
    try:
        # 2. now_eastern exists only so tests can control the clock. The window is the last
        # PRICES_LOOKBACK_YEARS years (timedelta, not date.replace, so a leap day cannot fail)
        if now_eastern is None:
            now_eastern = datetime.now(ZoneInfo("America/New_York"))
        today = now_eastern.date()
        start = today - timedelta(days=365 * settings.PRICES_LOOKBACK_YEARS)

        # 3. Which provider to use is a setting. The module is called through its name so tests
        # can replace get_price_provider
        provider = price_client.get_price_provider()

        companies_processed = 0
        failed_tickers: list[str] = []
        bars_created = 0
        bars_updated = 0

        for company in company_repository.list_all(db):
            # Read these now: a rollback below expires the company object
            company_id = company.id
            ticker = company.ticker

            # 4. Fetch the whole window and sync it with the database. Every run re-fetches the
            # full window (the first run on an empty table IS the backfill), because splits,
            # dividend adjustments and late corrections change old values. A failure here skips
            # only this company
            try:
                bars = provider.get_daily_bars(ticker, start, today)

                # A bar dated today is final only after the close. Before that, skip it
                if now_eastern.hour < STORE_TODAYS_BAR_FROM_HOUR:
                    bars = [bar for bar in bars if bar["trade_date"] != today]

                stored = {
                    bar.trade_date: bar for bar in price_repository.get_by_company(db, company_id)
                }
                company_created = 0
                company_updated = 0
                for bar in bars:
                    stored_bar = stored.get(bar["trade_date"])
                    if stored_bar is None:
                        price_repository.create(
                            db,
                            company_id,
                            bar["trade_date"],
                            bar["open"],
                            bar["high"],
                            bar["low"],
                            bar["close"],
                            bar["adj_close"],
                            bar["volume"],
                        )
                        company_created += 1
                    elif any(getattr(stored_bar, field) != bar[field] for field in BAR_FIELDS):
                        for field in BAR_FIELDS:
                            setattr(stored_bar, field, bar[field])
                        company_updated += 1

                # One commit per company, so a later failure never loses finished companies
                db.commit()
                companies_processed += 1
                bars_created += company_created
                bars_updated += company_updated
                logger.info(
                    "%s: %d bars created, %d updated", ticker, company_created, company_updated
                )
            except price_client.PriceProviderError as error:
                db.rollback()
                logger.error("Price ingestion failed for %s: %s", ticker, error)
                failed_tickers.append(ticker)

            # 5. Be polite to the data source
            time.sleep(PAUSE_BETWEEN_COMPANIES_SECONDS)

        # 6. Finish the run. "partial" means at least one company failed
        message = (
            f"Companies processed: {companies_processed}, companies failed: {len(failed_tickers)}, "
            f"bars created: {bars_created}, bars updated: {bars_updated}"
        )
        if failed_tickers:
            message += f". Failed companies: {', '.join(failed_tickers)}"
        ingestion_service.finish_run(db, run, len(failed_tickers), message)
        logger.info("Price ingestion finished (%s): %s", run.status, run.message)
        return run
    except Exception as error:
        # 7. Record the failure and re-raise it
        ingestion_service.fail_run(db, run, error)
        logger.exception("Price ingestion failed")
        raise


def get_prices(db: Session, ticker: str, days: int) -> PriceHistoryResponse:
    company = company_repository.get_by_ticker(db, ticker.upper())
    if company is None:
        raise NotFoundError("Company not found")

    # Oldest first. Prices stored at 4 decimals are exactly representable well within float
    # precision for this use, so float is safe here
    rows = price_repository.list_since(db, company.id, date.today() - timedelta(days=days))
    bars = [
        PriceBarResponse(
            trade_date=row.trade_date,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            adj_close=float(row.adj_close),
            volume=row.volume,
        )
        for row in rows
    ]
    return PriceHistoryResponse(ticker=company.ticker, name=company.name, bars=bars)
