import logging

from app.config import settings
from app.database import SessionLocal
from app.exceptions import ConflictError
from app.rag import chunking
from app.services import alerts as alert_service
from app.services import financials as financial_service
from app.services import ingestion as ingestion_service
from app.services import prices as price_service
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# No Celery retries: the next daily run catches up
@celery_app.task(name="ingest_filings")
def ingest_filings() -> str | None:
    db = SessionLocal()
    try:
        run = ingestion_service.ingest_filings(db)
    except ConflictError as error:
        # A skipped run is not a failure
        logger.warning("Filing ingestion skipped: %s", error.message)
        return None
    finally:
        db.close()

    logger.info("Filing ingestion run %d finished (%s): %s", run.id, run.status, run.message)

    # Chain the embedding of new filings: it runs after this task, because the worker has one
    # process (--concurrency=1). Only reached when the service returned normally. Filings that
    # are already chunked are skipped without an API call, and if this chain is ever lost the
    # next day's run catches up
    embed_filings.delay()
    return f"{run.status}: {run.message}"


# No Celery retries: the next daily run catches up
@celery_app.task(name="ingest_financial_facts")
def ingest_financial_facts() -> str | None:
    db = SessionLocal()
    try:
        run = financial_service.ingest_financial_facts(db)
    except ConflictError as error:
        # A skipped run is not a failure
        logger.warning("Financial facts ingestion skipped: %s", error.message)
        return None
    finally:
        db.close()

    logger.info(
        "Financial facts ingestion run %d finished (%s): %s", run.id, run.status, run.message
    )
    return f"{run.status}: {run.message}"


# No Celery retries: the next daily run catches up
@celery_app.task(name="ingest_prices")
def ingest_prices() -> str | None:
    db = SessionLocal()
    try:
        run = price_service.ingest_prices(db)
    except ConflictError as error:
        # A skipped run is not a failure, and there are no new prices to evaluate
        logger.warning("Price ingestion skipped: %s", error.message)
        return None
    finally:
        db.close()

    logger.info("Price ingestion run %d finished (%s): %s", run.id, run.status, run.message)

    # Chain the alert evaluation: it runs after this task, because the worker has one process
    # (--concurrency=1). Only reached when the service returned normally (success or partial).
    # If this chain is ever lost, the next day's run catches up
    evaluate_alerts.delay()
    return f"{run.status}: {run.message}"


# No Celery retries: the next evaluation catches up. It has no beat entry: it is chained after
# ingest_prices
@celery_app.task(name="evaluate_alerts")
def evaluate_alerts() -> str | None:
    db = SessionLocal()
    try:
        run = alert_service.evaluate_alerts(db)
    except ConflictError as error:
        # A skipped run is not a failure
        logger.warning("Alert evaluation skipped: %s", error.message)
        return None
    finally:
        db.close()

    logger.info("Alert evaluation run %d finished (%s): %s", run.id, run.status, run.message)
    return f"{run.status}: {run.message}"


# No Celery retries: the next run catches up. It has no beat entry: it is chained after
# ingest_filings
@celery_app.task(name="embed_filings")
def embed_filings() -> str | None:
    # An empty key means embeddings are switched off, which is not a failure: no run is created
    if not settings.OPENAI_API_KEY:
        logger.warning("embeddings disabled: OPENAI_API_KEY not set")
        return None

    db = SessionLocal()
    try:
        run = chunking.embed_filings(db)
    except ConflictError as error:
        # A skipped run is not a failure
        logger.warning("Embedding skipped: %s", error.message)
        return None
    finally:
        db.close()

    logger.info("Embedding run %d finished (%s): %s", run.id, run.status, run.message)
    return f"{run.status}: {run.message}"
