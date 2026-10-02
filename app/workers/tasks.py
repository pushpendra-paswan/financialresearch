import logging

from app.database import SessionLocal
from app.exceptions import ConflictError
from app.services import financials as financial_service
from app.services import ingestion as ingestion_service
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
