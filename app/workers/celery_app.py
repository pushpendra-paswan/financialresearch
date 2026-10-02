from celery import Celery
from celery.schedules import crontab

from app.config import settings

# Redis is both the broker (task queue) and the result backend.
# include makes the worker import the tasks module, so the tasks are registered
celery_app = Celery(
    "fin_copilot",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"],
)

celery_app.conf.timezone = "UTC"

# Celery beat sends these tasks on schedule. 02:00 UTC is after the US market has closed
celery_app.conf.beat_schedule = {
    "ingest-filings-daily": {
        "task": "ingest_filings",
        "schedule": crontab(hour=2, minute=0),
    },
    # One hour after the filings job, so the two jobs do not share the SEC throttle
    "ingest-financial-facts-daily": {
        "task": "ingest_financial_facts",
        "schedule": crontab(hour=3, minute=0),
    },
}
