from celery import Celery

from app.config import settings

# Redis is both the broker (task queue) and the result backend
celery_app = Celery(
    "fin_copilot",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)
