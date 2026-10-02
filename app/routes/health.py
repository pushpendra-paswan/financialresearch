import logging

import redis
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import settings
from app.dependencies import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
def health_ready(db: Session = Depends(get_db)) -> JSONResponse:
    # Check each dependency separately so the response says which one failed
    database_status = "ok"
    redis_status = "ok"

    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        logger.exception("Readiness check: database failed")
        database_status = "error"

    # Short timeouts so a stopped Redis gives a fast 503 instead of hanging
    try:
        redis_client = redis.Redis.from_url(
            settings.REDIS_URL, socket_connect_timeout=2, socket_timeout=2
        )
        redis_client.ping()
    except redis.RedisError:
        logger.exception("Readiness check: redis failed")
        redis_status = "error"

    if database_status == "ok" and redis_status == "ok":
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "database": "ok", "redis": "ok"},
        )
    return JSONResponse(
        status_code=503,
        content={
            "status": "not ready",
            "database": database_status,
            "redis": redis_status,
        },
    )
