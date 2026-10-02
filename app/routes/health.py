import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.redis_client import redis_client

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

    # The shared client has short timeouts, so a stopped Redis gives a fast 503
    try:
        redis_client.ping()
    except RedisError:
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
