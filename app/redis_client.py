import logging

from pydantic import BaseModel, ValidationError
from redis import Redis
from redis.exceptions import RedisError

from app.config import settings
from app.exceptions import RateLimitError

logger = logging.getLogger(__name__)

# The only module that talks to Redis directly (Celery has its own connection).
# Short timeouts make an outage fail fast instead of hanging requests. The client connects
# lazily, so importing this module needs no running Redis.
redis_client = Redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1,
)


def cache_get(key: str, model: type[BaseModel]) -> BaseModel | None:
    try:
        value = redis_client.get(key)
    except RedisError:
        logger.warning("Cache read failed for %s, treating it as a miss", key, exc_info=True)
        return None
    if value is None:
        return None

    # A corrupted or outdated entry (the schema changed) counts as a miss
    try:
        return model.model_validate_json(value)
    except ValidationError:
        logger.warning("Cache entry %s is invalid, treating it as a miss", key)
        return None


def cache_set(key: str, value: BaseModel) -> None:
    # TTL only, no explicit invalidation. The TTL is read now so tests can change it.
    try:
        redis_client.set(key, value.model_dump_json(), ex=settings.CACHE_TTL_SECONDS)
    except RedisError:
        logger.warning("Cache write failed for %s", key, exc_info=True)


def check_rate_limit(scope: str, identifier: str, limit: int, window_seconds: int) -> None:
    # Fixed window: the first request creates the counter and starts the clock (EXPIRE NX only
    # sets an expiry when there is none), later requests only increase it. One pipeline
    # (MULTI/EXEC) so the three commands are a single round trip.
    key = f"ratelimit:{scope}:{identifier}"
    try:
        pipeline = redis_client.pipeline()
        pipeline.incr(key)
        pipeline.expire(key, window_seconds, nx=True)
        pipeline.ttl(key)
        count, _, ttl = pipeline.execute()
    except RedisError:
        # Fail open: Redis being down must never take the API down
        logger.warning("Rate limit check failed for %s, letting the request through", key)
        return

    # Outside the try block so it is never swallowed as a Redis error
    if count > limit:
        retry_after = max(ttl, 1)
        raise RateLimitError(f"Too many requests. Try again in {retry_after} seconds.", retry_after)
