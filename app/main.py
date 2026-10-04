import logging
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.dependencies import limit_user
from app.exceptions import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    UnauthorizedError,
)
from app.routes import (
    agent,
    alerts,
    auth,
    chat,
    companies,
    filings,
    financials,
    health,
    notifications,
    prices,
    reports,
    users,
    watchlists,
)

# Configure logging once, at startup
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_NAME)

# Every protected router is limited per user; login and register are limited per IP on their
# own routes (see routes/auth.py). A new router must be included with limit_user.
limited = [Depends(limit_user)]

app.include_router(health.router)
app.include_router(agent.router, dependencies=limited)
app.include_router(alerts.router, dependencies=limited)
app.include_router(auth.router)
app.include_router(chat.router, dependencies=limited)
app.include_router(companies.router, dependencies=limited)
app.include_router(filings.router, dependencies=limited)
app.include_router(financials.router, dependencies=limited)
app.include_router(notifications.router, dependencies=limited)
app.include_router(prices.router, dependencies=limited)
app.include_router(reports.router, dependencies=limited)
app.include_router(users.router, dependencies=limited)
app.include_router(watchlists.router, dependencies=limited)

# The browser UI: plain files from the frontend folder, served under /app. Mounted AFTER the
# routers, and no API route starts with /app. Not rate limited and not cached.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# The one route outside routes/: the site root sends browsers to the UI
@app.get("/", include_in_schema=False)
def redirect_to_app() -> RedirectResponse:
    return RedirectResponse("/app/")


# Services raise custom exceptions; they are turned into HTTP responses here
@app.exception_handler(NotFoundError)
def handle_not_found(request: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": exc.message})


@app.exception_handler(ConflictError)
def handle_conflict(request: Request, exc: ConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": exc.message})


@app.exception_handler(UnauthorizedError)
def handle_unauthorized(request: Request, exc: UnauthorizedError) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"detail": exc.message},
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.exception_handler(ForbiddenError)
def handle_forbidden(request: Request, exc: ForbiddenError) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": exc.message})


@app.exception_handler(RateLimitError)
def handle_rate_limit(request: Request, exc: RateLimitError) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": exc.message},
        headers={"Retry-After": str(exc.retry_after)},
    )


@app.exception_handler(ServiceUnavailableError)
def handle_service_unavailable(request: Request, exc: ServiceUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": exc.message})


# Every error body is {"detail": "<string>"}, so 422 turns FastAPI's list of errors into one
# readable string. Only "msg" is used: "input" and "ctx" can hold the submitted password or
# objects that cannot be serialized.
@app.exception_handler(RequestValidationError)
def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    messages = []
    for error in exc.errors():
        # Drop the first part of the location (body, query or path)
        field = ".".join(str(part) for part in error["loc"][1:])
        messages.append(f"{field}: {error['msg']}" if field else error["msg"])
    return JSONResponse(status_code=422, content={"detail": "; ".join(messages)})


# A unique violation that got past the service checks (two simultaneous requests) is a conflict.
# Any other integrity error (foreign key, check constraint) is a bug.
@app.exception_handler(IntegrityError)
def handle_integrity_error(request: Request, exc: IntegrityError) -> JSONResponse:
    if getattr(exc.orig, "sqlstate", None) == "23505":
        return JSONResponse(status_code=409, content={"detail": "This resource already exists"})
    return handle_unexpected_error(request, exc)


# Catch-all: the exception text is only logged, never sent to the client
@app.exception_handler(Exception)
def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


logger.info("%s started (environment=%s)", settings.APP_NAME, settings.ENVIRONMENT)
