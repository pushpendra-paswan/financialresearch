import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.exceptions import ConflictError, ForbiddenError, NotFoundError, UnauthorizedError
from app.routes import auth, companies, filings, health, users, watchlists

# Configure logging once, at startup
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_NAME)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(companies.router)
app.include_router(filings.router)
app.include_router(users.router)
app.include_router(watchlists.router)


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


logger.info("%s started (environment=%s)", settings.APP_NAME, settings.ENVIRONMENT)
