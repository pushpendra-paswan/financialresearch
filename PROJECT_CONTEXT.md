# PROJECT_CONTEXT.md

## Current status
- Current phase: Phase 0 (Foundation)
- Last completed milestone: 0.2 Local infrastructure
- Next milestone: 0.3 Quality setup (pytest with a test database, ruff, GitHub Actions)

## How to run
- Create local settings: `cp .env.example .env`
- Start services: `docker compose up --build` (db, redis, api on port 8000, worker; add `-d` to run in the background)
- Run migrations: `docker compose exec api alembic upgrade head` (roll back one step: `docker compose exec api alembic downgrade -1`)
- Check it: `curl localhost:8000/health` returns `{"status":"ok"}`; `curl localhost:8000/health/ready` checks database and Redis
- Check the worker: `docker compose exec worker celery -A app.workers.celery_app inspect ping`
- psql shell: `docker compose exec db sh -c 'psql -U $POSTGRES_USER -d $POSTGRES_DB'`
- Reset the database (deletes all data): `docker compose down -v`, then start and migrate again
- Code changes under the project folder reload the API automatically. The worker does not reload: use `docker compose restart worker`
- Run tests: not set up yet (milestone 0.3)
- Seed / backfill commands: None yet

## Environment variables
| Name | Purpose |
|------|---------|
| APP_NAME | Application title (FastAPI title). Default "Financial Research Copilot" |
| ENVIRONMENT | Environment name, e.g. development. Default "development" |
| LOG_LEVEL | Python logging level for our own loggers (DEBUG, INFO, WARNING, ...). Default "INFO" |
| POSTGRES_USER | Postgres user. Read only by docker compose (db container); ignored by the app |
| POSTGRES_PASSWORD | Postgres password. Read only by docker compose |
| POSTGRES_DB | Postgres database name. Read only by docker compose |
| DATABASE_URL | SQLAlchemy URL, psycopg v3 driver: `postgresql+psycopg://user:password@db:5432/dbname`. Required, no default |
| REDIS_URL | Redis URL, used as Celery broker and result backend and by the readiness check: `redis://redis:6379/0`. Required, no default |

## Database tables
| Table | Purpose | Key columns / constraints | Added in |
|-------|---------|---------------------------|----------|
| None yet | Only the `vector` (pgvector) extension is enabled, plus Alembic's own `alembic_version` table | | 0.2 |

## API endpoints
| Method | Path | Purpose | Auth / role |
|--------|------|---------|-------------|
| GET | /health | Returns `{"status": "ok"}` (the app is running) | None |
| GET | /health/ready | Runs `SELECT 1` and a Redis ping. 200 `{"status": "ready", "database": "ok", "redis": "ok"}`, or 503 with `"error"` for whichever failed | None |

## Background jobs
| Task | Schedule | What it does |
|------|----------|--------------|
| None yet | | The Celery app exists and the worker runs, but there are no tasks. Beat comes in 1.4 |

## Key files
- `app/main.py` — FastAPI app, logging setup, router registration, exception handlers (NotFoundError → 404, ConflictError → 409)
- `app/config.py` — `Settings` (pydantic-settings, reads `.env`, ignores extra variables) and the single `settings` object
- `app/database.py` — sync SQLAlchemy `engine` (`pool_pre_ping=True`), `SessionLocal`, declarative `Base`
- `app/dependencies.py` — `get_db` (yields a session, always closes it)
- `app/exceptions.py` — `NotFoundError`, `ConflictError`
- `app/routes/health.py` — `GET /health` and `GET /health/ready`
- `app/workers/celery_app.py` — the Celery app (`celery_app`), Redis broker and result backend
- `app/models/__init__.py` — empty. Convention: every new model file is imported here so Alembic sees it
- `alembic.ini` — Alembic config (no database URL in it)
- `alembic/env.py` — reads the URL from `settings.DATABASE_URL`, uses `Base.metadata`, imports `app.models`
- `alembic/versions/69068f636609_enable_pgvector_extension.py` — first migration, `CREATE EXTENSION IF NOT EXISTS vector`
- `Dockerfile`, `.dockerignore` — one `python:3.12-slim` image used by api and worker
- `docker-compose.yml` — services `db` (pgvector/pgvector:pg16), `redis` (7-alpine), `api`, `worker`; named volume `postgres_data`
- `requirements.txt` — pinned dependencies (fastapi, uvicorn[standard], pydantic-settings, sqlalchemy, alembic, psycopg[binary], celery, redis)
- `.env.example` — template for `.env` (`.env` is git-ignored)
- Empty packages (only `__init__.py`): `app/schemas`, `app/repositories`, `app/services`, `app/clients`
- Placeholder folders (`.gitkeep`): `scripts/`, `evals/`, `frontend/`, `data/raw/`, `tests/`

## Design decisions
- The repository root is the project root (no `fin-copilot/` subfolder) — the repo was created in this directory (0.1)
- Logging is configured once with `logging.basicConfig` at the top of `app/main.py`, no separate logging module; it formats our own loggers only, and uvicorn keeps its own log format (0.1)
- Custom exceptions store their text in `.message`, and the handlers in `main.py` return `{"detail": message}` (0.1)
- `.gitignore` uses `data/raw/*` with `!data/raw/.gitkeep` so the folder stays in git while its contents are ignored (0.1)
- `app/agent/` and the auth parts of `app/dependencies.py` come later (Phase 3 and 1.1) (0.1, updated 0.2)
- psycopg v3 (`postgresql+psycopg://`) is the Postgres driver, not psycopg2 (0.2)
- SQLAlchemy is used in sync mode, matching the sync route handlers (0.2)
- Tables are created only through Alembic migrations, never `Base.metadata.create_all()` (0.2)
- pgvector is enabled in the first migration (`CREATE EXTENSION IF NOT EXISTS vector`), so later migrations can use the vector column type (0.2)
- The project folder is mounted into the api and worker containers (`.:/app`) so code changes reload without rebuilding; only the API auto-reloads (0.2)
- One Docker image is shared by api and worker; each service sets its own command in `docker-compose.yml` (0.2)
- `DATABASE_URL` and `REDIS_URL` have no defaults, so URLs are never hard-coded; the app will not start without a `.env` or real environment variables (0.2)
- `/health/ready` checks the database and Redis separately and reports each, with a 2s Redis timeout so a stopped Redis returns 503 quickly (0.2)
- `SessionLocal` uses `autoflush=False, expire_on_commit=False`, so objects stay readable after the service commits and returns them (0.2)

## Known issues and tech debt
- No pytest yet, so exception handlers and `/health/ready` were verified by hand (until 0.3).
- No ruff config yet. A default `ruff check` flags `B008` on `Depends(get_db)` in `app/routes/health.py`; this is the normal FastAPI pattern, so decide in 0.3 whether to ignore it in the ruff config.
- Alembic's generated `script.py.mako` template still uses `Union[...]` typing; new migrations may need `ruff --fix` once the ruff config exists.

## Milestone log
- 0.1 (2026-10-02): Folder structure, FastAPI app with config, logging and exception handlers, `GET /health`, pinned requirements, `.env.example`, `.gitignore`, this file.
- 0.2 (2026-10-02): Dockerfile and docker-compose (db with pgvector, redis, api, worker), SQLAlchemy engine/session/Base and `get_db`, Alembic with a hand-written pgvector-extension migration, Celery app (no tasks), `GET /health/ready`.
