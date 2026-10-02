# PROJECT_CONTEXT.md

## Current status
- Current phase: Phase 1 (Traditional backend)
- Last completed milestone: 0.3 Quality setup (Phase 0 is complete)
- Next milestone: 1.1 Auth and multi-tenancy

## How to run
- Create local settings: `cp .env.example .env`
- Start services: `docker compose up --build` (db, redis, api on port 8000, worker; add `-d` to run in the background)
- Run migrations: `docker compose exec api alembic upgrade head` (roll back one step: `docker compose exec api alembic downgrade -1`)
- Check it: `curl localhost:8000/health` returns `{"status":"ok"}`; `curl localhost:8000/health/ready` checks database and Redis
- Check the worker: `docker compose exec worker celery -A app.workers.celery_app inspect ping`
- psql shell: `docker compose exec db sh -c 'psql -U $POSTGRES_USER -d $POSTGRES_DB'`
- Reset the database (deletes all data): `docker compose down -v`, then start and migrate again
- Code changes under the project folder reload the API automatically. The worker does not reload: use `docker compose restart worker`
- Run tests: `docker compose exec api pytest` (needs the stack running; creates the `fincopilot_test` database on first run and reuses it afterwards)
- Lint: `docker compose exec api ruff check .` (add `--fix` to apply safe fixes)
- Format: `docker compose exec api ruff format .` (check only: `ruff format --check .`)
- CI: `.github/workflows/ci.yml` runs `ruff check .`, `ruff format --check .` and `pytest` on pushes to `main`/`master` and on all pull requests
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
| TEST_DATABASE_URL | Used only by pytest (`tests/conftest.py`), not part of `Settings`. Same format as `DATABASE_URL`, but the database name must end with `_test` (`fincopilot_test`). Replaces `DATABASE_URL` during tests. Required to run tests |

## Database tables
| Table | Purpose | Key columns / constraints | Added in |
|-------|---------|---------------------------|----------|
| None yet | Only the `vector` (pgvector) extension is enabled, plus Alembic's own `alembic_version` table. The `fincopilot_test` database has the same schema, created by the tests | | 0.2 |

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
- `pyproject.toml` — tool config only: ruff (line length 100, py312, rules E/F/I/B/UP, excludes `alembic/versions` and `*.md`, `fastapi.Depends` treated as immutable for B008, E402 allowed in `tests/conftest.py`) and pytest (`testpaths`, `pythonpath`)
- `requirements-dev.txt` — `-r requirements.txt` plus pinned pytest, httpx (for `TestClient`) and ruff
- `tests/conftest.py` — sets `DATABASE_URL` from `TEST_DATABASE_URL` before the app is imported, the `_test` name guard, session fixture that creates the test database and runs `alembic upgrade head`, and the `db` and `client` fixtures
- `tests/test_health.py`, `tests/test_exceptions.py`, `tests/test_database.py` — health endpoints, exception handlers (404 and 409 via throwaway routes), pgvector extension exists
- `.github/workflows/ci.yml` — CI: one job with pgvector/Postgres and Redis service containers; lint, format check, pytest
- `alembic.ini` — Alembic config (no database URL in it)
- `alembic/env.py` — reads the URL from `settings.DATABASE_URL`, uses `Base.metadata`, imports `app.models`
- `alembic/versions/69068f636609_enable_pgvector_extension.py` — first migration, `CREATE EXTENSION IF NOT EXISTS vector`
- `Dockerfile`, `.dockerignore` — one `python:3.12-slim` image used by api and worker; installs `requirements-dev.txt`
- `docker-compose.yml` — services `db` (pgvector/pgvector:pg16), `redis` (7-alpine), `api`, `worker`; named volume `postgres_data`
- `requirements.txt` — pinned dependencies (fastapi, uvicorn[standard], pydantic-settings, sqlalchemy, alembic, psycopg[binary], celery, redis)
- `.env.example` — template for `.env` (`.env` is git-ignored); includes `TEST_DATABASE_URL`
- Empty packages (only `__init__.py`): `app/schemas`, `app/repositories`, `app/services`, `app/clients`
- Placeholder folders (`.gitkeep`): `scripts/`, `evals/`, `frontend/`, `data/raw/`

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
- Tests run against a real Postgres database (`fincopilot_test`), never SQLite, because the app uses pgvector and Postgres full-text search (0.3)
- `tests/conftest.py` replaces `DATABASE_URL` with `TEST_DATABASE_URL` before importing the app, and aborts the run unless the database name ends with `_test`. The app, Alembic and the tests therefore cannot touch the development database. `TEST_DATABASE_URL` is deliberately not in `Settings` (0.3)
- The test database is created once and reused; the schema comes from `alembic upgrade head` (so migrations are tested too). Nothing is dropped between runs (0.3)
- Each test runs inside an outer transaction that is rolled back at the end; the `Session` uses `join_transaction_mode="create_savepoint"`, so service `db.commit()` calls only release a savepoint and nothing persists between tests (0.3)
- Dev dependencies (pytest, httpx, ruff) are installed in the shared Docker image, since this is a dev-only project for now. Split into separate images if the project is ever deployed (0.3)
- Ruff rules are E, F, I, B, UP. B008 is handled with `extend-immutable-calls = ["fastapi.Depends"]` instead of being disabled. `*.md` is excluded because ruff 0.16 also formats code blocks in Markdown and would rewrite `CLAUDE.md` (0.3)
- CI uses GitHub Actions service containers (`pgvector/pgvector:pg16`, `redis:7-alpine`) with throwaway credentials, and sets the environment variables at job level instead of using a `.env` file (0.3)

## Known issues and tech debt
- The `db` fixture's rollback isolation has no test yet, because no table exists. Its first real test (a write that does not leak into the next test, and a service `commit()` that is rolled back) goes in 1.1 with the first table.
- Starlette's `TestClient` warns that using it with `httpx` is deprecated and suggests `httpx2`. We follow the milestone spec (httpx); revisit when Starlette actually removes httpx support.
- Alembic's generated `script.py.mako` template still uses `Union[...]` typing. Ruff skips `alembic/versions`, so no lint failure, but new migrations keep the old style.
- `alembic/env.py` calls `fileConfig`, which can disable existing loggers in the test process when `alembic upgrade` runs. Nothing depends on captured app logs yet; keep in mind if a test needs `caplog`.

## Milestone log
- 0.1 (2026-10-02): Folder structure, FastAPI app with config, logging and exception handlers, `GET /health`, pinned requirements, `.env.example`, `.gitignore`, this file.
- 0.2 (2026-10-02): Dockerfile and docker-compose (db with pgvector, redis, api, worker), SQLAlchemy engine/session/Base and `get_db`, Alembic with a hand-written pgvector-extension migration, Celery app (no tasks), `GET /health/ready`.
- 0.3 (2026-10-02): pytest against a real Postgres test database (`fincopilot_test`, safety guard, savepoint rollback per test, `db` and `client` fixtures), 5 tests, ruff config (E/F/I/B/UP), `requirements-dev.txt` installed in the Docker image, GitHub Actions workflow with Postgres and Redis service containers.
