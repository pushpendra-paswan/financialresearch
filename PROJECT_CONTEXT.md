# PROJECT_CONTEXT.md

## Current status
- Current phase: Phase 0 (Foundation)
- Last completed milestone: 0.1 Project skeleton
- Next milestone: 0.2 Local infrastructure (Docker Compose, Postgres + pgvector, Redis, Celery worker, SQLAlchemy + Alembic)

## How to run
- Create virtual environment: `python3.12 -m venv .venv && source .venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`
- Create local settings: `cp .env.example .env`
- Start server: `uvicorn app.main:app --reload --port 8000`
- Check it: `curl http://localhost:8000/health` returns `{"status":"ok"}`
- Start services (Docker): not set up yet (milestone 0.2)
- Run migrations: not set up yet (milestone 0.2)
- Run tests: not set up yet (milestone 0.3)
- Seed / backfill commands: None yet

## Environment variables
| Name | Purpose |
|------|---------|
| APP_NAME | Application title (FastAPI title). Default "Financial Research Copilot" |
| ENVIRONMENT | Environment name, e.g. development. Default "development" |
| LOG_LEVEL | Python logging level for our own loggers (DEBUG, INFO, WARNING, ...). Default "INFO" |

## Database tables
| Table | Purpose | Key columns / constraints | Added in |
|-------|---------|---------------------------|----------|
| None yet | | | |

## API endpoints
| Method | Path | Purpose | Auth / role |
|--------|------|---------|-------------|
| GET | /health | Returns `{"status": "ok"}` | None |

## Background jobs
| Task | Schedule | What it does |
|------|----------|--------------|
| None yet | | |

## Key files
- `app/main.py` — FastAPI app, logging setup, router registration, exception handlers (NotFoundError → 404, ConflictError → 409)
- `app/config.py` — `Settings` (pydantic-settings, reads `.env`) and the single `settings` object
- `app/exceptions.py` — `NotFoundError`, `ConflictError`
- `app/routes/health.py` — `GET /health`; the pattern for writing and registering routes
- `requirements.txt` — pinned dependencies (fastapi, uvicorn[standard], pydantic-settings)
- `.env.example` — template for `.env` (`.env` is git-ignored)
- Empty packages (only `__init__.py`): `app/models`, `app/schemas`, `app/repositories`, `app/services`, `app/clients`, `app/workers`
- Placeholder folders (`.gitkeep`): `alembic/`, `scripts/`, `evals/`, `frontend/`, `data/raw/`, `tests/`

## Design decisions
- The repository root is the project root (no `fin-copilot/` subfolder) — the repo was created in this directory (0.1)
- Logging is configured once with `logging.basicConfig` at the top of `app/main.py`, no separate logging module; it formats our own loggers only, and uvicorn keeps its own log format (0.1)
- Custom exceptions store their text in `.message`, and the handlers in `main.py` return `{"detail": message}` (0.1)
- `.gitignore` uses `data/raw/*` with `!data/raw/.gitkeep` so the folder stays in git while its contents are ignored (0.1)
- `app/agent/`, `app/database.py` and `app/dependencies.py` are not created yet; they come with Phase 3, 0.2 and 1.1 (0.1)

## Known issues and tech debt
- None. Exception handlers were verified by hand only (no pytest until 0.3).

## Milestone log
- 0.1 (2026-10-02): Folder structure, FastAPI app with config, logging and exception handlers, `GET /health`, pinned requirements, `.env.example`, `.gitignore`, this file.
