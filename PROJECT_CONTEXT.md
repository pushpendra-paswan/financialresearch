# PROJECT_CONTEXT.md

## Current status
- Current phase: Phase 1 (Traditional backend)
- Last completed milestone: 1.3 Watchlists
- Next milestone: 1.4 SEC client and filing ingestion

## How to run
- Create local settings: `cp .env.example .env`, then set `JWT_SECRET_KEY` (generate with `openssl rand -hex 32`) and `SEC_USER_AGENT` (`FinCopilot your.name@example.com`, a real contact email). The app and the tests will not start without them. After changing `.env`, run `docker compose up -d` so the containers pick it up
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
- Seed / backfill commands: `docker compose exec api python -m scripts.seed_companies` (downloads the SEC ticker file and loads the 34 tickers listed in the script; safe to run repeatedly. Exits with status 1 and writes nothing if a ticker is missing from the SEC data or two tickers share a CIK)
- Seeded tickers (the `TICKERS` list in `scripts/seed_companies.py`): AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD, INTC, ORCL, CRM, ADBE, JPM, BAC, GS, MS, WFC, V, MA, JNJ, PFE, UNH, LLY, MRK, XOM, CVX, WMT, KO, PEP, MCD, NKE, DIS, BA, CAT

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
| JWT_SECRET_KEY | Secret that signs login tokens. Required, no default. Generate with `openssl rand -hex 32` |
| JWT_ALGORITHM | JWT signing algorithm. Default "HS256" |
| ACCESS_TOKEN_EXPIRE_MINUTES | Lifetime of an access token in minutes. Default 60 |
| SEC_USER_AGENT | Sent as the `User-Agent` on every SEC request. Required, no default. Must identify you with a real contact email, in the form `FinCopilot your.name@example.com` |
| RAW_DATA_DIR | Folder for raw downloaded files, relative to the project root. Default `data/raw` |
| TEST_DATABASE_URL | Used only by pytest (`tests/conftest.py`), not part of `Settings`. Same format as `DATABASE_URL`, but the database name must end with `_test` (`fincopilot_test`). Replaces `DATABASE_URL` during tests. Required to run tests |

## Database tables
| Table | Purpose | Key columns / constraints | Added in |
|-------|---------|---------------------------|----------|
| organizations | A tenant (a team) | `id` (identity), `name`, `created_at` | 1.1 |
| users | Members of an organization | `id`, `org_id` (FK, indexed), `email` (unique, lowercase), `hashed_password`, `role` (string, CHECK `ck_users_role` in admin/analyst/viewer), `created_at` | 1.1 |
| audit_logs | Who did what | `id`, `org_id` (FK, indexed), `user_id` (FK), `action` (e.g. `user.create`), `entity_id` (nullable), `created_at` | 1.1 |
| companies | Shared public catalog of US-listed companies (no `org_id`) | `id` (identity), `ticker` (unique), `cik` (unique, 10-character zero-padded string, e.g. `0000320193`), `name`, `exchange` (nullable), `created_at`. No extra indexes: the unique constraints are enough for ~34 rows. Sector arrives in 1.4 | 1.2 |
| watchlists | A named list of companies a team tracks (private, shared by the whole organization) | `id` (identity), `org_id` (FK, indexed), `created_by` (FK to users), `name` (100), `created_at`. UNIQUE INDEX `uq_watchlists_org_id_lower_name` on `(org_id, lower(name))`, so names are unique per organization ignoring letter case | 1.3 |
| watchlist_items | A company in a watchlist | PRIMARY KEY `(watchlist_id, company_id)` (no id column, no other indexes), `watchlist_id` FK with `ON DELETE CASCADE`, `company_id` FK to companies with no cascade, `added_at` | 1.3 |

The `vector` (pgvector) extension is also enabled, plus Alembic's `alembic_version` table. The `fincopilot_test` database has the same schema, created by the tests. All `created_at` columns are `timestamptz` with server default `now()`.

## API endpoints
| Method | Path | Purpose | Auth / role |
|--------|------|---------|-------------|
| GET | /health | Returns `{"status": "ok"}` (the app is running) | None |
| GET | /health/ready | Runs `SELECT 1` and a Redis ping. 200 `{"status": "ready", "database": "ok", "redis": "ok"}`, or 503 with `"error"` for whichever failed | None |
| POST | /auth/register | Creates a new organization and its first user (admin). 201 with `{access_token, token_type}`. 409 if the email exists, 422 for a bad password | None |
| POST | /auth/login | Email and password, returns `{access_token, token_type}`. 401 "Invalid email or password" for an unknown email or wrong password | None |
| GET | /auth/me | The current user | Any logged-in user |
| POST | /users | Creates a user (admin, analyst or viewer) in the caller's organization. 201. 409 if the email exists anywhere | Admin |
| GET | /users | Lists users of the caller's organization (no pagination) | Any logged-in user |
| GET | /users/{user_id} | One user of the caller's organization. 404 "User not found" if missing or in another organization | Any logged-in user |
| GET | /companies | Paginated company list. Query: `search` (max 50 chars; literal, case-insensitive substring match on ticker or name; exact ticker match first, then by ticker), `page` (>= 1, default 1), `page_size` (1-100, default 20). Returns `{items, total, page, page_size}`. 422 for invalid values | Any logged-in user |
| GET | /companies/{ticker} | One company by ticker (case-insensitive). 404 "Company not found" if unknown | Any logged-in user |
| POST | /watchlists | Creates a watchlist `{name}` (stripped, 1-100 characters). 201 with `{id, name, created_by, created_at, item_count}`. 409 if the name exists in the organization (any letter case) | Admin or analyst |
| GET | /watchlists | The organization's watchlists with `item_count`, ordered by name ignoring case. No pagination | Any logged-in user |
| GET | /watchlists/{watchlist_id} | One watchlist with its `companies` (full company objects, sorted by ticker). 404 "Watchlist not found" if missing or in another organization | Any logged-in user |
| PATCH | /watchlists/{watchlist_id} | Renames it `{name}`. 200 with the summary. 409 only if ANOTHER watchlist has the name; the same name or a case-only change is allowed | Admin or analyst |
| DELETE | /watchlists/{watchlist_id} | Deletes it and its items (database cascade); companies stay in the catalog. 204 | Admin or analyst |
| POST | /watchlists/{watchlist_id}/companies | Adds a company `{ticker}` (stripped, 1-15 characters, case-insensitive). 201 with the full updated watchlist. 404 "Company not found", 409 "Company is already in this watchlist" | Admin or analyst |
| DELETE | /watchlists/{watchlist_id}/companies/{ticker} | Removes a company. 204. 404 "Company not found" or "Company is not in this watchlist" | Admin or analyst |

Missing, invalid or expired token: 401 with `WWW-Authenticate: Bearer`. Wrong role: 403 (a viewer gets 403 on every watchlist write, before the watchlist is looked up). Another organization's watchlist: 404 "Watchlist not found", identical to a missing id.

## Background jobs
| Task | Schedule | What it does |
|------|----------|--------------|
| None yet | | The Celery app exists and the worker runs, but there are no tasks. Beat comes in 1.4 |

## Key files
- `app/main.py` — FastAPI app, logging setup, router registration, exception handlers (NotFoundError → 404, ConflictError → 409, UnauthorizedError → 401 with `WWW-Authenticate: Bearer`, ForbiddenError → 403)
- `app/config.py` — `Settings` (pydantic-settings, reads `.env`, ignores extra variables, includes the JWT settings) and the single `settings` object
- `app/database.py` — sync SQLAlchemy `engine` (`pool_pre_ping=True`), `SessionLocal`, declarative `Base`
- `app/dependencies.py` — `get_db` (yields a session, always closes it), `get_current_user` (bearer token → user, 401 otherwise), `require_admin` (403 unless admin), `require_editor` (403 unless admin or analyst)
- `app/exceptions.py` — `NotFoundError`, `ConflictError`, `UnauthorizedError`, `ForbiddenError`
- `app/security.py` — `hash_password`, `verify_password` (bcrypt), `create_access_token`, `decode_access_token` (PyJWT)
- `app/models/organizations.py`, `users.py` (also `UserRole`), `audit.py` — the 1.1 tables; `companies.py` — the 1.2 table (no `org_id`)
- `app/schemas/auth.py`, `users.py` — request and response schemas; `check_password_bytes` (72-byte limit) is shared by the register, login and create-user schemas
- `app/repositories/organizations.py`, `users.py`, `audit.py` — queries; add and flush, never commit
- `app/services/auth.py` (`register`, `login`) and `users.py` (`create_user`, `list_users`, `get_user`) — business logic, own the commits
- `app/routes/auth.py`, `users.py` — the 1.1 endpoints
- `app/routes/health.py` — `GET /health` and `GET /health/ready`
- `app/clients/sec.py` — `get_company_tickers()`: downloads the SEC ticker file, saves the raw response to `<RAW_DATA_DIR>/sec/company_tickers_exchange.json`, returns dicts with `cik` (padded), `ticker`, `name`, `exchange`. The first external client
- `app/schemas/companies.py`, `app/repositories/companies.py` (`get_by_ticker`, `get_by_cik`, `create`, `search_companies`), `app/services/companies.py` (`list_companies`, `get_company`, `seed_companies`), `app/routes/companies.py` — the 1.2 catalog
- `scripts/seed_companies.py` — the seed command (flat script with the `TICKERS` list); `scripts/__init__.py` makes `python -m scripts.seed_companies` work
- `alembic/versions/bf7faefe32ae_create_companies_table.py` — companies table (autogenerated, reviewed)
- `tests/fixtures/company_tickers_exchange.json` — 8-company fixture in the real columnar structure (includes GOOGL and GOOG sharing a CIK, and a null exchange); `tests/test_sec_client.py`, `tests/test_companies_seed.py`, `tests/test_companies.py`. The `sec_rows` fixture in `conftest.py` gives the parsed fixture (real client code, download mocked)
- `app/workers/celery_app.py` — the Celery app (`celery_app`), Redis broker and result backend
- `app/models/watchlists.py` (`Watchlist` and `WatchlistItem`, with the functional unique index declared after the class), `app/schemas/watchlists.py`, `app/repositories/watchlists.py` (reads take `org_id`; the item reads join through `watchlists.org_id`; the writes at the bottom rely on the service having checked ownership), `app/services/watchlists.py`, `app/routes/watchlists.py` — the 1.3 watchlists
- `alembic/versions/ab4b8dd0de03_create_watchlists_tables.py` — watchlists and watchlist_items (autogenerated, reviewed; it already renders the `lower(name)` index correctly)
- `tests/test_watchlists.py` — 48 tests: auth, roles, names, list/detail, rename, delete, items, cross-organization 404 (parametrized, with a direct database check that organization A's items did not change). The `seeded` fixture (seeds 7 catalog companies with the SEC client mocked) now lives in `tests/conftest.py`, moved from `test_companies.py` because two test files use it
- `app/models/__init__.py` — imports every model file so Alembic sees it. Convention: every new model file is imported here
- `pyproject.toml` — tool config only: ruff (line length 100, py312, rules E/F/I/B/UP, excludes `alembic/versions` and `*.md`, `fastapi.Depends` treated as immutable for B008, E402 allowed in `tests/conftest.py`) and pytest (`testpaths`, `pythonpath`)
- `requirements-dev.txt` — `-r requirements.txt` plus pinned pytest and ruff (httpx is in `requirements.txt` now and also serves `TestClient`)
- `tests/conftest.py` — sets `DATABASE_URL` from `TEST_DATABASE_URL` before the app is imported, the `_test` name guard, session fixture that creates the test database and runs `alembic upgrade head`, and the `db` and `client` fixtures
- `tests/test_health.py`, `tests/test_exceptions.py`, `tests/test_database.py` — health endpoints, exception handlers (404, 409, 401, 403 via throwaway routes), pgvector extension exists, rollback isolation of service commits
- `tests/test_auth.py`, `tests/test_users.py` — registration, login, token rejection, roles, cross-organization isolation. The `register_org` fixture in `conftest.py` registers an organization and returns the admin's auth headers
- `.github/workflows/ci.yml` — CI: one job with pgvector/Postgres and Redis service containers; lint, format check, pytest
- `alembic.ini` — Alembic config (no database URL in it)
- `alembic/env.py` — reads the URL from `settings.DATABASE_URL`, uses `Base.metadata`, imports `app.models`
- `alembic/versions/69068f636609_enable_pgvector_extension.py` — first migration, `CREATE EXTENSION IF NOT EXISTS vector`
- `alembic/versions/e1289657f92f_create_auth_tables.py` — organizations, users, audit_logs (autogenerated, reviewed)
- `Dockerfile`, `.dockerignore` — one `python:3.12-slim` image used by api and worker; installs `requirements-dev.txt`
- `docker-compose.yml` — services `db` (pgvector/pgvector:pg16), `redis` (7-alpine), `api`, `worker`; named volume `postgres_data`
- `requirements.txt` — pinned dependencies (fastapi, uvicorn[standard], pydantic-settings, sqlalchemy, alembic, psycopg[binary], celery, redis, PyJWT, bcrypt, email-validator, httpx)
- `.env.example` — template for `.env` (`.env` is git-ignored); includes `TEST_DATABASE_URL`
- Placeholder folders (`.gitkeep`): `scripts/`, `evals/`, `frontend/`, `data/raw/`

## Design decisions
- The repository root is the project root (no `fin-copilot/` subfolder) — the repo was created in this directory (0.1)
- Logging is configured once with `logging.basicConfig` at the top of `app/main.py`, no separate logging module; it formats our own loggers only, and uvicorn keeps its own log format (0.1)
- Custom exceptions store their text in `.message`, and the handlers in `main.py` return `{"detail": message}` (0.1)
- `.gitignore` uses `data/raw/*` with `!data/raw/.gitkeep` so the folder stays in git while its contents are ignored (0.1)
- `app/agent/` comes later (Phase 3) (0.1, updated 0.2, 1.1)
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
- Ids are integer identity columns (`GENERATED BY DEFAULT AS IDENTITY`), declared with `Identity()` on the model; a plain integer primary key would produce `SERIAL` (1.1)
- Emails are globally unique (one account per email across all organizations) and stored lowercase; the services lowercase before every lookup and insert (1.1)
- Roles are a plain string column with a named CHECK constraint (`ck_users_role`) and a matching Python `UserRole` enum, not a native Postgres enum, so adding a role is a simple constraint change (1.1)
- Passwords are hashed with the `bcrypt` library directly (no passlib). bcrypt only uses the first 72 bytes, so schemas reject longer passwords (422) instead of silently truncating them. Login also rejects over 72 bytes (422) because bcrypt would raise on it (1.1)
- Access tokens are HS256 JWTs holding only the user id (`sub`, as a string because PyJWT requires it) and an expiry. Role and organization are read from the database on every request, so a role change applies immediately (1.1)
- Unknown email and wrong password return the identical 401 "Invalid email or password", so accounts cannot be discovered (1.1)
- Cross-organization access returns 404 with the same body as a missing row, never 403 (1.1)
- Exactly two user queries have no `org_id`, each with a comment in `repositories/users.py`: `get_by_email` (login must find the user before the organization is known; emails are globally unique) and `get_by_id_for_auth` (the token only holds the user id, and this is how the caller's organization is learned). Every other user query takes `org_id` (1.1)
- Services own `db.commit()` (one commit per service call); repositories only add and flush (1.1)
- `require_admin` is a plain dependency, not a role-checking factory. Checks for other roles are added in the milestones that need them (1.1)
- `get_current_user` uses `HTTPBearer(auto_error=False)` so a missing token raises our own `UnauthorizedError` and has the same `{"detail": ...}` format and `WWW-Authenticate` header as other 401s (1.1)
- Companies are shared public data: the table has no `org_id` and no function in `repositories/companies.py` takes one (1.2)
- The CIK is stored as a zero-padded 10-character string, because that is the format the data.sec.gov endpoints need in 1.4 and 1.5. The SEC ticker file gives it as an integer, so the client pads it (1.2)
- One row per company, so only one ticker per company is seeded (GOOGL, not GOOG; both share CIK 1652044) (1.2)
- The seed matches existing rows by CIK (a CIK never changes, a ticker can), validates everything before writing anything, and commits once. It is idempotent: a second run creates 0 rows and refreshes ticker, name and exchange. Its "updated" count includes unchanged rows (1.2)
- The seed raises `ValueError` (not an HTTP error) because it is a command-line failure; the script catches only that, logs it and exits non-zero. There is no audit row, because audit_logs needs an organization and this is a system action (1.2)
- The sector column is deferred to 1.4, because the ticker file has no sector; it comes from the SEC submissions endpoint (1.2)
- The SEC ticker file is columnar (`{"fields": [...], "data": [[...]]}`); the client reads rows by field name, not position. The raw file is saved under `data/raw/sec/` before parsing (1.2)
- The detail route uses the ticker, not the id, and the ticker is case-insensitive (uppercased in the service) (1.2)
- Pagination uses `page` and `page_size` and returns the total, computed from the same filter as the page. There is no generic pagination utility (1.2)
- Search is a literal (`icontains(..., autoescape=True)`, so `%` and `_` are not wildcards), case-insensitive substring match on ticker or name. An exact ticker match (ignoring case) sorts first, then by ticker. Text that is empty after stripping means no filter (1.2)
- Any logged-in role (admin, analyst, viewer) can read the catalog (1.2)
- Watchlists are shared team resources: any admin or analyst of the organization can change ANY of its watchlists, not only the ones they created. `created_by` is only a record (1.3)
- Viewers are read-only. `require_editor` (admin or analyst) guards every write on organization data; it is a plain dependency like `require_admin`, not a role factory (1.3)
- Watchlist names are unique per organization ignoring letter case, enforced by a functional unique index on `(org_id, lower(name))`. The service also checks `get_by_name` first so the user gets a clean 409 (1.3)
- A rename is a conflict only when ANOTHER watchlist holds the name, so renaming to the same name or changing only the letter case of its own name ("tech" to "Tech") is allowed (1.3)
- `watchlist_items` has a composite primary key `(watchlist_id, company_id)` and no id column. The FK to watchlists is `ON DELETE CASCADE`; the FK to companies has no cascade, so deleting a watchlist never touches the catalog (1.3)
- Item reads are scoped through the watchlist's `org_id` (a join to `watchlists`). The item writes (`add_item`, `remove_item`, `delete_watchlist`) do not take `org_id`; they rely on the service having loaded the watchlist with `get_by_id(org_id)` first, and the cross-organization tests prove nothing changes (1.3)
- Services load the watchlist (org-scoped) BEFORE looking at the company, so a caller from another organization always gets "Watchlist not found" and learns nothing about companies (1.3)
- Companies are added and removed by ticker (case-insensitive, uppercased in the service), the same key as `GET /companies/{ticker}`. The add route returns the full updated watchlist (1.3)
- Audit rows for watchlist actions (`watchlist.create`, `.rename`, `.delete`, `.add_company`, `.remove_company`) always use the watchlist id as `entity_id` (1.3)

## Known issues and tech debt
- The SEC client has no throttling or retries yet (one request per seed run). Milestone 1.4 adds the 5 requests per second throttle and retries when the client makes many requests.
- If the SEC reassigns a ticker to a different CIK while another seeded row still holds it, the seed's update would hit the unique constraint on `ticker` and fail with nothing committed. Very unlikely for the current large-cap list; not handled.
- Files written by containers (e.g. `data/raw/sec/`) are owned by root on the host, because the containers run as root. Deleting them from the host needs `sudo`.
- Two simultaneous registrations (or user creations) with the same email both pass the "email exists" check, and the second then fails on the unique constraint, which surfaces as a 500 instead of a 409. Not handled yet.
- Login is faster for an unknown email than for a wrong password (bcrypt is skipped when no user is found), so response timing can reveal whether an email has an account even though the error bodies are identical. Fix later by verifying against a dummy hash.
- Access tokens cannot be revoked before they expire (no logout, refresh or deactivation yet). Deleting a user does invalidate their token, because the user is loaded on every request.
- Two simultaneous requests creating or renaming to the same watchlist name, or adding the same company to a watchlist, both pass the service check and the second then fails on the unique index or primary key, which surfaces as a 500 instead of a 409 (same class of issue as concurrent registration). Not handled yet.
- `GET /watchlists` is not paginated (the number of watchlists per organization is expected to be small).
- Audit rows for `watchlist.add_company` and `watchlist.remove_company` record only the watchlist id, not which company was added or removed.
- Starlette's `TestClient` warns that using it with `httpx` is deprecated and suggests `httpx2`. We follow the milestone spec (httpx); revisit when Starlette actually removes httpx support.
- Alembic's generated `script.py.mako` template still uses `Union[...]` typing. Ruff skips `alembic/versions`, so no lint failure, but new migrations keep the old style.
- `alembic/env.py` calls `fileConfig`, which can disable existing loggers in the test process when `alembic upgrade` runs. Nothing depends on captured app logs yet; keep in mind if a test needs `caplog`.

## Milestone log
- 0.1 (2026-10-02): Folder structure, FastAPI app with config, logging and exception handlers, `GET /health`, pinned requirements, `.env.example`, `.gitignore`, this file.
- 0.2 (2026-10-02): Dockerfile and docker-compose (db with pgvector, redis, api, worker), SQLAlchemy engine/session/Base and `get_db`, Alembic with a hand-written pgvector-extension migration, Celery app (no tasks), `GET /health/ready`.
- 0.3 (2026-10-02): pytest against a real Postgres test database (`fincopilot_test`, safety guard, savepoint rollback per test, `db` and `client` fixtures), 5 tests, ruff config (E/F/I/B/UP), `requirements-dev.txt` installed in the Docker image, GitHub Actions workflow with Postgres and Redis service containers.
- 1.1 (2026-10-02): Organizations, users and audit_logs tables (one migration), bcrypt + JWT security module, register/login/me and admin-only user creation, org-scoped user list/detail with 404 for other organizations, 401/403 handlers, 34 tests including cross-organization isolation and the db-fixture rollback test.
- 1.2 (2026-10-02): `companies` table (one migration, no org_id), SEC client for the ticker file with raw-file saving, idempotent seed script for 34 tickers (matches by CIK, validates before writing), paginated and searchable `GET /companies` and `GET /companies/{ticker}` for any role, 23 new tests (57 total). `SEC_USER_AGENT` and `RAW_DATA_DIR` settings; httpx moved to runtime requirements.
- 1.3 (2026-10-02): `watchlists` and `watchlist_items` tables (one migration, case-insensitive unique name per organization via a functional index, composite primary key with cascade), `require_editor` dependency, 7 org-scoped watchlist endpoints (viewers read-only, other organizations get 404), audit rows, 48 new tests (105 total). `CLAUDE.md` data model line and a roles rule updated. `seeded` test fixture moved to `conftest.py`.
