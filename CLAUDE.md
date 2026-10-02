# CLAUDE.md — Financial Research Copilot

This file tells you (Claude Code) what this project is, how the code must be written, and how to work through it milestone by milestone. Read it fully at the start of every session, together with `PROJECT_CONTEXT.md`.

---

## 1. What we are building

A multi-tenant web application for financial research on US-listed companies. It evolves in three phases:

1. **Traditional backend**: teams (organizations) track companies in watchlists, see filings, financial numbers and daily prices, and set price alerts. Data is ingested automatically by scheduled background jobs from SEC EDGAR and a price provider.
2. **RAG**: users ask questions about company filings (10-K, 10-Q) and get answers grounded only in filing text, with citations to the exact source passage.
3. **Agentic AI**: a research agent plans multi-step tasks (retrieve filings, pull financials and prices, compute metrics, compare companies, draft reports). Read-only actions run automatically; actions that write data require explicit user approval.

Each phase builds on the previous one. Never change a working earlier phase without a clear reason, and note any such change in `PROJECT_CONTEXT.md`.

This is a learning and portfolio project. The developer must be able to read and explain every line, so clarity beats cleverness everywhere.

---

## 2. Tech stack

- Python 3.12
- FastAPI (sync route handlers, `def` not `async def`, unless streaming or a clear reason requires async)
- SQLAlchemy 2.0 (sync, 2.0-style `select()` queries) + Alembic
- PostgreSQL 16 with the pgvector extension
- Redis (cache, Celery broker)
- Celery + Celery beat (background and scheduled jobs)
- Pydantic v2 + pydantic-settings
- JWT auth (PyJWT) + bcrypt password hashing
- httpx for external HTTP calls
- BeautifulSoup + lxml for filing HTML parsing
- LangChain 1.x for Phase 2 RAG: `langchain-core`, `langchain-text-splitters`, and the provider integration packages for the chosen chat model and embeddings. Section 6 defines how LangChain must be used.
- pgvector Python package for the vector column in SQLAlchemy models
- LangGraph for the agent (Phase 3)
- Embedding model, LLM provider and reranker: chosen at the milestone where they are first needed (embeddings in 2.2, chat model in 2.4, reranker in 2.5), and recorded in `PROJECT_CONTEXT.md`
- Langfuse for LLM tracing (Phase 3)
- pytest, ruff
- Docker Compose for local development
- Frontend: plain HTML, CSS and JavaScript served by FastAPI as static files. No frontend framework.

Do not add a new library without stating why in the milestone plan.

---

## 3. Coding style (most important section)

### 3.1 Plain, sequential code
- Write code that reads top to bottom like a recipe. A reader should understand a function without jumping to other functions.
- Prefer one clear function with sequential steps over many tiny functions.
- Use short comments to label steps and to explain *why*, not *what*.

### 3.2 Helper functions
- Do **not** create a helper function unless:
  - the same logic is used in two or more places, or
  - the logic is genuinely complex and naming it makes the calling code clearer.
- Do **not** create chains of helpers (a helper that calls another helper that calls another). Maximum one level of helper below a service function.
- Do not wrap a single library call in a function just to rename it.

### 3.3 Files
- Do **not** split code into extra files without a reason. Add new code to the existing file for that domain and layer.
- A new file is justified only for a new domain (e.g. `alerts`), a new layer, or when a file becomes very hard to navigate (roughly 400+ lines). State the reason in the milestone plan.

### 3.4 No premature abstraction
- No generic base repository, no base service classes, no factories, no dependency-injection containers beyond FastAPI's `Depends`.
- No classes where plain functions work. Allowed classes: SQLAlchemy models, Pydantic schemas, custom exceptions, and the `PriceProvider` interface (an intentional design choice so the price source can be swapped).
- Do not build for hypothetical future needs. Build what the current milestone needs.

### 3.5 General rules
- Type hints on all function signatures.
- Clear, descriptive names; no abbreviations except common ones (`id`, `db`, `url`).
- Use `logging`, never `print`.
- Catch specific exceptions only; never a bare `except:`.
- All configuration comes from environment variables via `app/config.py`. No hard-coded secrets, URLs or keys.
- Format and lint with ruff before finishing a milestone.

### 3.6 Example

Preferred:

```python
def create_watchlist(db: Session, org_id: int, user_id: int, data: WatchlistCreate) -> Watchlist:
    # Names must be unique within an organization
    existing = watchlist_repository.get_by_name(db, org_id, data.name)
    if existing:
        raise ConflictError("A watchlist with this name already exists")

    watchlist = watchlist_repository.create(db, org_id, user_id, data.name)

    audit_repository.create(db, org_id, user_id, action="watchlist.create", entity_id=watchlist.id)
    db.commit()
    return watchlist
```

Avoid:

```python
def create_watchlist(db, org_id, user_id, data):
    _validate_watchlist(db, org_id, data)          # calls _check_name, which calls _query_name...
    watchlist = _build_and_save_watchlist(db, org_id, user_id, data)
    _record_audit(db, org_id, user_id, watchlist)
    return _finalize(db, watchlist)
```

---

## 4. Project structure

Follow a basic layered structure: **route → service → repository**.

```
fin-copilot/
├── CLAUDE.md
├── PROJECT_CONTEXT.md
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── alembic/
├── app/
│   ├── main.py            # FastAPI app, router registration, exception handlers
│   ├── config.py          # Settings from environment variables
│   ├── database.py        # Engine and session
│   ├── dependencies.py    # get_db, get_current_user, role checks
│   ├── exceptions.py      # Custom exceptions (NotFoundError, ConflictError, ...)
│   ├── security.py        # Password hashing and JWT create/decode
│   ├── models/            # SQLAlchemy models, one file per domain
│   ├── schemas/           # Pydantic request/response schemas, one file per domain
│   ├── repositories/      # Database queries only, one file per domain
│   ├── services/          # Business logic, one file per domain
│   ├── routes/            # FastAPI routers, one file per domain
│   ├── clients/           # External services: SEC, price provider, LLM, embeddings
│   ├── workers/           # Celery app, tasks and beat schedule
│   └── agent/             # Phase 3 only: agent tools and LangGraph graph
├── scripts/               # One-off commands: seed companies, backfill prices
├── evals/                 # Phase 2/3 evaluation questions, scripts and results
├── frontend/              # Plain HTML, CSS, JS
├── data/raw/              # Raw downloaded SEC/price files (git-ignored)
└── tests/
```

Domains (use these names consistently across layers): `auth`, `organizations`, `users`, `companies`, `watchlists`, `filings`, `financials`, `prices`, `alerts`, `ingestion`, `audit`, `chunks`, `retrieval`, `chat`, `reports`, `agent`.

### Layer rules

- **Routes**: receive the request, call one service function, return a response schema. No database queries and no business logic in routes.
- **Services**: business logic and orchestration. Call repositories and clients. Own the transaction (`db.commit()`). Raise custom exceptions from `app/exceptions.py`, never `HTTPException`. `main.py` maps custom exceptions to HTTP responses.
- **Repositories**: SQLAlchemy queries only. No business rules, no commits, no external calls.
- **Clients**: all calls to external systems (SEC, price provider, LLM, embeddings). Nothing else talks to the network.
- **Workers**: Celery tasks are thin. They open a DB session, call a service function, and log the result.

RAG logic lives in services and repositories like everything else:
- `services/filings.py`: filing HTML parsing and section extraction (2.1)
- `services/chunks.py` + `repositories/chunks.py`: splitting, embedding, storing chunks, and the vector and full-text SQL queries (2.2, 2.3)
- `services/retrieval.py`: hybrid search, merging and reranking (2.3, 2.5)
- `services/chat.py` + `repositories/chat.py`: the Q&A flow, citations and chat history (2.4)
- `clients/llm.py`: creates the LangChain chat model and embeddings objects from config; services import them from here

Only the agent gets its own folder, because its tools and graph are a distinct layer.

---

## 5. Data model

Shared public data (no `org_id`; every organization reads the same rows):
- `companies` (ticker unique, cik unique as a 10-character zero-padded string, name, exchange, industry = the SEC's SIC description; there is no GICS sector)
- `filings` (company_id, accession_number unique, form_type, filed_on, report_date, fiscal_year, primary_document, raw_path relative to RAW_DATA_DIR)
- `financial_facts` (company_id, concept, value, unit, period_end, fiscal_year, fiscal_period)
- `price_bars` (company_id, trade_date, open, high, low, close, volume; unique on company_id + trade_date)
- `document_chunks` (filing_id, company_id, section, fiscal_year, content, embedding)
- `ingestion_runs` (job_type, status running/success/partial/failed, started_at, finished_at, message, error)

Private organization data (every table has `org_id`):
- `organizations`, `users` (org_id, email, role: admin / analyst / viewer)
- `watchlists` (org_id, created_by, name; name unique per organization, case-insensitive), `watchlist_items` (primary key is watchlist_id + company_id; deleting a watchlist cascades to its items)
- `alerts` (user_id, company_id, condition, active), `notifications` (alert_id, sent_at, is_read)
- `audit_logs` (org_id, user_id, action, entity_id)
- `chat_sessions`, `chat_messages`, `citations` (message_id, chunk_id, score)
- `agent_runs` (message_id, status, step_count), `tool_calls` (run_id, tool_name, input, output, approval_status)
- `reports` (org_id, user_id, agent_run_id nullable, title, content), `report_companies`

Tables are created only in the milestone that needs them.

---

## 6. Non-negotiable rules

### Multi-tenancy
- Every repository function that reads or writes private data takes `org_id` as a required argument and filters by it.
- Accessing another organization's resource returns 404 (not 403), so existence is not leaked.
- Every org-owned feature has a test proving organization A cannot see organization B's data.
- Roles: viewers are read-only on organization data; admin and analyst can create and change it (the `require_editor` dependency); managing users is admin-only.
- Agent tools enforce `org_id` the same way.

### External data
- SEC requests always send the `User-Agent` from `SEC_USER_AGENT`, and are throttled to at most 5 requests per second, with retries on transient errors.
- Save every raw downloaded file under `data/raw/` before parsing, so data can be re-processed without downloading again.
- Ingestion is idempotent: unique constraints plus "skip if exists" logic. Running a job twice must not create duplicates.
- Every scheduled job creates an `ingestion_runs` row with its status and any error.
- Start with 10-K and 10-Q filings and the `us-gaap` taxonomy only.

### Tests
- Tests never call real external APIs or real LLMs. Use saved fixture files and mocks. For LangChain code, use the fake chat model and deterministic fake embeddings from `langchain_core` instead of real providers.
- Tests run against a real Postgres test database, not SQLite.

### AI behaviour (Phases 2 and 3)
- Answers use only retrieved filing text. Every factual claim has a citation stored in `citations`.
- When the retrieved evidence is weak, the answer says it does not know.
- Read-only agent tools run automatically. Write tools (`create_alert`, `save_report`) set `approval_status = pending` and pause the run until the user approves.
- Every agent step is stored in `tool_calls`. Runs have a maximum step count and a timeout.
- Show a "not investment advice" notice in the UI wherever AI answers appear.

### LangChain (Phase 2)
LangChain provides the RAG building blocks, but the flow stays plain and visible inside our own service functions. A reader should see every step (rewrite question → retrieve → build prompt → call model → save citations) in order in the service code.

Use LangChain for:
- `Document` objects as the unit passed between parsing, chunking and retrieval.
- `RecursiveCharacterTextSplitter` for chunking, applied inside each section (never across section boundaries).
- The embeddings interface (`embed_documents`, `embed_query`), so changing the embedding provider is a config change.
- The standard chat model interface, `ChatPromptTemplate` for prompts, and simple LCEL pipes (`prompt | model`) called with `.invoke()` or `.stream()`.
- Callbacks for tracing, when added.

Do not use:
- Legacy chains (`RetrievalQA`, `ConversationalRetrievalChain` and similar) or anything from `langchain-classic`.
- LangChain vector store classes that run their own SQL or manage their own tables. Vector and full-text queries live in `repositories/chunks.py` using SQLAlchemy and pgvector, so the repository layer, foreign keys (`citations` → `document_chunks`) and our own table design stay intact.
- In-memory retrievers such as in-memory BM25. Keyword search uses Postgres full-text search.
- Custom `Runnable` subclasses or long multi-branch LCEL compositions. Logic between LangChain calls is plain Python in the service function.
- LangChain agents. The Phase 3 agent uses LangGraph.

Also:
- Check import paths against the installed LangChain version and avoid deprecated imports.
- Prompts are module-level constants in the service file that uses them (e.g. `QA_PROMPT` in `services/chat.py`).
- Chunk size, chunk overlap, top-k values and the relevance threshold come from `app/config.py`.

---

## 7. Milestones

Work on exactly one milestone at a time. Each milestone is finished only when its "done when" check passes as automated tests (or, for UI work, a described manual check).

### Phase 0: Foundation
- **0.1 Project skeleton**: folder structure, FastAPI app, config, logging, `PROJECT_CONTEXT.md` created from the template in section 9.
  Done when: `GET /health` returns OK.
- **0.2 Local infrastructure**: Docker Compose with Postgres (pgvector), Redis, API, Celery worker; SQLAlchemy + Alembic with a first migration.
  Done when: `docker compose up` starts everything and migrations run.
- **0.3 Quality setup**: pytest with a test database, ruff, GitHub Actions running lint and tests.
  Done when: CI passes.

### Phase 1: Traditional backend
- **1.1 Auth and multi-tenancy**: organizations, users, register/login with JWT, roles, `get_current_user`, audit log.
  Done when: tests prove cross-organization access returns 404.
- **1.2 Company catalog**: companies table, seed script from the SEC ticker file for a chosen list of 20–50 tickers, list/detail/search endpoints with pagination.
  Done when: searching "AAPL" returns Apple with its CIK.
- **1.3 Watchlists**: watchlist and item CRUD, organization-scoped, duplicate prevention.
  Done when: create/add/remove works and other organizations cannot see it.
- **1.4 SEC client and filing ingestion**: SEC client, raw storage, filings and ingestion_runs tables, Celery task plus beat schedule for daily filing checks.
  Done when: running the job twice creates no duplicates and filings appear on the company endpoint.
- **1.5 Financial facts**: financial_facts table and ingestion of selected us-gaap concepts (revenue, net income, assets, EPS, etc.).
  Done when: an endpoint returns a company's yearly revenue for the last 5 years.
- **1.6 Prices**: `PriceProvider` interface with one implementation, price_bars table, backfill script, daily end-of-day job.
  Done when: price history is returned, and swapping providers needs only a new class.
- **1.7 Alerts and notifications**: alert CRUD, evaluation job after the daily price job, notifications.
  Done when: a test with fixture prices triggers exactly one notification.
- **1.8 Hardening and frontend**: Redis caching for company and price reads, API rate limiting, consistent error responses, frontend pages (login, watchlists, company page with chart and filings, notifications).
  Done when: the full Phase 1 flow works in the browser.

### Phase 2: RAG
- **2.1 Filing parsing**: clean filing HTML from raw storage with BeautifulSoup and extract key 10-K sections (Risk Factors, MD&A first). Output one LangChain `Document` per section with metadata (company_id, ticker, filing_id, form_type, fiscal_year, section).
  Done when: clean section Documents are produced for 5 different companies (tests use saved fixture filings).
- **2.2 Chunking and embeddings**: document_chunks table with a pgvector column and a Postgres full-text search column; split each section Document with `RecursiveCharacterTextSplitter`; embed in batches through the LangChain embeddings interface; Celery task triggered after filing ingestion; idempotent (skip filings already chunked), plus a re-embed command for when the embedding model or chunk settings change. Record the embedding model and vector dimension in `PROJECT_CONTEXT.md`.
  Done when: every ingested 10-K has embedded chunks and running the task twice creates no duplicates.
- **2.3 Retrieval**: in `repositories/chunks.py`, a pgvector similarity query and a Postgres full-text query, both with metadata filters (ticker, year range, section). In `services/retrieval.py`: embed the query, run both searches, merge with reciprocal rank fusion, and return LangChain `Document`s with chunk id and score in metadata.
  Done when: tests with fixture chunks return the expected chunk for known queries, and a manual check on real data returns sensible sections for 5 sample questions.
- **2.4 Chat Q&A with citations**: chat_sessions, chat_messages and citations tables. The flow in `services/chat.py`, in this order:
  1. For a follow-up question, rewrite it into a standalone question using recent chat history (`prompt | model`).
  2. Retrieve chunks.
  3. If no chunk passes the relevance threshold, return a fixed "I don't know" answer without calling the LLM.
  4. Number the chunks in the prompt and instruct the model to answer only from them and cite as [1], [2].
  5. Stream the answer to the client with `.stream()`.
  6. After streaming, parse the citation markers, then save the message and citation rows (ignore markers that do not match a provided chunk).

  Plus a chat UI with streaming and clickable citations that open the source passage.
  Done when: an answer shows citations that open the exact source passage, a follow-up question works, and an unanswerable question gets the "I don't know" response.
- **2.5 Reranking and evaluation**: 30–50 hand-written evaluation questions with expected sections in `evals/`; an evaluation script measuring retrieval hit rate@k and MRR, and answer faithfulness with an LLM-as-judge prompt; a cross-encoder reranker applied after hybrid retrieval; before/after results saved in `evals/results.md`.
  Done when: a results table shows the effect of reranking and at least one chunk-size comparison.

### Phase 3: Agentic AI
- **3.1 Tool layer**: read-only tools wrapping existing services (`search_filings`, `get_financials`, `get_price_history`, `compute_metrics`, `compare_companies`), organization-scoped, with clear input schemas.
  Done when: each tool passes direct tests.
- **3.2 Agent graph and persistence**: LangGraph agent, agent_runs and tool_calls tables, step limit and timeout, routing between simple RAG and the agent.
  Done when: a multi-company comparison request completes with a stored trace of tool calls.
- **3.3 Human-in-the-loop approvals**: write tools pause the run as pending, approve/reject endpoint, run resumes after approval, audit logging, approval UI.
  Done when: the agent cannot create an alert without user approval.
- **3.4 Reports**: reports and report_companies tables, agent-generated cited reports, report list and view pages.
  Done when: a research request produces a saved, cited report linked to its companies.
- **3.5 Tracing and agent evaluation**: Langfuse tracing, 10–15 agent test tasks checking tool choice and groundedness.
  Done when: every run has a viewable trace and the agent eval script produces a results table.

---

## 8. Workflow for every milestone

1. **Read** `CLAUDE.md` and `PROJECT_CONTEXT.md`. Check the current state of the code before planning.
2. **Plan**: present a short plan with files to create or modify (with a reason for any new file), tables and migrations, endpoints, background jobs, new libraries (with reasons), and tests that prove the "done when" check. **Wait for approval before writing code.**
3. **Implement** following sections 3–6.
4. **Migrate**: if models changed, generate an Alembic migration, review it, and run it.
5. **Test**: write and run tests for the milestone. Run the full test suite and ruff. Fix everything before continuing.
6. **Update `PROJECT_CONTEXT.md`** as described in section 9. This step is mandatory; a milestone is not complete without it.
7. **Report**: summarize what was built, how to run or try it, any deviations from the plan, and suggest a commit message. Do not run `git commit` unless asked.
8. **Stop.** Do not start the next milestone until asked.

If something in this file conflicts with what the code needs, stop and ask instead of silently deviating.

---

## 9. PROJECT_CONTEXT.md

`PROJECT_CONTEXT.md` is the living record of the project's current state. Any new session must be able to continue the work by reading only `CLAUDE.md` and `PROJECT_CONTEXT.md`.

### Rules
- Create it in milestone 0.1 using the template below.
- Update it at the end of **every** milestone, before reporting completion.
- It must describe the code as it actually is, not as planned. If the code and this file disagree, fix the file.
- Keep it concise: rewrite sections to reflect the current state instead of appending endlessly. Only the "Milestone log" is append-only, with short entries.

### Template

```markdown
# PROJECT_CONTEXT.md

## Current status
- Current phase:
- Last completed milestone:
- Next milestone:

## How to run
- Start services:
- Run migrations:
- Run tests:
- Seed / backfill commands:

## Environment variables
| Name | Purpose |
|------|---------|

## Database tables
| Table | Purpose | Key columns / constraints | Added in |
|-------|---------|---------------------------|----------|

## API endpoints
| Method | Path | Purpose | Auth / role |
|--------|------|---------|-------------|

## Background jobs
| Task | Schedule | What it does |
|------|----------|--------------|

## Key files
- path — what it contains

## Design decisions
- Decision — reason (milestone)

## Known issues and tech debt
-

## Milestone log
- 0.1 (YYYY-MM-DD): one or two lines on what was built
```
