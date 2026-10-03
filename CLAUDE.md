# CLAUDE.md — Financial Research Copilot

This file tells you (Claude Code) what this project is, how the code must be written, and how to work through it milestone by milestone. Read it fully at the start of every session, together with `PROJECT_CONTEXT.md`.

---

## 1. What we are building

A multi-tenant web application for financial research on US-listed companies. It evolves in three phases:

1. **Traditional backend**: teams (organizations) track companies in watchlists, see filings, financial numbers and daily prices, and set price alerts. Data is ingested automatically by scheduled background jobs from SEC EDGAR and a price provider.
2. **RAG**: users ask questions about company filings (10-K, 10-Q) and get answers grounded only in filing text, with citations to the exact source passage.
3. **Agentic AI**: a research agent plans multi-step tasks (retrieve filings, pull financials and prices, compute metrics, compare companies, draft reports). Read-only actions run automatically; actions that write data require explicit user approval.

Each phase builds on the previous one. Never change a working earlier phase without a clear reason, and note any such change in `PROJECT_CONTEXT.md`.

**Phase 2 and 3 scope**: RAG and the agent work only on AAPL and NVDA (`RAG_TICKERS`) and only on filings filed in the last 2 years (`RAG_LOOKBACK_YEARS`), to keep embedding and LLM costs low. Phase 1 data and ingestion for all 34 companies are unchanged.

This is a learning and portfolio project. The developer must be able to read and explain every line, so clarity beats cleverness everywhere.

---

## 2. Tech stack

- Python 3.12
- FastAPI (sync route handlers, `def` not `async def`, unless streaming or a clear reason requires async)
- SQLAlchemy 2.0 (sync, 2.0-style `select()` queries) + Alembic
- PostgreSQL 16 with the pgvector extension
- Redis (cache, rate limiting, Celery broker)
- Celery + Celery beat (background and scheduled jobs)
- Pydantic v2 + pydantic-settings
- JWT auth (PyJWT) + bcrypt password hashing
- httpx for external HTTP calls
- LangChain 1.x for Phase 2 RAG: `langchain-core`, `langchain-text-splitters`, `langchain-community` (only `TextLoader` and `Html2TextTransformer`, with `html2text`, to load and convert filing HTML; the package is sunset and archived, so it is pinned and used for nothing else; it also installs `langchain-classic`, which is never imported), and `langchain-openai` (the OpenAI embeddings now, the OpenAI chat model from 2.4; it brings `openai`, which `app/rag/chunking.py` imports for its exception, and `tiktoken`, which the embedding script uses for token estimates). Section 6 defines how LangChain must be used.
- pgvector Python package for the vector column in SQLAlchemy models
- LangGraph for the agent (Phase 3)
- yfinance for daily prices (the first `PriceProvider` implementation; an unofficial Yahoo Finance wrapper, fine for a personal learning project, not for commercial use)
- Embeddings: OpenAI `text-embedding-3-small` (1536 dimensions) through LangChain's `OpenAIEmbeddings`. Chat model: OpenAI `gpt-5.4-mini` (used from 2.4). Reranker: Cohere through `langchain-cohere` (`CohereRerank`), default model `rerank-v4.0-fast` (chosen in 2.5, numbers in `evals/results.md`; `cohere` is also pinned because `app/rag/llm.py` builds the client with a timeout and `app/rag/retrieval.py` imports its `ApiError`)
- Langfuse for LLM tracing (Phase 3)
- pytest, ruff
- Docker Compose for local development
- Frontend: plain HTML, CSS and JavaScript (ES modules) served by FastAPI under /app; no framework, no build step, no external hosts.

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

### 3.7 JavaScript style
Frontend code follows the same rules as the Python code: plain, sequential code with short comments that say why. Functions exist only as event handlers or as section loaders (`loadX`, which fetch and render; re-render by calling the loader again). Shared code lives only in `frontend/common.js`; there is no state-management layer, no classes and no helper chains.

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
│   ├── main.py            # FastAPI app, router registration, exception handlers, /app static mount, GET / redirect to /app/
│   ├── config.py          # Settings from environment variables
│   ├── database.py        # Engine and session
│   ├── redis_client.py    # The only module that talks to Redis directly; cache and rate-limit helpers
│   ├── dependencies.py    # get_db, get_current_user, role checks
│   ├── exceptions.py      # Custom exceptions (NotFoundError, ConflictError, ...)
│   ├── security.py        # Password hashing and JWT create/decode
│   ├── models/            # SQLAlchemy models, one file per domain
│   ├── schemas/           # Pydantic request/response schemas, one file per domain
│   ├── repositories/      # Database queries only, one file per domain
│   ├── services/          # Business logic, one file per domain
│   ├── routes/            # FastAPI routers, one file per domain
│   ├── clients/           # External services: SEC, price provider
│   ├── rag/               # Phase 2 only: the RAG pipeline, one module per step (parsing, chunking, retrieval, chat, llm)
│   ├── workers/           # Celery app, tasks and beat schedule
│   └── agent/             # Phase 3 only: agent tools and LangGraph graph
├── scripts/               # One-off commands: seed companies, backfill prices
├── evals/                 # Phase 2/3 evaluation questions, scripts and results
├── frontend/              # Plain HTML, CSS, JS (ES modules), served at /app
│   ├── style.css          # The one stylesheet
│   ├── common.js          # Shared code: api(), apiStream(), token access, nav, pager, el(), showMessage()
│   └── <page>.html + <page>.js   # One pair per page: index, companies, company, watchlists, alerts, notifications, team, chat
├── data/raw/              # Raw downloaded SEC/price files (git-ignored)
└── tests/
```

Domains (use these names consistently across layers): `auth`, `organizations`, `users`, `companies`, `watchlists`, `filings`, `financials`, `prices`, `alerts`, `notifications`, `ingestion`, `audit`, `chunks`, `retrieval`, `chat`, `reports`, `agent`.

### Layer rules

- **Routes**: receive the request, call one service function, return a response schema. No database queries and no business logic in routes.
- **Services**: business logic and orchestration. Call repositories and clients. Own the transaction (`db.commit()`). Raise custom exceptions from `app/exceptions.py`, never `HTTPException`. `main.py` maps custom exceptions to HTTP responses.
- **Repositories**: SQLAlchemy queries only. No business rules, no commits, no external calls.
- **Clients**: all calls to external systems (SEC, price provider). Nothing else talks to the network, except the LangChain chat model and embeddings objects that `app/rag/llm.py` creates.
- **Workers**: Celery tasks are thin. They open a DB session, call a service function, and log the result.

RAG pipeline logic lives in `app/rag/`, one module per step:
- `parsing.py` (2.1): load filing HTML, convert to text, extract the 10-K sections
- `chunking.py` (2.2): splitting, embedding and storing chunks
- `retrieval.py` (2.3, 2.5): hybrid search, merging and reranking
- `chat.py` (2.4): the Q&A flow, citations and chat history
- `llm.py`: creates the LangChain chat model and embeddings objects from config (added when first needed)

These modules play the service role: they own commits and raise custom exceptions. Models stay in `app/models/` (Alembic), SQL stays in `app/repositories/` (including `repositories/chunks.py` and `repositories/chat.py`), and HTTP routes stay in `app/routes/`.

The agent (Phase 3) gets its own folder, `app/agent/`, because its tools and graph are a distinct layer.

---

## 5. Data model

Shared public data (no `org_id`; every organization reads the same rows):
- `companies` (ticker unique, cik unique as a 10-character zero-padded string, name, exchange, industry = the SEC's SIC description; there is no GICS sector)
- `filings` (company_id, accession_number unique, form_type, filed_on, report_date, fiscal_year, primary_document, raw_path relative to RAW_DATA_DIR)
- `financial_facts` (company_id, concept, unit, period_start nullable, period_end, value, fiscal_year = year of period_end, form_type, accession_number, filed_on; one row per period holding the value from the latest-filed 10-K; unique on company_id + concept + unit + period_start + period_end; annual data only)
- `price_bars` (company_id + trade_date as the composite primary key, open, high, low, close = split-adjusted, adj_close = split- and dividend-adjusted, volume)
- `document_chunks` (filing_id, company_id, section = a key of `SECTIONS`, fiscal_year nullable, chunk_index counted from 0 within each filing and section, content, embedding `Vector(1536)`, search_vector = a `tsvector` GENERATED ALWAYS from `to_tsvector('english', content)` STORED, embedding_model, created_at; unique on filing_id + section + chunk_index; GIN index on search_vector; no vector index at this size)
- `ingestion_runs` (job_type, status running/success/partial/failed, started_at, finished_at, message, error)

Private organization data (every table has `org_id`):
- `organizations`, `users` (org_id, email, role: admin / analyst / viewer)
- `watchlists` (org_id, created_by, name; name unique per organization, case-insensitive), `watchlist_items` (primary key is watchlist_id + company_id; deleting a watchlist cascades to its items)
- `alerts` (org_id, user_id = owner, company_id, alert_type price_above/price_below/daily_change_pct, threshold, active, watch_from = first day the alert may fire), `notifications` (org_id, user_id, alert_id with ON DELETE CASCADE, company_id, trade_date, trigger_value, message, is_read, created_at; unique on alert_id + trade_date)
- `audit_logs` (org_id, user_id, action, entity_id)
- `chat_sessions` (org_id, user_id = owner, title nullable = the first question cut to 100 characters, created_at, updated_at), `chat_messages` (session_id with ON DELETE CASCADE, role user/assistant, content, rewritten_question nullable, ticker nullable = the filter used, model nullable, created_at; no `org_id`: read only through the owner's session, like `watchlist_items`), `citations` (message_id with ON DELETE CASCADE, number = the [n] in the answer, chunk_id NULLABLE with ON DELETE SET NULL, score = the vector similarity, and a SNAPSHOT of what was cited: filing_id, ticker, fiscal_year, section, content; unique on message_id + number). `--reembed` replaces chunk rows, so a citation must not depend on its chunk row: after a re-embed `chunk_id` is null and the snapshot still shows the exact passage. Chats are PERSONAL, like alerts
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
- Alerts and notifications are personal: besides org_id, every query also filters by the owning user_id, and another user's alert or notification returns 404 even inside the same organization. Any role, including viewer, can manage their own alerts. The evaluation job's list_active is the only org-less alert query.
- Chat sessions are personal like alerts: every session query filters by org_id AND user_id, and a colleague's or another organization's session returns the same 404 "Chat session not found". Every role, including viewer, can chat (a chat changes no shared organization data). Messages and citations are read only through a session loaded this way.
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

### API conventions
- Every error response is `{"detail": "<string>"}`. A 429 also sends a `Retry-After` header. Unhandled errors return a generic 500 and never expose the exception text.
- Only shared public data (companies, filings, financial facts, prices) may be cached, with a TTL only. Data with an `org_id` or `user_id` is never cached. Cache failures never fail a request.
- Rate limits are enforced with Redis dependencies: `limit_auth` for login and register, `limit_user` for every protected route. A new router must be included with `dependencies=[Depends(limit_user)]`.
- Unique-constraint violations that slip past the service checks become 409.

### Frontend rules
- (a) Text is put on the page only through `textContent` (and `createElement`, `append`, `setAttribute` for non-style attributes). `innerHTML`, `outerHTML`, `insertAdjacentHTML`, `document.write`, `eval` and `new Function` are forbidden; `tests/test_frontend.py` enforces it.
- (b) Every HTML page has the Content-Security-Policy meta tag as the first element in `<head>`, and no inline script, no `style` attributes and no inline event handler attributes.
- (c) `fetch` and the login token (`localStorage`) are used only in `frontend/common.js`, which has exactly nine exports; the ninth, `apiStream`, is the only `fetch` besides `api()` (it reads the NDJSON chat stream).
- (d) Hiding controls by role is UX only; the server enforces permissions and its error messages are shown as they are.
- (e) Never construct a `Date` from a date-only string (like `2024-06-07`); show such values as received.
- (f) No polling, no timers, no auto-refresh.
- (g) Cached shared data may be up to `CACHE_TTL_SECONDS` old, and the UI does not hide that.

### AI behaviour (Phases 2 and 3)
- Answers use only retrieved filing text. Every factual claim has a citation stored in `citations`.
- When the retrieved evidence is weak, the answer says it does not know. "Weak" is decided with `vector_similarity` (never the RRF score) against `RELEVANCE_THRESHOLD` (0.30); chunks below it are neither numbered nor sent to the model.
- Read-only agent tools run automatically. Write tools (`create_alert`, `save_report`) set `approval_status = pending` and pause the run until the user approves.
- Every agent step is stored in `tool_calls`. Runs have a maximum step count and a timeout.
- Reranking (2.5) fails open: when the Cohere call raises a Cohere API error or an `httpx` error, `retrieve` logs a warning (class name only) and returns the fused order, so chat keeps working when Cohere is down (the same reasoning as the Redis fail-open). Reranking is on when `RERANK_ENABLED` is true and `COHERE_API_KEY` is set.
- The "I don't know" threshold stays on `vector_similarity`, never on the rerank score: the rerank score has another scale and changes between model versions.
- Show a "not investment advice" notice in the UI wherever AI answers appear.

### LangChain (Phase 2)
Prefer LangChain components (loaders, document transformers, text splitters, embeddings, chat models, prompts) over custom code; write plain Python only where no component fits, and say so in the plan. The flow itself stays plain and visible inside our own functions in `app/rag/`: a reader should see every step (rewrite question → retrieve → build prompt → call model → save citations) in order in the code.

Use LangChain for:
- `Document` objects as the unit passed between parsing, chunking and retrieval.
- `TextLoader` and `Html2TextTransformer` (from `langchain-community`) to load a stored filing and convert its HTML to text. The only custom step in parsing is the regex that cuts out the 10-K items, because no LangChain component knows them.
- `RecursiveCharacterTextSplitter` for chunking, applied inside each section (never across section boundaries).
- The embeddings interface (`embed_documents`, `embed_query`), so changing the embedding provider is a config change.
- The standard chat model interface, `ChatPromptTemplate` for prompts, and simple LCEL pipes (`prompt | model`) called with `.invoke()` or `.stream()`.
- Callbacks for tracing, when added.

Do not use:
- Legacy chains (`RetrievalQA`, `ConversationalRetrievalChain` and similar) or anything from `langchain-classic`.
- LangChain vector store classes that run their own SQL or manage their own tables. Vector and full-text queries live in `repositories/chunks.py` using SQLAlchemy and pgvector, so the repository layer, foreign keys (`citations` → `document_chunks`) and our own table design stay intact.
- In-memory retrievers such as in-memory BM25. Keyword search uses Postgres full-text search.
- `EnsembleRetriever` and custom `BaseRetriever` subclasses: hybrid fusion is plain Python (reciprocal rank fusion in `app/rag/retrieval.py`) because `EnsembleRetriever` lives in `langchain-classic`.
- Custom `Runnable` subclasses or long multi-branch LCEL compositions. Logic between LangChain calls is plain Python in the service function.
- LangChain agents. The Phase 3 agent uses LangGraph.

Also:
- Check import paths against the installed LangChain version and avoid deprecated imports.
- Prompts are module-level constants in the `app/rag/` module that uses them (e.g. `QA_PROMPT` in `app/rag/chat.py`).
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
- **1.5 Financial facts**: financial_facts table and ingestion of ANNUAL (10-K) us-gaap values for a fixed allowlist of concepts (revenue, net income, assets, EPS, etc.).
  Done when: an endpoint returns a company's yearly revenue for the last 5 years.
- **1.6 Prices**: `PriceProvider` interface with one implementation (yfinance), price_bars table, a daily job that re-syncs the full lookback window (the first run is the backfill).
  Done when: price history is returned, and swapping providers needs only a new class.
- **1.7 Alerts and notifications**: alert CRUD (personal alerts), an evaluation job chained after the daily price job, in-app notifications.
  Done when: a test with fixture prices triggers exactly one notification.
- **1.8 Backend hardening**: consistent error responses, Redis rate limiting, Redis caching of shared reads.
  Done when: the limits, the cache and the error format are proven by tests and by the manual checks.
- **1.9 Frontend**: login/registration, companies, company page (SVG price chart, financials table, filings, add to watchlist, create alert), watchlists, alerts, notifications, team.
  Done when: tests pass and the browser checklist passes.

### Phase 2: RAG
- **2.1 Filing parsing** (`app/rag/parsing.py`): load a stored 10-K from raw storage with LangChain `TextLoader`, convert the HTML to text with `Html2TextTransformer`, and cut out Item 1A Risk Factors and Item 7 MD&A with a regex step. Output one LangChain `Document` per section found with metadata (company_id, ticker, filing_id, form_type, fiscal_year, section). 10-K only; scope is `RAG_TICKERS` and `RAG_LOOKBACK_YEARS`.
  Done when: both sections are found in every in-scope 10-K of AAPL and NVDA (4 filings on 2026-10-04) and tests with saved fixture filings pass.
- **2.2 Chunking and embeddings**: document_chunks table with a pgvector column and a Postgres full-text search column; split each section Document with `RecursiveCharacterTextSplitter`; embed in batches through the LangChain embeddings interface; Celery task chained after the `ingest_filings` task (no schedule of its own); idempotent (skip filings already chunked, with no API call), plus a replace mode (`--reembed`, one transaction per filing) for when the embedding model or chunk settings change. Record the embedding model and vector dimension in `PROJECT_CONTEXT.md`.
  Done when: every in-scope 10-K has embedded chunks and running the task twice creates no duplicates.
  Done when: every in-scope 10-K has embedded chunks and running the task twice creates no duplicates.
- **2.3 Retrieval**: in `repositories/chunks.py`, a pgvector cosine-distance query and a Postgres full-text query (`ts_rank_cd`), both with metadata filters (tickers, year range, sections). The full-text query is an OR query: the question's `\w+` words joined with `or` and passed to `websearch_to_tsquery('english', ...)`, because an AND query on a natural sentence matches nothing. In `app/rag/retrieval.py`, `retrieve(db, question, tickers, year_from, year_to, sections, top_k)`: validate, embed the query once, run both searches (`RETRIEVAL_CANDIDATES_K` each), merge with reciprocal rank fusion (`score = sum(1 / (RRF_K + rank))`, ties by chunk id), keep `RETRIEVAL_TOP_K`, and return LangChain `Document`s whose metadata holds `chunk_id, filing_id, company_id, ticker, fiscal_year, section, chunk_index, score, vector_similarity, vector_rank, text_rank`. `vector_similarity` (1 minus cosine distance, filled for every returned chunk) is the signal 2.4 compares with its relevance threshold; the RRF `score` only orders chunks. `python -m scripts.try_retrieval "question"` (or `--samples`) is the manual check.
  Done when: tests with fixture chunks return the expected chunk for known queries, and a manual check on real data returns sensible sections for 5 sample questions.
- **2.4 Chat Q&A with citations**: chat_sessions, chat_messages and citations tables (section 5). The flow in `app/rag/chat.py`, in this order:
  1. Load the session with org_id AND user_id (404 otherwise) and the model (503 when `OPENAI_API_KEY` is empty); both BEFORE the response starts.
  2. For a follow-up question (the session has history), rewrite it into a standalone question using the last `CHAT_HISTORY_MESSAGES` messages (`prompt | model`). No history, no rewrite and no LLM call.
  3. Retrieve chunks with `retrieve(..., filing_ids=list_scope_filing_ids(db))`, so chat only sees the exact RAG scope (an empty scope answers "I don't know" without searching: an empty `filing_ids` list means "no filter").
  4. Keep the chunks with `vector_similarity >= RELEVANCE_THRESHOLD`. If none passes, the answer is the fixed `NO_ANSWER` text without calling the answer LLM. (Steps 3 and 4, and the numbering of step 5, live in `chat.prepare_context` since 2.5, shared with the evaluation script; `retrieve` reranks by itself when reranking is on.)
  5. Number the passing chunks in the prompt and instruct the model to answer only from them and to cite each sentence as [1], [2].
  6. Stream the answer with `.stream()` as NDJSON (`application/x-ndjson`, one JSON object per line): `{"type": "sources", "sources": [{number, chunk_id, ticker, fiscal_year, section, score, content}]}` first (empty for "I don't know"), then `{"type": "token", "text"}` pieces, then `{"type": "done", "message_id", "cited_numbers"}`. A failure after the stream started sends `{"type": "error", "detail"}` instead (generic text; the class name only is logged) and nothing is saved.
  7. After streaming, parse the markers [n] and [1, 2], ignore numbers outside 1..n, and save in ONE transaction the user message, the assistant message and one citation row per distinct valid number (with the snapshot of the chunk).

  Plus a chat UI with streaming and clickable citations that open the source passage.
  Done when: an answer shows citations that open the exact source passage, a follow-up question works, and an unanswerable question gets the "I don't know" response.
- **2.5 Reranking and evaluation**: Cohere reranking after hybrid retrieval (`retrieve(..., rerank=None)`: the best `RERANK_CANDIDATES_K` fused chunks go to `llm.get_reranker(top_n=top_k).compress_documents`, each Document gets `metadata["rerank_score"]`, `None` when reranking did not run; fails open) and an evaluation in `evals/`.
  - Labels are PHRASE based: `evals/questions.json` is a list of `{id, question, ticker (or null), answerable, expected}` with `expected` a list of `{ticker, section, phrases}`. A retrieved chunk is relevant when its ticker and section match an entry and its text contains one of that entry's phrases (case-insensitive, whitespace collapsed). Chunk ids are not used because they change on every re-embed and chunk-size change.
  - The set has 40 questions: 28 answerable single-company (AAPL and NVDA, both sections and both fiscal years, some in everyday words), 5 cross-company (ticker null, entries for both companies) and 7 unanswerable (at least 4 on-topic).
  - Metrics: hit@1/3/5 and MRR@5 over the answerable questions (computed on the top 5 returned by `retrieve`, before the threshold); unanswerable rejected (best `vector_similarity` below `RELEVANCE_THRESHOLD`, or the answer abstains) and answerable wrongly rejected (below the threshold); with `--answers` faithfulness (supported claims / claims, per answer, then averaged), citation coverage and abstention, judged by `JUDGE_PROMPT` with `get_chat_model().with_structured_output(...)`. Every question goes through `chat.prepare_context`, the same retrieval and threshold as chat.
  - Command: `python -m evals.run_eval --name <run name> [--no-rerank] [--rerank-model rerank-v4.0-pro] [--answers] [--limit N] [--rerank-pause SECONDS] [--validate-only]`. It validates the question file (including that every phrase exists in the stored chunks) before any paid call and writes `evals/runs/<name>.json`.
  - Run names: `baseline_1500` (`--no-rerank --answers`), `rerank_fast_1500` (`--answers`), `rerank_pro_1500`, `baseline_800` and `rerank_fast_800` (the 800 runs after `CHUNK_SIZE=800 CHUNK_OVERLAP=100 python -m scripts.run_embedding --reembed`, then re-embedded back).
  Done when: `evals/results.md` has a table with these runs, shows the effect of reranking on concrete questions and a chunk-size comparison, and the defaults chosen from the numbers are set.

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
