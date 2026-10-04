import json
import os
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import openai
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

# --- Test database safety ---
# This block must run before any "app" module is imported, because app.config reads
# DATABASE_URL when it is first imported. Replacing it here means the app, Alembic and the
# tests can never touch the development database.
test_database_url = os.environ.get("TEST_DATABASE_URL")
if not test_database_url:
    pytest.exit("TEST_DATABASE_URL is not set. Add it to .env (see .env.example).", returncode=2)

test_database_name = make_url(test_database_url).database
if not test_database_name or not test_database_name.endswith("_test"):
    pytest.exit(
        f"Refusing to run: TEST_DATABASE_URL points at database '{test_database_name}', "
        "but the name must end with '_test' so the development database is never touched.",
        returncode=2,
    )

os.environ["DATABASE_URL"] = test_database_url

# Same idea for Redis: the tests flush their Redis database before every test, so it must never
# be the development one (database 0)
test_redis_url = os.environ.get("TEST_REDIS_URL")
if not test_redis_url:
    pytest.exit("TEST_REDIS_URL is not set. Add it to .env (see .env.example).", returncode=2)

test_redis_database = urlparse(test_redis_url).path.lstrip("/") or "0"
if not test_redis_database.isdigit() or int(test_redis_database) == 0:
    pytest.exit(
        f"Refusing to run: TEST_REDIS_URL uses Redis database '{test_redis_database}', but it "
        "must be a NON-ZERO database number (0 is the development one and would be flushed).",
        returncode=2,
    )

os.environ["REDIS_URL"] = test_redis_url

# These imports must come after the environment override above
from alembic.config import Config
from fastapi.testclient import TestClient

from alembic import command
from app.agent import run as agent_run
from app.agent import tools as agent_tools
from app.clients import sec
from app.config import settings
from app.database import engine
from app.dependencies import get_db
from app.main import app
from app.models.chunks import DocumentChunk
from app.models.companies import Company
from app.rag import llm
from app.redis_client import redis_client
from app.repositories import chunks as chunk_repository
from app.repositories import companies as company_repository
from app.repositories import filings as filing_repository
from app.repositories import financials as financial_repository
from app.repositories import prices as price_repository
from app.schemas.financials import MetricName
from app.services import companies as company_service
from app.services.financials import METRICS


@pytest.fixture(scope="session", autouse=True)
def test_database() -> None:
    # Connect to the server's default "postgres" database. CREATE DATABASE cannot run inside
    # a transaction, so this connection uses autocommit.
    server_url = make_url(test_database_url).set(database="postgres")
    server_engine = create_engine(server_url, isolation_level="AUTOCOMMIT")
    with server_engine.connect() as connection:
        exists = connection.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": test_database_name},
        ).scalar()
        # The database is reused between runs, so only create it the first time
        if not exists:
            connection.execute(text(f'CREATE DATABASE "{test_database_name}"'))
    server_engine.dispose()

    # Bring the schema up to date. alembic/env.py reads DATABASE_URL, which now points at the
    # test database. This is a no-op when it is already at head.
    command.upgrade(Config("alembic.ini"), "head")


@pytest.fixture(autouse=True)
def clean_redis() -> None:
    # Rate-limit counters and cached responses must never leak from one test into the next
    redis_client.flushdb()


@pytest.fixture(autouse=True)
def high_rate_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    # The rest of the suite must never hit a limit. Rate-limit tests set small values themselves.
    monkeypatch.setattr(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 1_000_000)
    monkeypatch.setattr(settings, "RATE_LIMIT_API_PER_MINUTE", 1_000_000)


class SpyEmbeddings(DeterministicFakeEmbedding):
    # Deterministic fake vectors that also record every embed_documents call (the number of texts
    # in each) and can fail on cue, so tests can count API calls and simulate an OpenAI error
    calls: list[int] = Field(default_factory=list)
    fail_when_text_contains: str | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        if self.fail_when_text_contains and any(self.fail_when_text_contains in t for t in texts):
            request = httpx.Request("POST", "https://example.invalid/embeddings")
            raise openai.APIConnectionError(request=request)
        return super().embed_documents(texts)


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> SpyEmbeddings:
    # No test may ever reach OpenAI: every test gets fake 1536-dimension embeddings. Tests can
    # ask for this fixture to inspect the calls or to change its size
    fake = SpyEmbeddings(size=1536)
    monkeypatch.setattr(llm, "get_embeddings", lambda: fake)
    return fake


class ScriptedChatModel(GenericFakeChatModel):
    # A fake chat model that answers with the scripted messages, one per call, in order (an extra
    # call raises StopIteration). It records the messages of every call in `received`, can fail
    # in the middle of a stream like an OpenAI error, and with forbidden=True it fails the test
    # as soon as it is called.
    # Since 3.2 it also works for the agent: scripted AIMessages may have tool_calls (the stock
    # fake DROPS tool_calls when it streams, and LangGraph's messages mode makes a model stream
    # even inside .invoke()), bind_tools records the tool names, and with_structured_output (the
    # router) pops the next item of `structured`
    received: list[list] = Field(default_factory=list)
    forbidden: bool = False
    fail_after_pieces: int | None = None
    fail_on_call: bool = False
    structured: list = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)
    last_message: Any = None

    def _check_callable(self) -> None:
        if self.forbidden:
            raise AssertionError("The chat model was called, but this test expects no call")
        if self.fail_on_call:
            request = httpx.Request("POST", "https://example.invalid/chat/completions")
            raise openai.APIConnectionError(request=request)

    def _generate(self, messages, *args, **kwargs):
        self._check_callable()
        self.received.append(messages)
        result = super()._generate(messages, *args, **kwargs)
        self.last_message = result.generations[0].message
        return result

    def _stream(self, *args, **kwargs):
        count = 0
        for index, chunk in enumerate(super()._stream(*args, **kwargs)):
            if self.fail_after_pieces is not None and index >= self.fail_after_pieces:
                request = httpx.Request("POST", "https://example.invalid/chat/completions")
                raise openai.APIConnectionError(request=request)
            count += 1
            yield chunk
        # The tool calls of the scripted message, as the tool_call_chunks a real model streams
        for index, call in enumerate(self.last_message.tool_calls):
            args_json = json.dumps(call["args"])
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    id=self.last_message.id,
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": args_json,
                            "id": call["id"],
                            "index": index,
                            "type": "tool_call_chunk",
                        }
                    ],
                )
            )

    def bind_tools(self, tools, **kwargs):
        # Shares this object (and so `received`) with the code that uses the unbound model
        self.bound_tools = [tool.name for tool in tools]
        return self

    def with_structured_output(self, schema, **kwargs):
        def decide(prompt_value):
            self._check_callable()
            self.received.append(prompt_value.to_messages())
            return self.structured.pop(0)

        return RunnableLambda(decide)

    def prompt_text(self, call_index: int) -> str:
        # All messages of one call as one string, for assertions on what the model was given
        return "\n".join(str(message.content) for message in self.received[call_index])


def tool_calls_message(*calls: tuple[str, dict]) -> AIMessage:
    # A scripted model answer that asks for tools: tool_calls_message(("get_price_history", {...}))
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{index}_{name}", "type": "tool_call"}
            for index, (name, args) in enumerate(calls)
        ],
    )


@pytest.fixture(autouse=True)
def fake_chat_model(monkeypatch: pytest.MonkeyPatch) -> ScriptedChatModel:
    # No test may ever reach OpenAI: by default the chat model FAILS the test when it is called.
    # Tests that expect answers install scripted ones with script_chat
    model = ScriptedChatModel(messages=iter([]), forbidden=True)
    monkeypatch.setattr(llm, "get_chat_model", lambda: model)
    return model


@pytest.fixture(autouse=True)
def no_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    # No test may ever reach Cohere, and the developer's real COHERE_API_KEY (from .env) must not
    # switch reranking on in unrelated tests: the key is emptied and get_reranker FAILS the test
    # when it is called. Rerank tests install their own fake reranker (tests/test_rag_rerank.py)
    def forbidden_reranker(top_n: int) -> None:
        raise AssertionError("The reranker was called, but this test expects no rerank")

    monkeypatch.setattr(settings, "COHERE_API_KEY", "")
    monkeypatch.setattr(llm, "get_reranker", forbidden_reranker)


@pytest.fixture
def script_chat(monkeypatch: pytest.MonkeyPatch) -> Callable[..., ScriptedChatModel]:
    # script_chat("answer 1", "answer 2") installs a model that gives these texts to the first and
    # second call. An item can also be an AIMessage (with tool_calls, see tool_calls_message).
    # fail_after_pieces=N makes a stream fail after N pieces; structured=[...] are the answers of
    # with_structured_output (the router)
    def install(
        *items: str | AIMessage,
        fail_after_pieces: int | None = None,
        structured: list | None = None,
    ) -> ScriptedChatModel:
        model = ScriptedChatModel(
            messages=iter(
                [AIMessage(content=item) if isinstance(item, str) else item for item in items]
            ),
            fail_after_pieces=fail_after_pieces,
            structured=list(structured or []),
        )
        monkeypatch.setattr(llm, "get_chat_model", lambda: model)
        return model

    return install


@pytest.fixture
def db() -> Generator[Session, None, None]:
    # Open one connection and start an outer transaction that is never committed
    connection = engine.connect()
    outer_transaction = connection.begin()

    # "create_savepoint" turns each db.commit() in our services into a savepoint release
    # instead of a real commit, so services can commit as in production
    session = Session(bind=connection, join_transaction_mode="create_savepoint")

    yield session

    # Rolling back the outer transaction discards everything the test wrote,
    # so no data is ever persisted between tests
    session.close()
    outer_transaction.rollback()
    connection.close()


@pytest.fixture
def client(db: Session) -> Generator[TestClient, None, None]:
    # Every request in the test uses the same session as the test itself,
    # so the test sees what the endpoint wrote (and everything is rolled back at the end)
    def override_get_db() -> Generator[Session, None, None]:
        yield db

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def register_org(client: TestClient) -> Callable[[str, str], dict[str, str]]:
    # Registers a new organization through the API and returns the admin's auth headers,
    # so tests can set up several organizations in one line each
    def register(organization_name: str, email: str) -> dict[str, str]:
        response = client.post(
            "/auth/register",
            json={
                "organization_name": organization_name,
                "email": email,
                "password": "correct-horse-battery",
            },
        )
        assert response.status_code == 201
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    return register


# A small file in the same structure as the real SEC company_tickers_exchange.json
SEC_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "company_tickers_exchange.json"


@pytest.fixture
def sec_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    # The parsed fixture, produced by the real client code with the download mocked,
    # so no test ever calls the SEC
    def fake_get(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(
            200, content=SEC_FIXTURE_PATH.read_bytes(), request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(tmp_path))
    return sec.get_company_tickers()


@pytest.fixture
def seeded(db: Session, monkeypatch: pytest.MonkeyPatch, sec_rows: list[dict]) -> None:
    # Seeds the catalog through the service with the SEC client mocked. These are the tickers
    # from the fixture file that tests use.
    monkeypatch.setattr(sec, "get_company_tickers", lambda: sec_rows)
    company_service.seed_companies(db, ["AAPL", "MSFT", "MA", "V", "AMZN", "GOOGL", "JNJ"])


@pytest.fixture
def people(
    client: TestClient, seeded: None, register_org: Callable[[str, str], dict[str, str]]
) -> dict[str, dict]:
    # The users the alert and notification tests need, each as {"headers", "org_id", "user_id"}:
    #   admin, analyst, viewer: three roles of organization "Acme"
    #   colleague: a second analyst of Acme (same organization, a different person)
    #   outsider: the admin of another organization, "Globex"
    password = "correct-horse-battery"
    admin_headers = register_org("Acme", "admin@acme.com")
    outsider_headers = register_org("Globex", "admin@globex.com")
    headers_by_name = {"admin": admin_headers, "outsider": outsider_headers}

    for name, role in [("analyst", "analyst"), ("viewer", "viewer"), ("colleague", "analyst")]:
        email = f"{name}@acme.com"
        created = client.post(
            "/users",
            json={"email": email, "password": password, "role": role},
            headers=admin_headers,
        )
        assert created.status_code == 201
        login = client.post("/auth/login", json={"email": email, "password": password})
        headers_by_name[name] = {"Authorization": f"Bearer {login.json()['access_token']}"}

    result = {}
    for name, headers in headers_by_name.items():
        me = client.get("/auth/me", headers=headers).json()
        result[name] = {"headers": headers, "org_id": me["org_id"], "user_id": me["id"]}
    return result


# Chunks for the chat tests: 3 chunks in the RAG scope (10-Ks filed within RAG_LOOKBACK_YEARS,
# downloaded) and 1 chunk of an OLD 10-K that is outside it. The text is hand-written; the fake
# embeddings give the same vector for the same text, so a question equal to a chunk's content has
# similarity 1 with it, and about 0 with the others (below RELEVANCE_THRESHOLD)
CHAT_CHUNK_DATA = {
    "nvda_export": (
        "NVDA",
        "risk_factors",
        "Export controls restrict sales of data center products to China.",
    ),
    "nvda_mdna": (
        "NVDA",
        "mdna",
        "Data Center revenue grew on demand for accelerated computing.",
    ),
    "aapl_risk": (
        "AAPL",
        "risk_factors",
        "Apple depends on outsourcing partners in Asia for the manufacture of its products.",
    ),
    "aapl_old": (
        "AAPL",
        "risk_factors",
        "Legacy product transitions in an old filing created inventory risk.",
    ),
}


@pytest.fixture
def chat_chunks(db: Session) -> dict[str, DocumentChunk]:
    today = date.today()
    companies = {}
    for ticker, cik in (("AAPL", "0000320193"), ("NVDA", "0001045810")):
        # The seeded fixture may have created the company already
        company = company_repository.get_by_ticker(db, ticker)
        companies[ticker] = company or company_repository.create(
            db, ticker, cik, f"{ticker} Inc.", None
        )

    # (ticker, filed_on) -> a downloaded 10-K. The AAPL 10-K filed 900 days ago is out of scope
    filings = {}
    for ticker, days_ago, fiscal_year in (
        ("AAPL", 60, today.year - 1),
        ("NVDA", 30, today.year - 1),
    ):
        filing = filing_repository.create(
            db,
            companies[ticker].id,
            f"{ticker}-in-scope",
            "10-K",
            today - timedelta(days=days_ago),
            report_date=today - timedelta(days=days_ago + 30),
            fiscal_year=fiscal_year,
            primary_document="document.htm",
        )
        filing.raw_path = "document.htm"
        filings[(ticker, "in")] = filing
    old = filing_repository.create(
        db,
        companies["AAPL"].id,
        "AAPL-old",
        "10-K",
        today - timedelta(days=900),
        report_date=today - timedelta(days=930),
        fiscal_year=today.year - 3,
        primary_document="document.htm",
    )
    old.raw_path = "document.htm"
    filings[("AAPL", "old")] = old
    db.flush()

    vectors = llm.get_embeddings().embed_documents([data[2] for data in CHAT_CHUNK_DATA.values()])
    rows = {}
    for (name, (ticker, section, content)), vector in zip(
        CHAT_CHUNK_DATA.items(), vectors, strict=True
    ):
        filing = filings[(ticker, "old" if name == "aapl_old" else "in")]
        rows[name] = DocumentChunk(
            filing_id=filing.id,
            company_id=companies[ticker].id,
            section=section,
            fiscal_year=filing.fiscal_year,
            chunk_index=0,
            content=content,
            embedding=vector,
            embedding_model="fake",
        )
    chunk_repository.create_many(db, list(rows.values()))
    return rows


# --- The 3.1 / 3.2 market data (moved here from test_agent_tools.py, used by the agent tests) ---
@pytest.fixture
def companies(db: Session) -> dict[str, Company]:
    result = {}
    for ticker, cik in (
        ("AAPL", "0000320193"),
        ("NVDA", "0001045810"),
        ("MSFT", "0000789019"),
    ):
        # chat_chunks may have created AAPL and NVDA already
        result[ticker] = company_repository.get_by_ticker(db, ticker) or company_repository.create(
            db, ticker, cik, f"{ticker} Inc.", None
        )
    return result


def add_facts(
    db: Session, company: Company, metric: str, values: dict[int, float], month: int, day: int
) -> None:
    # One annual fact per fiscal year; the period ends on month/day of that year
    _label, unit, concepts = METRICS[MetricName(metric)]
    for fiscal_year, value in values.items():
        period_end = date(fiscal_year, month, day)
        financial_repository.create(
            db,
            company.id,
            concepts[0],
            unit,
            None,
            period_end,
            Decimal(str(value)),
            fiscal_year,
            "10-K",
            f"{company.ticker}-{fiscal_year}",
            period_end,
        )


def add_bars(db: Session, company: Company, closes: list[float]) -> None:
    # Consecutive calendar days, the last close yesterday; volume is 1000 + the position
    for position, close in enumerate(closes):
        price = Decimal(str(close))
        price_repository.create(
            db,
            company.id,
            date.today() - timedelta(days=len(closes) - position),
            price,
            price,
            price,
            price,
            price,
            1000 + position,
        )


@pytest.fixture
def market(db: Session, companies: dict[str, Company]) -> dict[str, Company]:
    # Hand-picked numbers (so every derived value can be computed by hand). AAPL: fiscal years
    # 2022 to 2024 ending late September; NVDA: 2023 to 2025 ending late January, with NEGATIVE
    # equity in 2024 and no gross profit at all
    aapl = companies["AAPL"]
    nvda = companies["NVDA"]
    aapl_values = {
        "revenue": {2022: 100, 2023: 120, 2024: 150},
        "net_income": {2022: 20, 2023: 30, 2024: 45},
        "operating_income": {2022: 25, 2023: 36, 2024: 60},
        "gross_profit": {2022: 40, 2023: 60, 2024: 75},
        "shareholders_equity": {2022: 200, 2023: 150, 2024: 300},
        "total_liabilities": {2022: 100, 2023: 150, 2024: 600},
    }
    for metric, values in aapl_values.items():
        add_facts(db, aapl, metric, values, 9, 28)
    nvda_values = {
        "revenue": {2023: 50, 2024: 100, 2025: 300},
        "net_income": {2023: 10, 2024: 50, 2025: 150},
        "shareholders_equity": {2023: 20, 2024: -10, 2025: 100},
        "total_liabilities": {2023: 40, 2024: 60, 2025: 100},
    }
    for metric, values in nvda_values.items():
        add_facts(db, nvda, metric, values, 1, 26)
    return companies


@pytest.fixture
def agent_environment(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    # The agent opens its own short sessions (the tools and the run). Give them all the test
    # session and never close it, so the rollback isolation is kept. ToolNode runs tool calls in
    # threads and a Session is not thread-safe: the lock makes the tool bodies run one at a time.
    # The checkpointer is an in-memory saver. A (fake) key enables the tools and the agent
    lock = threading.RLock()

    @contextmanager
    def shared_session():
        with lock:
            yield db

    monkeypatch.setattr(agent_tools, "SessionLocal", shared_session)
    monkeypatch.setattr(agent_run, "SessionLocal", shared_session)

    @contextmanager
    def in_memory_checkpointer():
        yield InMemorySaver()

    monkeypatch.setattr(agent_run, "open_checkpointer", in_memory_checkpointer)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test-not-real")
