import logging
import socket
from collections.abc import Callable
from datetime import date, timedelta

import httpx
import pytest
from cohere.errors import TooManyRequestsError, UnauthorizedError
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from sqlalchemy.orm import Session

from app.config import settings
from app.models.chat import ChatSession
from app.models.chunks import DocumentChunk
from app.models.users import User
from app.rag import chat, llm, retrieval
from app.rag.llm import get_reranker as real_get_reranker  # the real one, not the test fake
from app.repositories import chunks as chunk_repository
from app.repositories import companies as company_repository
from app.repositories import filings as filing_repository
from app.repositories import organizations as organization_repository
from app.repositories import users as user_repository
from tests.conftest import ScriptedChatModel

TODAY = date.today()
QUESTION = "Chunk number 3 describes supplier risk."
EXPORT_QUESTION = "Export controls restrict sales of data center products to China."


class FakeReranker:
    # Stands in for CohereRerank. It records what it received and answers in a scripted order:
    # `order` lists the positions (in the list it received) to return, best first; the default
    # reverses the input. Scores go down from 0.9 in steps of 0.1. With `error` it raises instead
    def __init__(self, order: list[int] | None = None, error: Exception | None = None) -> None:
        self.order = order
        self.error = error
        self.top_n: int | None = None
        self.received: list[list[Document]] = []
        self.queries: list[str] = []
        self.created = 0

    def get_reranker(self, top_n: int) -> "FakeReranker":
        self.created += 1
        self.top_n = top_n
        return self

    def compress_documents(self, documents: list[Document], query: str) -> list[Document]:
        self.received.append(list(documents))
        self.queries.append(query)
        if self.error:
            raise self.error
        order = self.order if self.order is not None else list(range(len(documents)))[::-1]
        results = []
        for place, position in enumerate(order[: self.top_n]):
            document = documents[position]
            # Like CohereRerank: a copy with the extra key "relevance_score"
            metadata = {**document.metadata, "relevance_score": 0.9 - 0.1 * place}
            results.append(Document(page_content=document.page_content, metadata=metadata))
        return results


@pytest.fixture(autouse=True)
def default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Do not let a local .env change the numbers the tests expect
    monkeypatch.setattr(settings, "RETRIEVAL_CANDIDATES_K", 20)
    monkeypatch.setattr(settings, "RETRIEVAL_TOP_K", 5)
    monkeypatch.setattr(settings, "RRF_K", 60)
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RERANK_MODEL", "rerank-v4.0-fast")
    monkeypatch.setattr(settings, "RERANK_CANDIDATES_K", 20)
    monkeypatch.setattr(settings, "RELEVANCE_THRESHOLD", 0.30)
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA")
    monkeypatch.setattr(settings, "RAG_LOOKBACK_YEARS", 2)
    # alembic's fileConfig can disable existing loggers in the test process (see Known issues)
    logging.getLogger("app.rag.retrieval").disabled = False


@pytest.fixture
def many_chunks(db: Session) -> list[DocumentChunk]:
    # One company, one filing and 30 chunks of one section, with texts that share words
    company = company_repository.create(db, "AAPL", "0000320193", "AAPL Inc.", None)
    filing = filing_repository.create(
        db,
        company.id,
        "AAPL-rerank",
        "10-K",
        TODAY - timedelta(days=30),
        report_date=TODAY - timedelta(days=60),
        fiscal_year=TODAY.year - 1,
        primary_document="document.htm",
    )
    texts = [f"Chunk number {number} describes supplier risk." for number in range(30)]
    vectors = llm.get_embeddings().embed_documents(texts)
    rows = [
        DocumentChunk(
            filing_id=filing.id,
            company_id=company.id,
            section="risk_factors",
            fiscal_year=filing.fiscal_year,
            chunk_index=index,
            content=text,
            embedding=vector,
            embedding_model="fake",
        )
        for index, (text, vector) in enumerate(zip(texts, vectors, strict=True))
    ]
    chunk_repository.create_many(db, rows)
    return rows


@pytest.fixture
def fake_reranker(monkeypatch: pytest.MonkeyPatch) -> FakeReranker:
    # Reranking is on (a key is set) and get_reranker returns the scripted fake
    fake = FakeReranker()
    monkeypatch.setattr(settings, "COHERE_API_KEY", "test-key")
    monkeypatch.setattr(llm, "get_reranker", fake.get_reranker)
    return fake


def chunk_ids(documents: list[Document]) -> list[int]:
    return [document.metadata["chunk_id"] for document in documents]


# ---------- the rerank step ----------


def test_result_order_follows_the_reranker(
    db: Session, many_chunks: list[DocumentChunk], fake_reranker: FakeReranker
) -> None:
    fake_reranker.order = [3, 0, 2, 1]

    documents = retrieval.retrieve(db, QUESTION, top_k=4)

    candidates = fake_reranker.received[0]
    assert chunk_ids(documents) == [
        candidates[3].metadata["chunk_id"],
        candidates[0].metadata["chunk_id"],
        candidates[2].metadata["chunk_id"],
        candidates[1].metadata["chunk_id"],
    ]
    assert fake_reranker.queries == [QUESTION]


def test_rerank_score_is_set_and_every_old_metadata_key_is_kept(
    db: Session, many_chunks: list[DocumentChunk], fake_reranker: FakeReranker
) -> None:
    fused = {
        document.metadata["chunk_id"]: document.metadata
        for document in retrieval.retrieve(db, QUESTION, top_k=1000, rerank=False)
    }

    documents = retrieval.retrieve(db, QUESTION, top_k=5)

    assert len(documents) == 5
    for place, document in enumerate(documents):
        metadata = document.metadata
        assert metadata["rerank_score"] == pytest.approx(0.9 - 0.1 * place)
        assert "relevance_score" not in metadata
        # Everything the fused search produced is still there, unchanged
        old = fused[metadata["chunk_id"]]
        assert set(metadata) == set(old)
        for key, value in old.items():
            if key != "rerank_score":
                assert metadata[key] == value
    assert fused[documents[0].metadata["chunk_id"]]["rerank_score"] is None


def test_reranker_gets_exactly_rerank_candidates_k_fused_chunks_and_top_n_is_top_k(
    db: Session,
    many_chunks: list[DocumentChunk],
    fake_reranker: FakeReranker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "RERANK_CANDIDATES_K", 7)
    fused_order = chunk_ids(retrieval.retrieve(db, QUESTION, top_k=7, rerank=False))

    documents = retrieval.retrieve(db, QUESTION, top_k=3)

    assert len(fake_reranker.received[0]) == 7
    assert chunk_ids(fake_reranker.received[0]) == fused_order  # the best 7, in fused order
    assert fake_reranker.top_n == 3
    assert len(documents) == 3


def test_fewer_candidates_are_sent_when_fewer_exist(
    db: Session,
    many_chunks: list[DocumentChunk],
    fake_reranker: FakeReranker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "RERANK_CANDIDATES_K", 100)
    fused_count = len(retrieval.retrieve(db, QUESTION, top_k=1000, rerank=False))
    assert 20 < fused_count <= 30  # the two searches found more than one list's worth

    retrieval.retrieve(db, QUESTION, top_k=5)

    assert len(fake_reranker.received[0]) == fused_count


def test_top_k_larger_than_the_candidate_count_still_works(
    db: Session,
    many_chunks: list[DocumentChunk],
    fake_reranker: FakeReranker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # RERANK_CANDIDATES_K smaller than top_k: top_k wins, so the result is never short
    monkeypatch.setattr(settings, "RERANK_CANDIDATES_K", 2)

    documents = retrieval.retrieve(db, QUESTION, top_k=6)

    assert len(fake_reranker.received[0]) == 6
    assert len(documents) == 6


def test_similarity_exists_for_every_candidate_sent_to_the_reranker(
    db: Session, many_chunks: list[DocumentChunk], fake_reranker: FakeReranker
) -> None:
    # Text-only candidates beyond the top 5 need a vector_similarity too: the chat threshold
    # reads it from whichever documents come back
    retrieval.retrieve(db, QUESTION, top_k=5)

    assert all(
        isinstance(document.metadata["vector_similarity"], float)
        for document in fake_reranker.received[0]
    )


def test_rerank_false_skips_the_reranker(
    db: Session, many_chunks: list[DocumentChunk], fake_reranker: FakeReranker
) -> None:
    documents = retrieval.retrieve(db, QUESTION, rerank=False)

    assert fake_reranker.created == 0
    assert len(documents) == 5
    assert all(document.metadata["rerank_score"] is None for document in documents)


def test_empty_key_skips_the_reranker(
    db: Session,
    many_chunks: list[DocumentChunk],
    fake_reranker: FakeReranker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "COHERE_API_KEY", "")

    documents = retrieval.retrieve(db, QUESTION)

    assert fake_reranker.created == 0
    assert all(document.metadata["rerank_score"] is None for document in documents)


def test_rerank_enabled_false_skips_the_reranker(
    db: Session,
    many_chunks: list[DocumentChunk],
    fake_reranker: FakeReranker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "RERANK_ENABLED", False)

    retrieval.retrieve(db, QUESTION)

    assert fake_reranker.created == 0


def test_rerank_none_uses_the_settings_and_true_overrides_them(
    db: Session,
    many_chunks: list[DocumentChunk],
    fake_reranker: FakeReranker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval.retrieve(db, QUESTION)  # key set and enabled: reranks
    assert fake_reranker.created == 1

    monkeypatch.setattr(settings, "RERANK_ENABLED", False)
    retrieval.retrieve(db, QUESTION, rerank=True)  # an explicit True wins over the setting
    assert fake_reranker.created == 2


def test_no_results_means_no_rerank_call(
    db: Session, many_chunks: list[DocumentChunk], fake_reranker: FakeReranker
) -> None:
    assert retrieval.retrieve(db, QUESTION, tickers=["ZZZZ"]) == []
    assert fake_reranker.created == 0


# ---------- fail open ----------


@pytest.mark.parametrize(
    "error",
    [
        TooManyRequestsError(body={"message": "secret-detail trial key limit"}),
        UnauthorizedError(body={"message": "secret-detail invalid api token"}),
        httpx.ConnectTimeout("secret-detail timed out"),
        httpx.ConnectError("secret-detail refused"),
    ],
    ids=["rate-limit", "auth", "timeout", "connection"],
)
def test_a_cohere_error_gives_the_fused_order_with_a_warning(
    db: Session,
    many_chunks: list[DocumentChunk],
    fake_reranker: FakeReranker,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    fused = retrieval.retrieve(db, QUESTION, top_k=5, rerank=False)
    fake_reranker.error = error

    with caplog.at_level(logging.WARNING, logger="app.rag.retrieval"):
        documents = retrieval.retrieve(db, QUESTION, top_k=5)

    assert fake_reranker.received  # it was tried
    assert chunk_ids(documents) == chunk_ids(fused)
    assert all(document.metadata["rerank_score"] is None for document in documents)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    # The class name is logged, never the message (which could hold part of a key)
    assert type(error).__name__ in warnings[0].getMessage()
    assert "secret-detail" not in caplog.text
    assert "test-key" not in caplog.text


def test_an_unexpected_error_is_not_swallowed(
    db: Session, many_chunks: list[DocumentChunk], fake_reranker: FakeReranker
) -> None:
    fake_reranker.error = RuntimeError("a bug, not a Cohere outage")

    with pytest.raises(RuntimeError):
        retrieval.retrieve(db, QUESTION)


# ---------- get_reranker ----------


def test_get_reranker_builds_cohere_rerank_without_a_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("get_reranker made a network call")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(settings, "COHERE_API_KEY", "test-key")
    monkeypatch.setattr(settings, "RERANK_MODEL", "rerank-v4.0-pro")
    monkeypatch.setattr(settings, "RERANK_TIMEOUT_SECONDS", 7)

    reranker = real_get_reranker(top_n=4)

    assert isinstance(reranker, CohereRerank)
    assert reranker.model == "rerank-v4.0-pro"
    assert reranker.top_n == 4
    # The SDK default would be 300 seconds
    assert reranker.client._client_wrapper.get_timeout() == 7


def test_get_reranker_raises_value_error_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "COHERE_API_KEY", "")

    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        real_get_reranker(top_n=5)


# ---------- chat ----------


@pytest.fixture
def user(db: Session) -> User:
    organization = organization_repository.create(db, "Acme")
    return user_repository.create(db, organization.id, "viewer@acme.com", "hash", "viewer")


@pytest.fixture
def chat_session(db: Session, user: User) -> ChatSession:
    return chat.create_session(db, user.org_id, user.id)


def test_chat_with_reranking_still_applies_the_threshold_on_vector_similarity(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    fake_reranker: FakeReranker,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    model = script_chat("Answer [1].")
    # The reranker puts the two irrelevant chunks FIRST with high rerank scores and the
    # matching chunk last. Only vector_similarity may decide what the model sees

    def scripted_order(documents: list[Document], query: str) -> list[Document]:
        fake_reranker.received.append(list(documents))
        ranked = sorted(documents, key=lambda d: d.metadata["vector_similarity"])  # worst first
        return [
            Document(
                page_content=document.page_content,
                metadata={**document.metadata, "relevance_score": 0.99 - 0.1 * place},
            )
            for place, document in enumerate(ranked)
        ]

    fake_reranker.compress_documents = scripted_order  # type: ignore[method-assign]

    events = list(chat.ask(db, user.org_id, user.id, chat_session.id, EXPORT_QUESTION, ticker=None))

    assert [source["content"] for source in events[0]["sources"]] == [EXPORT_QUESTION]
    assert events[0]["sources"][0]["score"] == pytest.approx(1.0)  # the similarity, not 0.79
    prompt = model.prompt_text(0)
    assert chat_chunks["nvda_mdna"].content not in prompt
    assert chat_chunks["aapl_risk"].content not in prompt


def test_prepare_context_gives_the_sources_that_ask_streams(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    fake_reranker: FakeReranker,
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat("Answer [1].")

    retrieved, passing, sources, context_text = chat.prepare_context(
        db, EXPORT_QUESTION, ticker=None
    )
    events = list(chat.ask(db, user.org_id, user.id, chat_session.id, EXPORT_QUESTION, ticker=None))

    assert events[0]["sources"] == sources
    assert len(retrieved) == 3  # every in-scope chunk was retrieved
    assert len(passing) == len(sources) == 1
    assert passing[0].page_content == EXPORT_QUESTION
    year = chat_chunks["nvda_export"].fiscal_year
    assert context_text == f"[1] NVDA FY{year} 10-K, Risk Factors:\n{EXPORT_QUESTION}"


def test_prepare_context_returns_empty_values_when_nothing_is_relevant(
    db: Session, chat_chunks: dict[str, DocumentChunk]
) -> None:
    retrieved, passing, sources, context_text = chat.prepare_context(
        db, "What is a good recipe for banana bread?", ticker=None
    )

    assert len(retrieved) == 3
    assert passing == []
    assert sources == []
    assert context_text == ""
