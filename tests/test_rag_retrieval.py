import math
from datetime import date, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.chunks import DocumentChunk
from app.rag import llm, retrieval
from app.repositories import chunks as chunk_repository
from app.repositories import companies as company_repository
from app.repositories import filings as filing_repository

TODAY = date.today()
LAST_YEAR = TODAY.year - 1
YEAR_BEFORE = TODAY.year - 2

# (name, ticker, fiscal year, section, chunk_index, content). The content is hand-written, and
# each chunk has a distinctive word ("tariff", "outsourcing", ...). The fake embeddings give the
# same vector for the same text, so a question equal to a chunk's content finds that chunk first.
# The vectors of different texts are random: no test relies on "similar meaning"
CHUNK_DATA = [
    (
        "aapl_risk",
        "AAPL",
        LAST_YEAR,
        "risk_factors",
        0,
        "Apple depends on outsourcing partners in Asia for the manufacture of its products.",
    ),
    (
        "aapl_risk_old",
        "AAPL",
        YEAR_BEFORE,
        "risk_factors",
        0,
        "Apple faces intense competition in smartphones and personal computers.",
    ),
    (
        "aapl_mdna",
        "AAPL",
        LAST_YEAR,
        "mdna",
        0,
        "iPhone net sales increased during the year compared with the prior year.",
    ),
    (
        "aapl_mdna_old",
        "AAPL",
        YEAR_BEFORE,
        "mdna",
        0,
        "Services net sales grew because of advertising and the App Store.",
    ),
    (
        "nvda_tariff",
        "NVDA",
        LAST_YEAR,
        "risk_factors",
        0,
        "New tariff measures could raise the cost of our graphics processors.",
    ),
    (
        "nvda_export",
        "NVDA",
        LAST_YEAR,
        "risk_factors",
        1,
        "Export controls restrict sales of data center products to China.",
    ),
    (
        "nvda_mdna",
        "NVDA",
        LAST_YEAR,
        "mdna",
        0,
        "Data Center revenue grew on demand for accelerated computing.",
    ),
    (
        "nvda_mdna_old",
        "NVDA",
        YEAR_BEFORE,
        "mdna",
        0,
        "Gaming revenue declined as channel inventory normalized.",
    ),
]


@pytest.fixture(autouse=True)
def default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Do not let a local .env change the numbers the tests expect
    monkeypatch.setattr(settings, "RETRIEVAL_CANDIDATES_K", 20)
    monkeypatch.setattr(settings, "RETRIEVAL_TOP_K", 5)
    monkeypatch.setattr(settings, "RRF_K", 60)


@pytest.fixture
def chunks(db: Session) -> dict[str, DocumentChunk]:
    # Two companies, two 10-Ks each (last year and the year before) and the 8 chunks above.
    # Chunks are created in the order of CHUNK_DATA, so their ids increase in that order
    filing_ids = {}
    company_ids = {}
    for ticker, cik in (("AAPL", "0000320193"), ("NVDA", "0001045810")):
        company = company_repository.create(db, ticker, cik, f"{ticker} Inc.", None)
        company_ids[ticker] = company.id
        for number, fiscal_year in enumerate((LAST_YEAR, YEAR_BEFORE)):
            filing = filing_repository.create(
                db,
                company.id,
                f"{cik}-{number}",
                "10-K",
                TODAY - timedelta(days=30 + 365 * number),
                report_date=TODAY - timedelta(days=60 + 365 * number),
                fiscal_year=fiscal_year,
                primary_document="document.htm",
            )
            filing_ids[(ticker, fiscal_year)] = filing.id

    vectors = llm.get_embeddings().embed_documents([row[5] for row in CHUNK_DATA])
    rows = [
        DocumentChunk(
            filing_id=filing_ids[(ticker, fiscal_year)],
            company_id=company_ids[ticker],
            section=section,
            fiscal_year=fiscal_year,
            chunk_index=chunk_index,
            content=content,
            embedding=vector,
            embedding_model="fake",
        )
        for (_name, ticker, fiscal_year, section, chunk_index, content), vector in zip(
            CHUNK_DATA, vectors, strict=True
        )
    ]
    chunk_repository.create_many(db, rows)
    return {row[0]: chunk for row, chunk in zip(CHUNK_DATA, rows, strict=True)}


def cosine_similarity(first: list[float], second: list[float]) -> float:
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    return dot / (math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second)))


def fake_searches(
    monkeypatch: pytest.MonkeyPatch,
    vector_chunks: list[DocumentChunk],
    text_chunks: list[DocumentChunk],
) -> None:
    # Replace both searches with fixed ranked lists, so the fusion arithmetic is tested with
    # known ranks. The tickers and scores in the tuples are not used by the fusion
    monkeypatch.setattr(
        chunk_repository,
        "vector_search",
        lambda *args, **kwargs: [(chunk, "AAPL", 0.5) for chunk in vector_chunks],
    )
    monkeypatch.setattr(
        chunk_repository,
        "fulltext_search",
        lambda *args, **kwargs: [(chunk, "AAPL", 1.0) for chunk in text_chunks],
    )


# ---------- vector search ----------


def test_question_equal_to_a_chunk_returns_that_chunk_first(
    db: Session, chunks: dict[str, DocumentChunk]
) -> None:
    target = chunks["nvda_export"]

    documents = retrieval.retrieve(db, target.content)

    first = documents[0]
    assert first.metadata["chunk_id"] == target.id
    assert first.page_content == target.content
    assert first.metadata["vector_rank"] == 1
    assert first.metadata["vector_similarity"] == pytest.approx(1.0)


def test_vector_similarity_matches_the_cosine_formula(
    db: Session, chunks: dict[str, DocumentChunk]
) -> None:
    question = "What is the outlook for gaming demand?"
    query_embedding = llm.get_embeddings().embed_query(question)

    documents = retrieval.retrieve(db, question, top_k=8)

    assert len(documents) == 8
    for document in documents:
        chunk = next(c for c in chunks.values() if c.id == document.metadata["chunk_id"])
        expected = cosine_similarity(query_embedding, list(chunk.embedding))
        assert document.metadata["vector_similarity"] == pytest.approx(expected, abs=1e-6)


# ---------- full-text search ----------


def test_distinctive_word_finds_the_chunk_with_a_text_rank(
    db: Session, chunks: dict[str, DocumentChunk]
) -> None:
    documents = retrieval.retrieve(db, "Is there any tariff risk?")

    # Only one chunk contains "tariff". It is in both lists, so it must beat every chunk that is
    # in the vector list only
    first = documents[0]
    assert first.metadata["chunk_id"] == chunks["nvda_tariff"].id
    assert first.metadata["text_rank"] == 1
    assert first.metadata["vector_rank"] is not None
    assert all(document.metadata["text_rank"] is None for document in documents[1:])


def test_question_with_no_matching_words_still_returns_vector_results(
    db: Session, chunks: dict[str, DocumentChunk]
) -> None:
    documents = retrieval.retrieve(db, "zzzquux blorptastic")

    assert len(documents) == 5
    assert all(document.metadata["text_rank"] is None for document in documents)
    assert all(document.metadata["vector_rank"] is not None for document in documents)


def test_odd_input_does_not_raise(db: Session, chunks: dict[str, DocumentChunk]) -> None:
    questions = [
        "\"quoted phrase\" -excluded it's Apple's",  # quotes, a leading minus, apostrophes
        "the and or of a to",  # only stop words
        "or or or",  # the word "or" is an operator for websearch_to_tsquery
        "???",  # no words at all
        "'; DROP TABLE document_chunks; --",
        "tariff:* & (export | china) !data <-> centre",  # tsquery operator characters
    ]
    for question in questions:
        documents = retrieval.retrieve(db, question)
        assert len(documents) == 5


def test_text_search_uses_or_not_and(db: Session, chunks: dict[str, DocumentChunk]) -> None:
    # No chunk has both words, so an AND query would match nothing
    documents = retrieval.retrieve(db, "tariff outsourcing", top_k=8)

    with_text_rank = {
        document.metadata["chunk_id"]
        for document in documents
        if document.metadata["text_rank"] is not None
    }
    assert with_text_rank == {chunks["nvda_tariff"].id, chunks["aapl_risk"].id}


# ---------- fusion ----------


def test_chunk_found_by_both_searches_outranks_chunks_found_by_one(
    db: Session, chunks: dict[str, DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    # vector list: E, A, B, C   text list: C, D, A   (E, A, B, C are ranks 1..4 in the vector list)
    e, a, b, c, d = (
        chunks["aapl_risk"],
        chunks["aapl_mdna"],
        chunks["aapl_mdna_old"],
        chunks["nvda_tariff"],
        chunks["nvda_export"],
    )
    fake_searches(monkeypatch, [e, a, b, c], [c, d, a])

    documents = retrieval.retrieve(db, "anything", top_k=5)

    # Scores: C = 1/64 + 1/61, A = 1/62 + 1/63, E = 1/61, D = 1/62, B = 1/63
    assert [document.metadata["chunk_id"] for document in documents] == [
        c.id,
        a.id,
        e.id,
        d.id,
        b.id,
    ]
    scores = [document.metadata["score"] for document in documents]
    assert scores == pytest.approx(
        [1 / 64 + 1 / 61, 1 / 62 + 1 / 63, 1 / 61, 1 / 62, 1 / 63], abs=1e-12
    )
    by_id = {document.metadata["chunk_id"]: document.metadata for document in documents}
    assert (by_id[c.id]["vector_rank"], by_id[c.id]["text_rank"]) == (4, 1)
    assert (by_id[d.id]["vector_rank"], by_id[d.id]["text_rank"]) == (None, 2)
    assert (by_id[e.id]["vector_rank"], by_id[e.id]["text_rank"]) == (1, None)


def test_equal_scores_are_ordered_by_chunk_id(
    db: Session, chunks: dict[str, DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = chunks["aapl_risk"], chunks["aapl_risk_old"]
    assert first.id < second.id

    # Both chunks have the same score, 1/61 + 1/62, whichever list order is used
    for vector_list, text_list in (
        ([first, second], [second, first]),
        ([second, first], [first, second]),
    ):
        fake_searches(monkeypatch, vector_list, text_list)

        documents = retrieval.retrieve(db, "anything")

        assert [document.metadata["chunk_id"] for document in documents] == [first.id, second.id]
        assert documents[0].metadata["score"] == pytest.approx(1 / 61 + 1 / 62)
        assert documents[1].metadata["score"] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_k_setting_is_used(
    db: Session, chunks: dict[str, DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "RRF_K", 10)
    fake_searches(monkeypatch, [chunks["aapl_risk"]], [chunks["aapl_risk"]])

    documents = retrieval.retrieve(db, "anything")

    assert documents[0].metadata["score"] == pytest.approx(2 / 11)


def test_text_only_chunk_still_gets_a_vector_similarity(
    db: Session, chunks: dict[str, DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    # With 2 vector candidates, a chunk can be missing from the vector list and still be found by
    # the text search. The fake vectors are random, so pick a distinctive word whose chunk is NOT
    # among the 2 nearest chunks of that question
    monkeypatch.setattr(settings, "RETRIEVAL_CANDIDATES_K", 2)
    word_to_chunk = {
        "tariff": "nvda_tariff",
        "outsourcing": "aapl_risk",
        "advertising": "aapl_mdna_old",
        "inventory": "nvda_mdna_old",
    }
    for question, chunk_name in word_to_chunk.items():
        query_embedding = llm.get_embeddings().embed_query(question)
        nearest = chunk_repository.vector_search(db, query_embedding, 2, None, None, None, None)
        if chunks[chunk_name].id not in [chunk.id for chunk, _ticker, _distance in nearest]:
            break
    else:
        pytest.fail("every distinctive word was in the vector top 2: change the words")

    documents = retrieval.retrieve(db, question)

    found = next(d for d in documents if d.metadata["chunk_id"] == chunks[chunk_name].id)
    assert found.metadata["vector_rank"] is None
    assert found.metadata["text_rank"] == 1
    expected = cosine_similarity(query_embedding, list(chunks[chunk_name].embedding))
    assert found.metadata["vector_similarity"] == pytest.approx(expected, abs=1e-6)


# ---------- filters ----------


def test_ticker_filter_is_case_insensitive(db: Session, chunks: dict[str, DocumentChunk]) -> None:
    documents = retrieval.retrieve(db, "tariff export controls", tickers=["nvda"], top_k=8)

    assert len(documents) == 4
    assert {document.metadata["ticker"] for document in documents} == {"NVDA"}


def test_year_range_filter(db: Session, chunks: dict[str, DocumentChunk]) -> None:
    newest = retrieval.retrieve(db, "sales", year_from=LAST_YEAR, top_k=8)
    oldest = retrieval.retrieve(db, "sales", year_to=YEAR_BEFORE, top_k=8)
    both = retrieval.retrieve(db, "sales", year_from=YEAR_BEFORE, year_to=YEAR_BEFORE, top_k=8)

    assert {d.metadata["fiscal_year"] for d in newest} == {LAST_YEAR}
    assert len(newest) == 5
    assert {d.metadata["fiscal_year"] for d in oldest} == {YEAR_BEFORE}
    assert len(oldest) == 3
    assert len(both) == 3


def test_section_filter(db: Session, chunks: dict[str, DocumentChunk]) -> None:
    mdna = retrieval.retrieve(db, "revenue", sections=["mdna"], top_k=8)
    both = retrieval.retrieve(db, "revenue", sections=["mdna", "risk_factors"], top_k=8)

    assert {d.metadata["section"] for d in mdna} == {"mdna"}
    assert len(mdna) == 4
    assert len(both) == 8


def test_combined_filters(db: Session, chunks: dict[str, DocumentChunk]) -> None:
    documents = retrieval.retrieve(
        db,
        "Apple outsourcing partners",
        tickers=["AAPL"],
        sections=["risk_factors"],
        year_from=LAST_YEAR,
        top_k=8,
    )

    assert [d.metadata["chunk_id"] for d in documents] == [chunks["aapl_risk"].id]


def test_unknown_ticker_returns_an_empty_list(
    db: Session, chunks: dict[str, DocumentChunk]
) -> None:
    assert retrieval.retrieve(db, "tariff", tickers=["ZZZZ"]) == []


def test_empty_filter_lists_mean_no_filter(db: Session, chunks: dict[str, DocumentChunk]) -> None:
    documents = retrieval.retrieve(db, "tariff", tickers=[], sections=[], top_k=8)

    assert len(documents) == 8


def test_filters_apply_to_the_text_search_too(
    db: Session, chunks: dict[str, DocumentChunk]
) -> None:
    # "tariff" only exists in an NVDA chunk, so an AAPL search must not see it, in either list
    documents = retrieval.retrieve(db, "tariff", tickers=["AAPL"], top_k=8)

    assert len(documents) == 4
    assert all(d.metadata["ticker"] == "AAPL" for d in documents)
    assert all(d.metadata["text_rank"] is None for d in documents)


# ---------- output ----------


def test_metadata_has_exactly_the_documented_keys_and_values(
    db: Session, chunks: dict[str, DocumentChunk]
) -> None:
    target = chunks["aapl_mdna"]

    document = retrieval.retrieve(db, target.content)[0]

    assert set(document.metadata) == {
        "chunk_id",
        "filing_id",
        "company_id",
        "ticker",
        "fiscal_year",
        "section",
        "chunk_index",
        "score",
        "vector_similarity",
        "vector_rank",
        "text_rank",
    }
    assert document.metadata["chunk_id"] == target.id
    assert document.metadata["filing_id"] == target.filing_id
    assert document.metadata["company_id"] == target.company_id
    assert document.metadata["ticker"] == "AAPL"
    assert document.metadata["fiscal_year"] == LAST_YEAR
    assert document.metadata["section"] == "mdna"
    assert document.metadata["chunk_index"] == 0
    assert document.metadata["vector_rank"] == 1
    assert document.metadata["text_rank"] == 1
    assert document.metadata["score"] == pytest.approx(2 / 61)


def test_question_is_embedded_once(
    db: Session, chunks: dict[str, DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    # Patch the class of the fake in use (tests.conftest would be a second copy of the module)
    fake_class = type(llm.get_embeddings())
    original = fake_class.embed_query

    def counting_embed_query(self, text: str) -> list[float]:
        calls.append(text)
        return original(self, text)

    monkeypatch.setattr(fake_class, "embed_query", counting_embed_query)

    retrieval.retrieve(db, "  What about the tariff?  ")

    assert calls == ["What about the tariff?"]


# ---------- validation and config ----------


def test_invalid_input_raises_value_error(db: Session, chunks: dict[str, DocumentChunk]) -> None:
    with pytest.raises(ValueError, match="empty"):
        retrieval.retrieve(db, "   ")
    with pytest.raises(ValueError, match="Unknown section"):
        retrieval.retrieve(db, "tariff", sections=["risk_factors", "business"])
    with pytest.raises(ValueError, match="year_from"):
        retrieval.retrieve(db, "tariff", year_from=LAST_YEAR, year_to=YEAR_BEFORE)
    with pytest.raises(ValueError, match="top_k"):
        retrieval.retrieve(db, "tariff", top_k=0)


def test_top_k_argument_and_settings_are_respected(
    db: Session, chunks: dict[str, DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert len(retrieval.retrieve(db, "sales")) == 5  # RETRIEVAL_TOP_K
    assert len(retrieval.retrieve(db, "sales", top_k=2)) == 2

    monkeypatch.setattr(settings, "RETRIEVAL_TOP_K", 3)
    assert len(retrieval.retrieve(db, "sales")) == 3

    # One candidate from each search gives at most 2 different chunks, even with a large top_k
    monkeypatch.setattr(settings, "RETRIEVAL_CANDIDATES_K", 1)
    assert len(retrieval.retrieve(db, "sales", top_k=8)) <= 2


def test_retrieval_writes_nothing(db: Session, chunks: dict[str, DocumentChunk]) -> None:
    count = select(func.count()).select_from(DocumentChunk)

    retrieval.retrieve(db, "tariff")

    assert db.execute(count).scalar_one() == 8
