import logging
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.exceptions import ConflictError
from app.models.chunks import EMBEDDING_DIMENSIONS, DocumentChunk
from app.models.companies import Company
from app.models.filings import Filing
from app.models.ingestion import IngestionRun, IngestionStatus
from app.rag import chunking
from app.rag.llm import get_embeddings as real_get_embeddings  # the real one, not the test fake
from app.rag.parsing import SECTIONS
from app.repositories import companies as company_repository
from app.repositories import filings as filing_repository

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "filings"
TODAY = date.today()

# Small chunks, so each section of the fixture filings gives several chunks
TEST_CHUNK_SIZE = 300
TEST_CHUNK_OVERLAP = 50


def make_company(db: Session, ticker: str, cik: str) -> Company:
    return company_repository.create(db, ticker, cik, f"{ticker} Inc.", None)


def make_filing(
    db: Session,
    company: Company,
    accession_number: str,
    form_type: str,
    filed_on: date,
    raw_path: str | None,
    fiscal_year: int,
) -> Filing:
    filing = filing_repository.create(
        db,
        company.id,
        accession_number,
        form_type,
        filed_on,
        report_date=filed_on,
        fiscal_year=fiscal_year,
        primary_document="document.htm",
    )
    # The repository's create does not take raw_path, the download step sets it
    filing.raw_path = raw_path
    db.flush()
    return filing


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(FIXTURES_DIR))
    monkeypatch.setattr(settings, "CHUNK_SIZE", TEST_CHUNK_SIZE)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP", TEST_CHUNK_OVERLAP)
    # alembic's fileConfig can disable existing loggers in the test process (see Known issues)
    logging.getLogger("app.rag.chunking").disabled = False


@pytest.fixture
def filings(db: Session) -> dict[str, Filing]:
    # Two in-scope 10-Ks: AAPL filed 60 days ago and NVDA filed 30 days ago
    apple = make_company(db, "AAPL", "0000320193")
    nvidia = make_company(db, "NVDA", "0001045810")
    return {
        "AAPL": make_filing(
            db,
            apple,
            "0000000000-26-000001",
            "10-K",
            TODAY - timedelta(days=60),
            "aapl_10k_excerpt.htm",
            2025,
        ),
        "NVDA": make_filing(
            db,
            nvidia,
            "0000000000-26-000002",
            "10-K",
            TODAY - timedelta(days=30),
            "nvda_10k_excerpt.htm",
            2026,
        ),
    }


def get_chunks(db: Session, filing_id: int) -> list[DocumentChunk]:
    statement = (
        select(DocumentChunk)
        .where(DocumentChunk.filing_id == filing_id)
        .order_by(DocumentChunk.section, DocumentChunk.chunk_index)
    )
    return list(db.execute(statement).scalars().all())


def count_chunks(db: Session) -> int:
    return db.execute(select(func.count()).select_from(DocumentChunk)).scalar_one()


# ---------- chunks are created ----------


def test_chunks_created_for_both_sections(db: Session, filings: dict[str, Filing]) -> None:
    run = chunking.embed_filings(db)

    assert run.status == IngestionStatus.success
    assert run.job_type == "embed_filings"

    total = 0
    for filing in filings.values():
        chunks = get_chunks(db, filing.id)
        total += len(chunks)

        # Both sections have chunks, and each section was split into several
        for key in SECTIONS:
            section_chunks = [chunk for chunk in chunks if chunk.section == key]
            assert len(section_chunks) > 3

            # chunk_index counts 0..n-1 inside the (filing, section)
            assert [chunk.chunk_index for chunk in section_chunks] == list(
                range(len(section_chunks))
            )

        for chunk in chunks:
            assert chunk.company_id == filing.company_id
            assert chunk.fiscal_year == filing.fiscal_year
            assert 0 < len(chunk.content) <= TEST_CHUNK_SIZE
            assert len(chunk.embedding) == EMBEDDING_DIMENSIONS == 1536
            assert chunk.embedding_model == settings.EMBEDDING_MODEL

    assert run.message == (
        f"2 filings chunked, 0 skipped, 0 failed, {total} chunks created, 0 sections missing"
    )


def test_chunks_stay_inside_their_section(db: Session, filings: dict[str, Filing]) -> None:
    # An MD&A sentence must not appear in a risk factors chunk, and the other way around
    chunking.embed_filings(db)

    chunks = get_chunks(db, filings["AAPL"].id)
    risk_text = " ".join(chunk.content for chunk in chunks if chunk.section == "risk_factors")
    mdna_text = " ".join(chunk.content for chunk in chunks if chunk.section == "mdna")

    # The first sentence of the real AAPL MD&A (checked in the 2.1 tests)
    assert "The following discussion should be read in conjunction" in mdna_text
    assert "The following discussion should be read in conjunction" not in risk_text


def test_full_text_search_vector_is_filled(db: Session, filings: dict[str, Filing]) -> None:
    chunking.embed_filings(db)

    def matches(word: str, filing_id: int) -> int:
        return db.execute(
            text(
                "SELECT count(*) FROM document_chunks WHERE filing_id = :filing_id "
                "AND search_vector @@ plainto_tsquery('english', :word)"
            ),
            {"filing_id": filing_id, "word": word},
        ).scalar_one()

    # "manufacturing" is in the NVDA excerpt, "iPhone" only in the AAPL one
    assert matches("manufacturing", filings["NVDA"].id) > 0
    assert matches("iPhone", filings["AAPL"].id) > 0
    assert matches("iPhone", filings["NVDA"].id) == 0
    assert matches("zzyzxqq", filings["NVDA"].id) == 0


# ---------- idempotency and replace ----------


def test_second_run_creates_nothing_and_makes_no_api_call(
    db: Session, filings: dict[str, Filing], fake_embeddings: DeterministicFakeEmbedding
) -> None:
    chunking.embed_filings(db)
    count_after_first = count_chunks(db)
    calls_after_first = len(fake_embeddings.calls)
    assert calls_after_first == 2  # one embed_documents call per filing

    run = chunking.embed_filings(db)

    assert run.status == IngestionStatus.success
    assert run.message == (
        "0 filings chunked, 2 skipped, 0 failed, 0 chunks created, 0 sections missing"
    )
    assert count_chunks(db) == count_after_first
    assert len(fake_embeddings.calls) == calls_after_first


def test_replace_swaps_the_old_rows_for_new_ones(
    db: Session, filings: dict[str, Filing], fake_embeddings: DeterministicFakeEmbedding
) -> None:
    chunking.embed_filings(db)
    old_ids = {chunk.id for chunk in get_chunks(db, filings["AAPL"].id)}
    count_before = count_chunks(db)
    calls_before = len(fake_embeddings.calls)

    # Re-embed only the AAPL filing
    run = chunking.embed_filings(db, replace=True, filing_ids=[filings["AAPL"].id])

    new_ids = {chunk.id for chunk in get_chunks(db, filings["AAPL"].id)}
    assert run.status == IngestionStatus.success
    assert new_ids and not (new_ids & old_ids)  # all rows are new
    assert len(new_ids) == len(old_ids)
    assert count_chunks(db) == count_before  # the count is consistent: nothing doubled
    assert len(fake_embeddings.calls) == calls_before + 1
    # The NVDA rows were not touched
    assert get_chunks(db, filings["NVDA"].id)


def test_replace_reports_the_new_model(
    db: Session, filings: dict[str, Filing], monkeypatch: pytest.MonkeyPatch
) -> None:
    chunking.embed_filings(db)
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "another-model")

    chunking.embed_filings(db, replace=True)

    models = {
        chunk.embedding_model for filing in filings.values() for chunk in get_chunks(db, filing.id)
    }
    assert models == {"another-model"}


# ---------- failures ----------


def test_failing_filing_has_no_chunks_and_the_run_is_partial(
    db: Session, filings: dict[str, Filing], fake_embeddings: DeterministicFakeEmbedding
) -> None:
    # An OpenAI error for the NVDA filing only: "GPU" appears in the NVDA excerpt, not in AAPL's
    fake_embeddings.fail_when_text_contains = "GPU"

    run = chunking.embed_filings(db)

    assert run.status == IngestionStatus.partial
    assert get_chunks(db, filings["NVDA"].id) == []
    assert get_chunks(db, filings["AAPL"].id)
    assert "1 filings chunked, 0 skipped, 1 failed" in run.message


def test_failing_replace_keeps_the_old_chunks(
    db: Session, filings: dict[str, Filing], fake_embeddings: DeterministicFakeEmbedding
) -> None:
    chunking.embed_filings(db)
    old_nvda = [chunk.id for chunk in get_chunks(db, filings["NVDA"].id)]
    old_apple = {chunk.id for chunk in get_chunks(db, filings["AAPL"].id)}

    fake_embeddings.fail_when_text_contains = "GPU"
    run = chunking.embed_filings(db, replace=True)

    assert run.status == IngestionStatus.partial
    # NVDA failed: exactly the same old rows are still there
    assert [chunk.id for chunk in get_chunks(db, filings["NVDA"].id)] == old_nvda
    # AAPL did not fail: it was replaced
    new_apple = {chunk.id for chunk in get_chunks(db, filings["AAPL"].id)}
    assert new_apple and not (new_apple & old_apple)


def test_wrong_dimension_fails_the_filing_with_a_clear_message(
    db: Session,
    filings: dict[str, Filing],
    fake_embeddings: DeterministicFakeEmbedding,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_embeddings.size = 8

    with caplog.at_level(logging.ERROR, logger="app.rag.chunking"):
        run = chunking.embed_filings(db)

    assert run.status == IngestionStatus.partial
    assert count_chunks(db) == 0
    assert "2 failed" in run.message
    assert f"Embedding has 8 dimensions but the table expects {EMBEDDING_DIMENSIONS}" in (
        caplog.text
    )


def test_filing_without_sections_is_failed_and_not_embedded(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_embeddings: DeterministicFakeEmbedding,
) -> None:
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(tmp_path))
    (tmp_path / "empty.htm").write_text("<html><body><p>Nothing here.</p></body></html>")
    company = make_company(db, "AAPL", "0000320193")
    filing = make_filing(
        db, company, "0000000000-26-000003", "10-K", TODAY, "empty.htm", TODAY.year
    )

    run = chunking.embed_filings(db, filing_ids=[filing.id])

    assert run.status == IngestionStatus.partial
    assert "0 filings chunked, 0 skipped, 1 failed, 0 chunks created, 2 sections missing" == (
        run.message
    )
    assert fake_embeddings.calls == []  # nothing to embed, so no API call


def test_missing_raw_file_fails_the_filing_not_the_run(
    db: Session, filings: dict[str, Filing]
) -> None:
    # TextLoader raises RuntimeError for a missing file; the other filing must still be done
    filings["AAPL"].raw_path = "does_not_exist.htm"
    db.flush()

    run = chunking.embed_filings(db)

    assert run.status == IngestionStatus.partial
    assert get_chunks(db, filings["AAPL"].id) == []
    assert get_chunks(db, filings["NVDA"].id)


def test_unexpected_error_marks_the_run_failed_and_reraises(
    db: Session, filings: dict[str, Filing], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args, **kwargs):
        raise KeyError("boom")

    monkeypatch.setattr(chunking, "load_filing_sections", broken)

    with pytest.raises(KeyError):
        chunking.embed_filings(db)

    run = db.execute(
        select(IngestionRun).where(IngestionRun.job_type == "embed_filings")
    ).scalar_one()
    assert run.status == IngestionStatus.failed
    assert "KeyError" in run.error


def test_overlapping_run_is_refused(db: Session, filings: dict[str, Filing]) -> None:
    db.add(IngestionRun(job_type="embed_filings", status=IngestionStatus.running))
    db.flush()

    with pytest.raises(ConflictError) as error:
        chunking.embed_filings(db)

    assert error.value.message == "A filing embedding ingestion run is already in progress"


# ---------- scope ----------


def test_out_of_scope_filings_are_never_chunked(
    db: Session, filings: dict[str, Filing], fake_embeddings: DeterministicFakeEmbedding
) -> None:
    apple = db.get(Company, filings["AAPL"].company_id)
    microsoft = make_company(db, "MSFT", "0000789019")
    # Another ticker, a 10-Q of an in-scope ticker, and a 10-K filed 900 days ago
    other_ticker = make_filing(
        db, microsoft, "0000000000-26-000010", "10-K", TODAY, "aapl_10k_excerpt.htm", 2026
    )
    quarterly = make_filing(
        db, apple, "0000000000-26-000011", "10-Q", TODAY, "aapl_10k_excerpt.htm", 2026
    )
    too_old = make_filing(
        db,
        apple,
        "0000000000-24-000012",
        "10-K",
        TODAY - timedelta(days=900),
        "aapl_10k_excerpt.htm",
        2023,
    )

    run = chunking.embed_filings(db)

    assert run.message.startswith("2 filings chunked, 0 skipped, 0 failed")
    for out_of_scope in (other_ticker, quarterly, too_old):
        assert get_chunks(db, out_of_scope.id) == []
    assert len(fake_embeddings.calls) == 2  # only the two in-scope filings


def test_explicit_filing_ids_limit_the_run(db: Session, filings: dict[str, Filing]) -> None:
    run = chunking.embed_filings(db, filing_ids=[filings["NVDA"].id])

    assert run.message.startswith("1 filings chunked, 0 skipped, 0 failed")
    assert get_chunks(db, filings["AAPL"].id) == []
    assert get_chunks(db, filings["NVDA"].id)


# ---------- llm.get_embeddings ----------


def test_get_embeddings_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        real_get_embeddings()


def test_get_embeddings_uses_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only builds the object: no request is sent
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 25)

    embeddings = real_get_embeddings()

    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == "text-embedding-3-small"
    assert embeddings.chunk_size == 25
    assert embeddings.check_embedding_ctx_length is False


# ---------- Celery ----------


def test_embed_task_is_registered_and_has_no_schedule() -> None:
    from app.workers import tasks  # noqa: F401  (importing registers the tasks)
    from app.workers.celery_app import celery_app

    assert "embed_filings" in celery_app.tasks
    # It is chained after ingest_filings, so it has no beat entry of its own
    scheduled_tasks = [entry["task"] for entry in celery_app.conf.beat_schedule.values()]
    assert "embed_filings" not in scheduled_tasks


def test_ingest_filings_enqueues_embedding_after_a_normal_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.workers import tasks

    run = SimpleNamespace(id=1, status="success", message="done")
    monkeypatch.setattr(tasks, "SessionLocal", MagicMock())
    monkeypatch.setattr(tasks.ingestion_service, "ingest_filings", lambda db: run)
    delay = MagicMock()
    monkeypatch.setattr(tasks.embed_filings, "delay", delay)

    tasks.ingest_filings()

    delay.assert_called_once_with()


def test_ingest_filings_does_not_enqueue_embedding_when_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.workers import tasks

    def skipped(db):
        raise ConflictError("A filing ingestion run is already in progress")

    monkeypatch.setattr(tasks, "SessionLocal", MagicMock())
    monkeypatch.setattr(tasks.ingestion_service, "ingest_filings", skipped)
    delay = MagicMock()
    monkeypatch.setattr(tasks.embed_filings, "delay", delay)

    assert tasks.ingest_filings() is None

    delay.assert_not_called()


def test_embed_task_without_a_key_does_nothing(
    db: Session, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from app.workers import tasks

    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    session_factory = MagicMock()
    monkeypatch.setattr(tasks, "SessionLocal", session_factory)
    service = MagicMock()
    monkeypatch.setattr(tasks.chunking, "embed_filings", service)
    logging.getLogger("app.workers.tasks").disabled = False

    with caplog.at_level(logging.WARNING, logger="app.workers.tasks"):
        result = tasks.embed_filings()

    assert result is None
    assert "embeddings disabled: OPENAI_API_KEY not set" in caplog.text
    # No session, no service call, so no ingestion_runs row can have been created
    session_factory.assert_not_called()
    service.assert_not_called()
    rows = db.execute(
        select(func.count())
        .select_from(IngestionRun)
        .where(IngestionRun.job_type == "embed_filings")
    ).scalar_one()
    assert rows == 0


def test_embed_task_with_a_key_runs_the_service(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.workers import tasks

    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test-not-a-real-key")
    session = MagicMock()
    monkeypatch.setattr(tasks, "SessionLocal", lambda: session)
    run = SimpleNamespace(id=7, status="success", message="4 filings chunked")
    monkeypatch.setattr(tasks.chunking, "embed_filings", lambda db: run)

    assert tasks.embed_filings() == "success: 4 filings chunked"
    session.close.assert_called_once_with()
