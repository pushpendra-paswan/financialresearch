import logging

import openai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from app.config import settings
from app.models.chunks import EMBEDDING_DIMENSIONS, DocumentChunk
from app.models.ingestion import IngestionRun
from app.rag import llm
from app.rag.parsing import SECTIONS, list_scope_filing_ids, load_filing_sections
from app.repositories import chunks as chunk_repository
from app.services import ingestion as ingestion_service

logger = logging.getLogger(__name__)

JOB_TYPE = "embed_filings"


def embed_filings(
    db: Session, replace: bool = False, filing_ids: list[int] | None = None
) -> IngestionRun:
    # 1. Guard against an overlapping run and record this one
    run = ingestion_service.start_run(db, JOB_TYPE, "filing embedding")

    # The one broad except below only marks the run "failed", logs and re-raises, so a crash
    # never leaves the run "running" forever. Per-filing errors are caught inside the loop.
    try:
        # 2. Create the embeddings object once, and the splitter from the config values
        embeddings = llm.get_embeddings()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
        )
        if filing_ids is None:
            filing_ids = list_scope_filing_ids(db)

        filings_chunked = 0
        filings_skipped = 0
        filings_failed = 0
        chunks_created = 0
        sections_missing = 0

        # 3. One filing at a time, so one failure never stops the others
        for filing_id in filing_ids:
            # Already chunked: skip with no API call (a re-embed run replaces instead)
            if not replace and chunk_repository.has_chunks(db, filing_id):
                filings_skipped += 1
                continue

            try:
                # Parse the filing into one Document per section, then split inside each section
                # only. split_documents copies the section's metadata to every chunk
                sections = load_filing_sections(db, filing_id)
                sections_missing += len(SECTIONS) - len(sections)
                chunks = splitter.split_documents(sections)
                if not chunks:
                    raise ValueError("no section found in the filing")

                # Embed all chunks of the filing. Nothing is written to the database before this
                # succeeds, so a failure leaves the filing's old chunks untouched.
                # RuntimeError: TextLoader raises it when the raw file is missing or unreadable
                vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
                if len(vectors) != len(chunks):
                    raise ValueError(
                        f"Got {len(vectors)} embeddings for {len(chunks)} chunks "
                        f"(model {settings.EMBEDDING_MODEL})"
                    )
                for vector in vectors:
                    if len(vector) != EMBEDDING_DIMENSIONS:
                        raise ValueError(
                            f"Embedding has {len(vector)} dimensions but the table expects "
                            f"{EMBEDDING_DIMENSIONS}: check EMBEDDING_MODEL "
                            f"({settings.EMBEDDING_MODEL})"
                        )
            except (openai.OpenAIError, ValueError, OSError, RuntimeError) as error:
                # An OpenAI error message can contain part of the API key, so only its name is
                # logged
                detail = type(error).__name__ if isinstance(error, openai.OpenAIError) else error
                logger.error("Embedding failed for filing %d: %s", filing_id, detail)
                filings_failed += 1
                continue

            # Build the rows. chunk_index counts from 0 inside each (filing, section)
            next_index: dict[str, int] = {}
            rows = []
            for chunk, vector in zip(chunks, vectors, strict=True):
                section = chunk.metadata["section"]
                chunk_index = next_index.get(section, 0)
                next_index[section] = chunk_index + 1
                rows.append(
                    DocumentChunk(
                        filing_id=chunk.metadata["filing_id"],
                        company_id=chunk.metadata["company_id"],
                        section=section,
                        fiscal_year=chunk.metadata["fiscal_year"],
                        chunk_index=chunk_index,
                        content=chunk.page_content,
                        embedding=vector,
                        embedding_model=settings.EMBEDDING_MODEL,
                    )
                )

            # One transaction per filing: the old chunks (re-embed only) go and the new ones
            # arrive together
            if replace:
                chunk_repository.delete_by_filing(db, filing_id)
            chunk_repository.create_many(db, rows)
            db.commit()
            filings_chunked += 1
            chunks_created += len(rows)
            logger.info("Filing %d: %d chunks stored", filing_id, len(rows))

        # 4. Finish the run: "partial" when at least one filing failed
        message = (
            f"{filings_chunked} filings chunked, {filings_skipped} skipped, "
            f"{filings_failed} failed, {chunks_created} chunks created, "
            f"{sections_missing} sections missing"
        )
        return ingestion_service.finish_run(db, run, filings_failed, message)
    except Exception as error:
        logger.exception("Embedding run %d failed", run.id)
        ingestion_service.fail_run(db, run, error)
        raise
