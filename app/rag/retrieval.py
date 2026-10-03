import logging
import re

from langchain_core.documents import Document
from sqlalchemy.orm import Session

from app.config import settings
from app.rag import llm
from app.rag.parsing import SECTIONS
from app.repositories import chunks as chunk_repository

logger = logging.getLogger(__name__)


# Hybrid search: a pgvector similarity search and a Postgres full-text search, merged with
# reciprocal rank fusion (RRF). It is plain Python on purpose: LangChain's EnsembleRetriever lives
# in langchain-classic (not allowed here), and a BaseRetriever subclass would be a custom Runnable.
# Chunks are shared public data, so there is no org_id.
def retrieve(
    db: Session,
    question: str,
    tickers: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    sections: list[str] | None = None,
    top_k: int | None = None,
    filing_ids: list[int] | None = None,
) -> list[Document]:
    # 1. Validate the input. An unknown ticker is not an error, it simply matches nothing
    question = question.strip()
    if not question:
        raise ValueError("The question is empty")
    if sections:
        unknown_sections = [section for section in sections if section not in SECTIONS]
        if unknown_sections:
            raise ValueError(f"Unknown section(s) {unknown_sections}: use one of {list(SECTIONS)}")
    if year_from is not None and year_to is not None and year_from > year_to:
        raise ValueError("year_from must not be greater than year_to")
    if top_k is None:
        top_k = settings.RETRIEVAL_TOP_K
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    # An empty list means "no filter", the same as None (the repository checks for that). That
    # includes filing_ids: a caller that restricts to a scope must handle an empty scope itself
    if tickers:
        tickers = [ticker.strip().upper() for ticker in tickers]
    candidates_k = settings.RETRIEVAL_CANDIDATES_K

    # 2. Embed the question once; both the vector search and the distances below use it
    query_embedding = llm.get_embeddings().embed_query(question)

    # 3. Vector search: the nearest chunks by cosine distance, nearest first
    vector_hits = chunk_repository.vector_search(
        db, query_embedding, candidates_k, tickers, year_from, year_to, sections, filing_ids
    )

    # 4. Full-text search. A question is a natural sentence, so an AND query (plainto_tsquery)
    # usually matches nothing. We join the words with "or" instead: websearch_to_tsquery reads it
    # as OR, drops stop words and never raises on user input. Extracting \w+ words first removes
    # quotes, a leading "-" and apostrophes, which that function would treat as operators
    words = re.findall(r"\w+", question)
    text_hits = []
    if words:
        text_hits = chunk_repository.fulltext_search(
            db, " or ".join(words), candidates_k, tickers, year_from, year_to, sections, filing_ids
        )

    # 5. Reciprocal rank fusion: score = sum of 1 / (RRF_K + rank) over the lists a chunk is in
    # (rank starts at 1). An RRF score only orders chunks, it has no absolute meaning
    scores: dict[int, float] = {}
    chunks_by_id = {}
    tickers_by_id: dict[int, str] = {}
    vector_ranks: dict[int, int] = {}
    text_ranks: dict[int, int] = {}
    distances: dict[int, float] = {}
    for rank, (chunk, ticker, distance) in enumerate(vector_hits, start=1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (settings.RRF_K + rank)
        chunks_by_id[chunk.id] = chunk
        tickers_by_id[chunk.id] = ticker
        vector_ranks[chunk.id] = rank
        distances[chunk.id] = distance
    for rank, (chunk, ticker, _text_score) in enumerate(text_hits, start=1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (settings.RRF_K + rank)
        chunks_by_id[chunk.id] = chunk
        tickers_by_id[chunk.id] = ticker
        text_ranks[chunk.id] = rank
    # Highest score first; equal scores are ordered by chunk id so the result is deterministic
    top_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]

    # 6. Chunks that only the full-text search found have no distance yet. The relevance
    # threshold in 2.4 uses vector_similarity, so every returned chunk needs one
    missing_ids = [chunk_id for chunk_id in top_ids if chunk_id not in distances]
    distances.update(chunk_repository.get_distances(db, missing_ids, query_embedding))

    # 7. Build the Documents: the chunk text plus everything needed to cite and debug it
    documents = []
    for chunk_id in top_ids:
        chunk = chunks_by_id[chunk_id]
        documents.append(
            Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.id,
                    "filing_id": chunk.filing_id,
                    "company_id": chunk.company_id,
                    "ticker": tickers_by_id[chunk_id],
                    "fiscal_year": chunk.fiscal_year,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "score": scores[chunk_id],
                    "vector_similarity": 1 - distances[chunk_id],
                    "vector_rank": vector_ranks.get(chunk_id),
                    "text_rank": text_ranks.get(chunk_id),
                },
            )
        )

    logger.info(
        "retrieved %d chunks (%d vector candidates, %d full-text candidates)",
        len(documents),
        len(vector_hits),
        len(text_hits),
    )
    return documents
