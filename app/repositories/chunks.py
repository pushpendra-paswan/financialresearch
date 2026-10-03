# Chunks are shared public data, so no function in this file takes an org_id.
from sqlalchemy import Select, delete, func, select
from sqlalchemy.orm import Session

from app.models.chunks import DocumentChunk
from app.models.companies import Company


def has_chunks(db: Session, filing_id: int) -> bool:
    statement = select(DocumentChunk.id).where(DocumentChunk.filing_id == filing_id).limit(1)
    return db.execute(statement).first() is not None


def delete_by_filing(db: Session, filing_id: int) -> int:
    result = db.execute(delete(DocumentChunk).where(DocumentChunk.filing_id == filing_id))
    return result.rowcount


def create_many(db: Session, rows: list[DocumentChunk]) -> None:
    db.add_all(rows)
    db.flush()


def apply_filters(
    statement: Select,
    tickers: list[str] | None,
    year_from: int | None,
    year_to: int | None,
    sections: list[str] | None,
    filing_ids: list[int] | None = None,
) -> Select:
    # Shared by both searches so they always filter the same way. Each filter is applied only
    # when given. The statement must already join companies. A chunk with no fiscal_year never
    # matches a year filter (NULL compares as unknown)
    if tickers:
        statement = statement.where(Company.ticker.in_(tickers))
    if year_from is not None:
        statement = statement.where(DocumentChunk.fiscal_year >= year_from)
    if year_to is not None:
        statement = statement.where(DocumentChunk.fiscal_year <= year_to)
    if sections:
        statement = statement.where(DocumentChunk.section.in_(sections))
    if filing_ids:
        statement = statement.where(DocumentChunk.filing_id.in_(filing_ids))
    return statement


def vector_search(
    db: Session,
    query_embedding: list[float],
    limit: int,
    tickers: list[str] | None,
    year_from: int | None,
    year_to: int | None,
    sections: list[str] | None,
    filing_ids: list[int] | None = None,
) -> list[tuple[DocumentChunk, str, float]]:
    # Exact nearest-neighbour scan by cosine distance (the <=> operator): there is no vector index.
    # Ties are broken by id so the order is deterministic
    distance = DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
    statement = (
        select(DocumentChunk, Company.ticker, distance)
        .join(Company, Company.id == DocumentChunk.company_id)
        .order_by(distance, DocumentChunk.id)
        .limit(limit)
    )
    statement = apply_filters(statement, tickers, year_from, year_to, sections, filing_ids)
    return [(chunk, ticker, dist) for chunk, ticker, dist in db.execute(statement).all()]


def fulltext_search(
    db: Session,
    query_text: str,
    limit: int,
    tickers: list[str] | None,
    year_from: int | None,
    year_to: int | None,
    sections: list[str] | None,
    filing_ids: list[int] | None = None,
) -> list[tuple[DocumentChunk, str, float]]:
    # query_text is parsed by websearch_to_tsquery with the same "english" configuration that
    # built search_vector. It never raises on odd input. Only chunks that match are returned.
    # ts_rank_cd gives coarse values with many ties, so the id breaks them
    query = func.websearch_to_tsquery("english", query_text)
    rank = func.ts_rank_cd(DocumentChunk.search_vector, query).label("rank")
    statement = (
        select(DocumentChunk, Company.ticker, rank)
        .join(Company, Company.id == DocumentChunk.company_id)
        .where(DocumentChunk.search_vector.op("@@")(query))
        .order_by(rank.desc(), DocumentChunk.id)
        .limit(limit)
    )
    statement = apply_filters(statement, tickers, year_from, year_to, sections, filing_ids)
    return [
        (chunk, ticker, rank_value) for chunk, ticker, rank_value in db.execute(statement).all()
    ]


def get_distances(
    db: Session, chunk_ids: list[int], query_embedding: list[float]
) -> dict[int, float]:
    # Cosine distance for specific chunks, used for chunks that only the full-text search found
    if not chunk_ids:
        return {}
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    statement = select(DocumentChunk.id, distance).where(DocumentChunk.id.in_(chunk_ids))
    return {chunk_id: dist for chunk_id, dist in db.execute(statement).all()}
