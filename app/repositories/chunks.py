# Chunks are shared public data, so no function in this file takes an org_id.
# Similarity and full-text queries are added in 2.3.
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.chunks import DocumentChunk


def has_chunks(db: Session, filing_id: int) -> bool:
    statement = select(DocumentChunk.id).where(DocumentChunk.filing_id == filing_id).limit(1)
    return db.execute(statement).first() is not None


def delete_by_filing(db: Session, filing_id: int) -> int:
    result = db.execute(delete(DocumentChunk).where(DocumentChunk.filing_id == filing_id))
    return result.rowcount


def create_many(db: Session, rows: list[DocumentChunk]) -> None:
    db.add_all(rows)
    db.flush()
