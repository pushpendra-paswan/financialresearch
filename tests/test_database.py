from sqlalchemy import text
from sqlalchemy.orm import Session


def test_vector_extension_exists(db: Session) -> None:
    # The first migration enables pgvector, so this proves the migration ran on the test database
    extension = db.execute(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    ).scalar()

    assert extension == "vector"
