from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import engine
from app.schemas.auth import RegisterRequest
from app.services.auth import register


def test_vector_extension_exists(db: Session) -> None:
    # The first migration enables pgvector, so this proves the migration ran on the test database
    extension = db.execute(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    ).scalar()

    assert extension == "vector"


def test_service_commit_is_rolled_back_and_invisible_to_other_connections(db: Session) -> None:
    # register() calls db.commit(). With the savepoint setup in conftest.py that commit must
    # not reach the database for real, so a separate connection must not see the row
    register(
        db,
        RegisterRequest(
            organization_name="Isolation Test Org",
            email="iso@example.com",
            password="correct-horse-battery",
        ),
    )

    visible_to_service_session = db.execute(
        text("SELECT count(*) FROM organizations WHERE name = 'Isolation Test Org'")
    ).scalar()
    assert visible_to_service_session == 1

    with engine.connect() as other_connection:
        visible_to_other_connection = other_connection.execute(
            text("SELECT count(*) FROM organizations WHERE name = 'Isolation Test Org'")
        ).scalar()
    assert visible_to_other_connection == 0
