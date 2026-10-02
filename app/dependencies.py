from collections.abc import Generator

from sqlalchemy.orm import Session

from app.database import SessionLocal


def get_db() -> Generator[Session, None, None]:
    # One session per request, always closed afterwards
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
