from collections.abc import Generator

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.exceptions import ForbiddenError, UnauthorizedError
from app.models.users import User, UserRole
from app.repositories import users as user_repository
from app.security import decode_access_token

# auto_error=False so a missing token raises our UnauthorizedError (same format as the others)
bearer_scheme = HTTPBearer(auto_error=False)


def get_db() -> Generator[Session, None, None]:
    # One session per request, always closed afterwards
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise UnauthorizedError("Not authenticated")

    user_id = decode_access_token(credentials.credentials)

    # The user is loaded on every request, so role changes and deleted users apply immediately
    user = user_repository.get_by_id_for_auth(db, user_id)
    if user is None:
        raise UnauthorizedError("Invalid or expired token")
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != UserRole.admin:
        raise ForbiddenError("Admin role required")
    return current_user
