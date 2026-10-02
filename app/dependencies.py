from collections.abc import Generator

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.exceptions import ForbiddenError, UnauthorizedError
from app.models.users import User, UserRole
from app.redis_client import check_rate_limit
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


def require_editor(current_user: User = Depends(get_current_user)) -> User:
    # Viewers are read-only; admins and analysts can create and change organization data
    if current_user.role not in (UserRole.admin, UserRole.analyst):
        raise ForbiddenError("Admin or analyst role required")
    return current_user


def limit_auth(request: Request) -> None:
    # Brute-force protection for login and register: per client IP, one shared counter.
    # X-Forwarded-For is NOT read because any client can fake it.
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit("auth", client_ip, settings.RATE_LIMIT_AUTH_PER_MINUTE, 60)


def limit_user(current_user: User = Depends(get_current_user)) -> None:
    # Every protected route is limited per user. A request with a bad token fails with 401 in
    # get_current_user first, so it is never counted here.
    check_rate_limit("user", str(current_user.id), settings.RATE_LIMIT_API_PER_MINUTE, 60)
