from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from app.config import settings
from app.exceptions import UnauthorizedError


def hash_password(password: str) -> str:
    # bcrypt only accepts up to 72 bytes; the schemas enforce that before we get here
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(user_id: int) -> str:
    # The token holds only the user id and an expiry. Role and organization are read from the
    # database on every request, so a role change applies immediately.
    # PyJWT requires "sub" to be a string.
    expires_at = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expires_at}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> int:
    # Returns the user id. Invalid, expired and badly signed tokens all give the same error.
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["sub", "exp"]},
        )
        return int(payload["sub"])
    except (jwt.InvalidTokenError, ValueError) as error:
        raise UnauthorizedError("Invalid or expired token") from error
