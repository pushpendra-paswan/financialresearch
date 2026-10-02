from pydantic import BaseModel, EmailStr, Field, field_validator


def check_password_bytes(password: str) -> str:
    # bcrypt ignores or rejects anything past 72 bytes, so reject it up front.
    # Kept as a function because the register, login and create-user schemas all use it.
    if len(password.encode("utf-8")) > 72:
        raise ValueError("Password must be at most 72 bytes when UTF-8 encoded")
    return password


class RegisterRequest(BaseModel):
    organization_name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    password: str = Field(min_length=8)

    _check_password = field_validator("password")(check_password_bytes)


class LoginRequest(BaseModel):
    email: EmailStr
    # Only the upper limit is checked here: a longer password could never match, and
    # bcrypt would raise an error on it
    password: str

    _check_password = field_validator("password")(check_password_bytes)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
