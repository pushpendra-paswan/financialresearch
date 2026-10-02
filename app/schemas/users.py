from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.models.users import UserRole
from app.schemas.auth import check_password_bytes


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    role: UserRole

    _check_password = field_validator("password")(check_password_bytes)


class UserResponse(BaseModel):
    # There is no hashed_password field here on purpose, so it can never be returned
    model_config = ConfigDict(from_attributes=True)

    id: int
    org_id: int
    email: EmailStr
    role: UserRole
    created_at: datetime
