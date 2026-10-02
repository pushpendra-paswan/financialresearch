from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.dependencies import get_current_user, get_db, limit_auth, limit_user
from app.models.users import User
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from app.schemas.users import UserResponse
from app.services import auth as auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=201,
    dependencies=[Depends(limit_auth)],
)
def register(data: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    return auth_service.register(db, data)


@router.post("/login", response_model=TokenResponse, dependencies=[Depends(limit_auth)])
def login(data: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    return auth_service.login(db, data)


@router.get("/me", response_model=UserResponse, dependencies=[Depends(limit_user)])
def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user
