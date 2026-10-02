from sqlalchemy.orm import Session

from app.exceptions import ConflictError, UnauthorizedError
from app.models.users import UserRole
from app.repositories import audit as audit_repository
from app.repositories import organizations as organization_repository
from app.repositories import users as user_repository
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from app.security import create_access_token, hash_password, verify_password


def register(db: Session, data: RegisterRequest) -> TokenResponse:
    # Emails are stored lowercase and are unique across all organizations
    email = data.email.lower()
    existing = user_repository.get_by_email(db, email)
    if existing:
        raise ConflictError("A user with this email already exists")

    # Registering creates a new organization and its first user, who is its admin
    organization = organization_repository.create(db, data.organization_name)
    user = user_repository.create(
        db,
        org_id=organization.id,
        email=email,
        hashed_password=hash_password(data.password),
        role=UserRole.admin,
    )

    audit_repository.create(db, organization.id, user.id, action="auth.register", entity_id=user.id)

    # One commit for the organization, the user and the audit row
    db.commit()
    return TokenResponse(access_token=create_access_token(user.id))


def login(db: Session, data: LoginRequest) -> TokenResponse:
    email = data.email.lower()
    user = user_repository.get_by_email(db, email)

    # Unknown email and wrong password give the exact same error, so accounts cannot be discovered
    if user is None or not verify_password(data.password, user.hashed_password):
        raise UnauthorizedError("Invalid email or password")

    audit_repository.create(db, user.org_id, user.id, action="auth.login", entity_id=user.id)
    db.commit()
    return TokenResponse(access_token=create_access_token(user.id))
