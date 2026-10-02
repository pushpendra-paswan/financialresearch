from sqlalchemy.orm import Session

from app.exceptions import ConflictError, NotFoundError
from app.models.users import User
from app.repositories import audit as audit_repository
from app.repositories import users as user_repository
from app.schemas.users import UserCreate
from app.security import hash_password


def create_user(db: Session, org_id: int, actor_user_id: int, data: UserCreate) -> User:
    # Emails are unique across all organizations, so this also catches other organizations' users
    email = data.email.lower()
    existing = user_repository.get_by_email(db, email)
    if existing:
        raise ConflictError("A user with this email already exists")

    # The new user always joins the caller's organization, never one chosen by the request
    user = user_repository.create(
        db,
        org_id=org_id,
        email=email,
        hashed_password=hash_password(data.password),
        role=data.role,
    )

    audit_repository.create(db, org_id, actor_user_id, action="user.create", entity_id=user.id)
    db.commit()
    return user


def list_users(db: Session, org_id: int) -> list[User]:
    return user_repository.list_by_org(db, org_id)


def get_user(db: Session, org_id: int, user_id: int) -> User:
    user = user_repository.get_by_id(db, org_id, user_id)

    # Same message whether the user does not exist or belongs to another organization,
    # so the existence of other organizations' users is not leaked
    if user is None:
        raise NotFoundError("User not found")
    return user
