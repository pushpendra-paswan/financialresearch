from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.users import User


def get_by_id(db: Session, org_id: int, user_id: int) -> User | None:
    # A user of another organization is treated as "not found"
    statement = select(User).where(User.id == user_id, User.org_id == org_id)
    return db.execute(statement).scalar_one_or_none()


def list_by_org(db: Session, org_id: int) -> list[User]:
    statement = select(User).where(User.org_id == org_id).order_by(User.id)
    return list(db.execute(statement).scalars().all())


def get_by_email(db: Session, email: str) -> User | None:
    # No org_id on purpose: login must find the user before the organization is known, and
    # emails are unique across all organizations. The caller passes a lowercase email.
    statement = select(User).where(User.email == email)
    return db.execute(statement).scalar_one_or_none()


def get_by_id_for_auth(db: Session, user_id: int) -> User | None:
    # No org_id on purpose: the token only contains the user id, and this lookup is how we
    # learn who the caller is (and therefore which organization they belong to).
    # Never use it to fetch a user on behalf of someone else.
    statement = select(User).where(User.id == user_id)
    return db.execute(statement).scalar_one_or_none()


def create(db: Session, org_id: int, email: str, hashed_password: str, role: str) -> User:
    user = User(org_id=org_id, email=email, hashed_password=hashed_password, role=role)
    db.add(user)
    db.flush()
    return user
