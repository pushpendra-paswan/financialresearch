from sqlalchemy.orm import Session

from app.models.organizations import Organization


def create(db: Session, name: str) -> Organization:
    organization = Organization(name=name)
    db.add(organization)
    # Flush (not commit) so the id is available; the service owns the commit
    db.flush()
    return organization
