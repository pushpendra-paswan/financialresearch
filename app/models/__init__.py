# Every model file is imported here so Alembic's autogenerate sees all tables
from app.models import (  # noqa: F401
    agent,
    alerts,
    audit,
    chat,
    chunks,
    companies,
    filings,
    financials,
    ingestion,
    notifications,
    organizations,
    prices,
    reports,
    users,
    watchlists,
)
