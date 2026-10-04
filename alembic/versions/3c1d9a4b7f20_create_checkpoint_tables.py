"""create LangGraph checkpoint tables

Revision ID: 3c1d9a4b7f20
Revises: 7e5f97537b7e
Create Date: 2026-10-04 06:00:00.000000

The four tables belong to the library (langgraph-checkpoint-postgres), so this migration lets the
library create them with its own setup() instead of copying its SQL. setup() creates indexes with
CREATE INDEX CONCURRENTLY, which cannot run inside a transaction: autocommit_block() ends
Alembic's transaction first, and the saver uses its own autocommit connection. setup() is
idempotent (IF NOT EXISTS). A newer library version that adds checkpoint migrations needs a new
migration that calls setup() again; the app itself never calls it.
"""
from typing import Sequence, Union

from langgraph.checkpoint.postgres import PostgresSaver

from alembic import op
from app.config import settings

# revision identifiers, used by Alembic.
revision: str = '3c1d9a4b7f20'
down_revision: Union[str, Sequence[str], None] = '7e5f97537b7e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # psycopg wants a plain postgresql:// URL, not the SQLAlchemy "+psycopg" form
    url = settings.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://", 1)
    with op.get_context().autocommit_block():
        with PostgresSaver.from_conn_string(url) as saver:
            saver.setup()


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        "DROP TABLE IF EXISTS checkpoint_writes, checkpoint_blobs, checkpoints, "
        "checkpoint_migrations"
    )
