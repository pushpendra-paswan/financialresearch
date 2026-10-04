from logging.config import fileConfig

from sqlalchemy import create_engine, pool

import app.models  # noqa: F401  (importing the package registers every model on Base.metadata)
from alembic import context
from app.config import settings
from app.database import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# The tables of the LangGraph Postgres checkpointer belong to the library and are created by a
# migration that calls the library's own setup(). They are not in our models, so autogenerate
# must not propose to drop them
CHECKPOINT_TABLES = {
    "checkpoint_migrations",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
}


def include_object(object, name, type_, reflected, compare_to) -> bool:
    return not (type_ == "table" and name in CHECKPOINT_TABLES)


def run_migrations_offline() -> None:
    # Emits SQL without connecting to the database
    context.configure(
        url=settings.DATABASE_URL,
        target_metadata=target_metadata,
        include_object=include_object,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # The URL comes from settings, never from alembic.ini
    connectable = create_engine(settings.DATABASE_URL, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, include_object=include_object
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
