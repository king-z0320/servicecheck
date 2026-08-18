from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from qc.database import Base
import qc.orm_models  # noqa: F401 - registers all tables on Base.metadata


config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def configured_url() -> str:
    value = config.get_main_option("sqlalchemy.url") or os.getenv("DATABASE_URL", "")
    value = value.strip()
    if not value:
        raise RuntimeError("DATABASE_URL is not configured for Alembic")
    if not value.startswith("postgresql+psycopg://"):
        raise RuntimeError("Alembic requires a postgresql+psycopg URL")
    return value


def run_migrations_offline() -> None:
    context.configure(
        url=configured_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = configured_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
