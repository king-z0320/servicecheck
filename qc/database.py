from __future__ import annotations

import os
from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def database_url_from_env(variable: str = "DATABASE_URL") -> str:
    value = os.getenv(variable, "").strip()
    if not value:
        raise RuntimeError(f"{variable} is not configured")
    if not value.startswith("postgresql+psycopg://"):
        raise RuntimeError(f"{variable} must use postgresql+psycopg")
    return value


def create_database_engine(database_url: str | None = None) -> Engine:
    return create_engine(
        database_url or database_url_from_env(),
        pool_pre_ping=True,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
