"""Database engine and sessions (one cached engine per URL)."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


@lru_cache
def get_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def session_factory(url: str) -> sessionmaker[Session]:
    return sessionmaker(get_engine(url), expire_on_commit=False)
