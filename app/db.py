from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import load_config


def get_engine():
    config = load_config()
    return create_engine(config.db_url, pool_pre_ping=True)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


@contextmanager
def get_session() -> Iterator:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def run_migrations(sql_path: Path) -> None:
    engine = get_engine()
    raw_sql = sql_path.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in raw_sql.split(";") if stmt.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
