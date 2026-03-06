from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import load_config

_engine: Engine | None = None
_engine_lock = threading.Lock()

# ──────────────────────────────────────────────
# Slow Query Detection
# ──────────────────────────────────────────────
SLOW_QUERY_THRESHOLD: float = float(os.environ.get("SLOW_QUERY_THRESHOLD", "1.0"))
_slow_query_logger = logging.getLogger("stock_bot.slow_query")
_SLOW_QUERIES_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "slow_queries.jsonl"


def _record_slow_query(statement: str, duration: float) -> None:
    """Append a slow query record to artifacts/slow_queries.jsonl (JSONL format)."""
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(duration, 3),
        "statement": statement[:500].replace("\n", " "),
    }
    try:
        _SLOW_QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SLOW_QUERIES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # Never let logging failures crash the app


def _setup_slow_query_events(eng: Engine) -> None:
    """Register SQLAlchemy cursor-execute events to detect slow queries."""

    @event.listens_for(eng, "before_cursor_execute")
    def _before(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        conn.info.setdefault("_sq_times", []).append(time.perf_counter())

    @event.listens_for(eng, "after_cursor_execute")
    def _after(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        times = conn.info.get("_sq_times")
        if not times:
            return
        duration = time.perf_counter() - times.pop()
        if duration >= SLOW_QUERY_THRESHOLD:
            _slow_query_logger.warning(
                "[SLOW_QUERY] %.3fs | %s",
                duration,
                statement[:200].replace("\n", " "),
            )
            _record_slow_query(statement, duration)


def get_engine() -> Engine:
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        config = load_config()
        _engine = create_engine(
            config.db_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            pool_recycle=1800,
        )
        _setup_slow_query_events(_engine)
        return _engine


def _get_session_factory() -> sessionmaker:
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


@contextmanager
def get_session() -> Iterator[Session]:
    factory = _get_session_factory()
    session = factory()
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
