"""Durable shared FinMind budget. Sponsor hard cap 6,000/h, 10% reserve.

All local checkouts/workers share one ledger; usage elsewhere is not observable.
"""
from __future__ import annotations

import math
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SPONSOR_LIMIT = 6000
WINDOW_SECONDS = 3600


def state_path() -> Path:
    return Path(os.environ.get(
        "FINMIND_STATE_PATH", str(Path.home() / ".cache/stock-bot/finmind.sqlite3")
    )).expanduser().resolve()


@dataclass
class RateLimitStats:
    requests_in_window: int
    window_start_time: float
    total_requests: int
    total_wait_time: float
    effective_limit: int
    remaining_requests: int
    retry_after_seconds: float
    cooldown_until: float


class RateLimiter:
    def __init__(self, requests_per_hour: int = 6000, buffer_percent: float = 0.1,
                 *, path: Path | str | None = None):
        if requests_per_hour <= 0 or not 0 <= buffer_percent < 1:
            raise ValueError("requests_per_hour must be positive; buffer_percent must be in [0, 1)")
        self._requests_per_hour = min(int(requests_per_hour), SPONSOR_LIMIT)
        self._effective_limit = max(1, int(self._requests_per_hour * (1 - buffer_percent)))
        self.path = Path(path) if path is not None else state_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as con:
            con.execute("CREATE TABLE IF NOT EXISTS requests (requested_at REAL NOT NULL)")
            con.execute("CREATE INDEX IF NOT EXISTS requests_time ON requests(requested_at)")
            con.execute("""CREATE TABLE IF NOT EXISTS budget (
                id INTEGER PRIMARY KEY CHECK(id=1), cooldown_until REAL NOT NULL DEFAULT 0,
                total_requests INTEGER NOT NULL DEFAULT 0, total_wait REAL NOT NULL DEFAULT 0,
                active_limit INTEGER NOT NULL)""")
            con.execute("INSERT OR IGNORE INTO budget(id, active_limit) VALUES(1, ?)",
                        (self._effective_limit,))

    @contextmanager
    def _connection(self):
        con = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            # A failed ledger is an explicit error, never an unmetered request.
            con.execute("BEGIN IMMEDIATE")
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    @property
    def requests_per_hour(self) -> int:
        return self._requests_per_hour

    @property
    def effective_limit(self) -> int:
        return self._effective_limit

    def _snapshot(self, con, now: float) -> RateLimitStats:
        con.execute("DELETE FROM requests WHERE requested_at <= ?", (now - WINDOW_SECONDS,))
        count, oldest = con.execute("SELECT count(*), min(requested_at) FROM requests").fetchone()
        cooldown, total, waited, active_limit = con.execute(
            "SELECT cooldown_until, total_requests, total_wait, active_limit FROM budget WHERE id=1"
        ).fetchone()
        # A stricter caller cannot be undone by a later default-config worker in this window.
        limit = min(active_limit, self._effective_limit) if count else self._effective_limit
        con.execute("UPDATE budget SET active_limit=? WHERE id=1", (limit,))
        delay = max(0.0, cooldown - now)
        if count >= limit:
            release_at = con.execute(
                "SELECT requested_at FROM requests ORDER BY requested_at LIMIT 1 OFFSET ?",
                (count - limit,),
            ).fetchone()[0] + WINDOW_SECONDS
            delay = max(delay, release_at - now)
        return RateLimitStats(count, oldest or now, total, waited, limit,
                              max(0, limit - count), delay, cooldown)

    def acquire(self, timeout: float | None = 0) -> bool:
        """Reserve before HTTP; default fails fast, explicit timeout permits bounded waiting."""
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be non-negative and finite")
        started = time.monotonic()
        while True:
            with self._connection() as con:
                now = time.time()
                stats = self._snapshot(con, now)
                if stats.remaining_requests > 0 and stats.retry_after_seconds <= 0:
                    con.execute("INSERT INTO requests VALUES(?)", (now,))
                    con.execute("UPDATE budget SET total_requests=total_requests+1 WHERE id=1")
                    return True
            left = None if timeout is None else timeout - (time.monotonic() - started)
            if left is not None and left <= 0:
                return False
            delay = min(max(stats.retry_after_seconds, 0.01), 1.0)
            if left is not None:
                delay = min(delay, left)
            before = time.monotonic()
            time.sleep(delay)
            with self._connection() as con:
                con.execute("UPDATE budget SET total_wait=total_wait+? WHERE id=1",
                            (time.monotonic() - before,))

    def defer(self, seconds: float = WINDOW_SECONDS) -> None:
        """Share a provider 402/429 cooldown with every worker, without retry storms."""
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("cooldown must be non-negative and finite")
        with self._connection() as con:
            con.execute("UPDATE budget SET cooldown_until=max(cooldown_until, ?) WHERE id=1",
                        (time.time() + seconds,))

    def get_stats(self) -> RateLimitStats:
        with self._connection() as con:
            return self._snapshot(con, time.time())

    def remaining_requests(self) -> int:
        stats = self.get_stats()
        return 0 if stats.cooldown_until > time.time() else stats.remaining_requests


def get_rate_limiter(requests_per_hour: int = 6000) -> RateLimiter:
    configured = int(os.environ.get("FINMIND_REQUESTS_PER_HOUR", SPONSOR_LIMIT))
    return RateLimiter(min(requests_per_hour, configured))


def reset_global_limiter() -> None:
    """Compatibility hook. Restarting a worker must NEVER erase persisted usage."""
