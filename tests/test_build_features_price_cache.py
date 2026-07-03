"""build_features 價格查詢走本地 parquet cache 的守門測試（2026-07-03 健檢發現 6）。

鎖定行為：
  1. cache 夠新且窗口列數與 MySQL 一致 → 用 parquet
  2. cache max_date < end_date（落後 MySQL）→ 回 None（fallback MySQL，絕不拿舊資料）
  3. 窗口列數不一致（歷史被回補）→ 回 None
  4. cache 不存在 → 回 None
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import skills.data_store as ds
from skills.build_features import _fetch_prices_from_cache


class _FakeSession:
    """回傳固定 count(*) 的假 session（只服務 BETWEEN count 查詢）。"""

    def __init__(self, count: int):
        self._count = count

    def execute(self, stmt, params=None):
        class _R:
            def scalar(self_inner):
                return self._count
        _R.scalar = lambda s: self._count  # noqa: E731
        return _R()


@pytest.fixture()
def price_parquet(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "stock_id": ["1101", "2330", "1101", "2330"],
        "trading_date": [date(2026, 1, 2), date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 3)],
        "open": [10.0, 500.0, 10.5, 505.0],
        "high": [10.1, 501.0, 10.6, 506.0],
        "low": [9.9, 499.0, 10.4, 504.0],
        "close": [10.0, 500.0, 10.5, 505.0],
        "volume": [1000, 2000, 1100, 2100],
    })
    p = tmp_path / "prices.parquet"
    df.to_parquet(p, index=False)
    monkeypatch.setattr(ds, "PRICES_PARQUET", p)
    return df


def test_cache_hit_when_fresh_and_counts_match(price_parquet):
    out = _fetch_prices_from_cache(_FakeSession(4), date(2026, 1, 2), date(2026, 1, 3))
    assert out is not None
    assert len(out) == 4
    assert list(out.columns) == ["stock_id", "trading_date", "open", "high", "low", "close", "volume"]


def test_fallback_when_cache_stale(price_parquet):
    """cache max=2026-01-03 < end_date=2026-01-06 → 絕不能拿舊資料。"""
    out = _fetch_prices_from_cache(_FakeSession(4), date(2026, 1, 2), date(2026, 1, 6))
    assert out is None


def test_fallback_when_row_count_mismatch(price_parquet):
    """MySQL 窗口列數 5 != parquet 4（歷史回補）→ fallback。"""
    out = _fetch_prices_from_cache(_FakeSession(5), date(2026, 1, 2), date(2026, 1, 3))
    assert out is None


def test_fallback_when_cache_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "PRICES_PARQUET", tmp_path / "nope.parquet")
    out = _fetch_prices_from_cache(_FakeSession(4), date(2026, 1, 2), date(2026, 1, 3))
    assert out is None
