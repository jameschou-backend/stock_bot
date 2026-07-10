"""TAIEX 發行量加權股價報酬指數（TR）對照臂：解析 / 快取增量 / vs_taiex_tr 計算。

全部離線：網路層以注入的 fetch_month_fn 取代（真實 endpoint 格式已於
skills/taiex_tr.py docstring 記錄並實測驗證）。
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from skills.taiex_tr import (
    TaiexTRError,
    _months_between,
    compute_vs_taiex_tr,
    load_taiex_tr,
    parse_mfi94u_payload,
    update_taiex_tr_cache,
)


# ── 1. payload 解析 ─────────────────────────────────────────────────────────


def test_parse_ok_payload_roc_dates_and_commas():
    payload = {
        "stat": "OK",
        "fields": ["日　期", "發行量加權股價報酬指數"],
        "data": [["113/01/02", "38,475.17"], ["113/01/03", "37,840.62"]],
    }
    df = parse_mfi94u_payload(payload)
    assert df["date"].tolist() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert df["tr_index"].tolist() == [38475.17, 37840.62]


def test_parse_no_data_stat_returns_empty():
    for stat in ("很抱歉，沒有符合條件的資料!", "查詢日期大於今日，請重新查詢!"):
        assert parse_mfi94u_payload({"stat": stat}).empty


def test_parse_unknown_stat_raises():
    with pytest.raises(TaiexTRError):
        parse_mfi94u_payload({"stat": "系統維護中"})


def test_parse_skips_missing_values_without_fabricating():
    payload = {"stat": "OK", "data": [["113/01/02", "--"], ["113/01/03", "37,840.62"]]}
    df = parse_mfi94u_payload(payload)
    assert len(df) == 1
    assert df["date"].tolist() == [date(2024, 1, 3)]


def test_months_between():
    assert _months_between(date(2024, 11, 15), date(2025, 2, 3)) == [
        (2024, 11), (2024, 12), (2025, 1), (2025, 2),
    ]
    assert _months_between(date(2025, 3, 1), date(2025, 3, 31)) == [(2025, 3)]
    assert _months_between(date(2025, 4, 1), date(2025, 3, 1)) == []


# ── 2. 快取增量更新 ─────────────────────────────────────────────────────────


def _fake_month_df(year: int, month: int, days=(3, 15)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date(year, month, d) for d in days],
            "tr_index": [10000.0 + year + month + d for d in days],
        }
    )


def test_cache_initial_backfill_and_incremental(tmp_path):
    cache = tmp_path / "taiex_tr.parquet"
    fetched: list = []

    def fake_fetch(y, m):
        fetched.append((y, m))
        return _fake_month_df(y, m)

    # 首次：抓 2024-01 ~ 2024-03 共 3 個月
    df = update_taiex_tr_cache(
        date(2024, 1, 1), date(2024, 3, 31), cache,
        fetch_month_fn=fake_fetch, request_delay=0,
    )
    assert fetched == [(2024, 1), (2024, 2), (2024, 3)]
    assert len(df) == 6
    assert cache.exists()

    # 增量：延伸到 2024-05 → 只補 max 月（2024-03，可能原本抓到月中）+ 04 + 05，
    # 內部月（01、02）不重抓
    fetched.clear()
    df2 = update_taiex_tr_cache(
        date(2024, 1, 1), date(2024, 5, 31), cache,
        fetch_month_fn=fake_fetch, request_delay=0,
    )
    assert fetched == [(2024, 3), (2024, 4), (2024, 5)]
    assert len(df2) == 10  # 5 個月 × 2 天，重抓月以 date 去重
    assert df2["date"].is_monotonic_increasing
    assert not df2["date"].duplicated().any()

    # 已涵蓋：不再抓
    fetched.clear()
    df3 = update_taiex_tr_cache(
        date(2024, 1, 1), date(2024, 4, 30), cache,
        fetch_month_fn=fake_fetch, request_delay=0,
    )
    assert fetched == []
    assert len(df3) == 10


def test_cache_partial_last_month_refetched_with_new_rows(tmp_path):
    """max 月當初只抓到月中；下次增量重抓該月應補齊後半月列。"""
    cache = tmp_path / "taiex_tr.parquet"

    update_taiex_tr_cache(
        date(2024, 1, 1), date(2024, 1, 31), cache,
        fetch_month_fn=lambda y, m: _fake_month_df(y, m, days=(3,)),  # 月中快照：僅 1 列
        request_delay=0,
    )
    assert len(load_taiex_tr(cache)) == 1

    df = update_taiex_tr_cache(
        date(2024, 1, 1), date(2024, 1, 31), cache,
        fetch_month_fn=lambda y, m: _fake_month_df(y, m, days=(3, 15, 30)),  # 全月
        request_delay=0,
    )
    assert df["date"].tolist() == [date(2024, 1, 3), date(2024, 1, 15), date(2024, 1, 30)]


def test_cache_repairs_missing_interior_month(tmp_path):
    """歷史缺月自動修復（presence-based 覆蓋判定）。

    2026-07-10 首次回補實測：TWSE payload 汙染導致 9 個歷史月被 silent 跳過；
    span-based 判定會把缺月誤認為已覆蓋、永不補抓。此測試鎖定修復行為。
    """
    cache = tmp_path / "taiex_tr.parquet"
    # 快取有 2024-01 與 2024-03，缺 2024-02（模擬汙染跳過）
    pd.concat([_fake_month_df(2024, 1), _fake_month_df(2024, 3)]).to_parquet(cache, index=False)

    fetched: list = []

    def fake_fetch(y, m):
        fetched.append((y, m))
        return _fake_month_df(y, m)

    df = update_taiex_tr_cache(
        date(2024, 1, 1), date(2024, 3, 31), cache,
        fetch_month_fn=fake_fetch, request_delay=0,
    )
    # 缺月 2024-02 補抓；max 月 2024-03 重抓；2024-01 不重抓
    assert fetched == [(2024, 2), (2024, 3)]
    assert {(d.year, d.month) for d in df["date"]} == {(2024, 1), (2024, 2), (2024, 3)}


def test_cache_drops_wrong_month_payload_rows(tmp_path):
    """fetcher 回傳非請求月份的列（server payload 汙染）→ 剔除，不入快取。"""
    cache = tmp_path / "taiex_tr.parquet"

    def poisoned_fetch(y, m):
        # 請求 (y, m) 卻回傳上個月的資料 + 一列正確資料
        prev_y, prev_m = (y - 1, 12) if m == 1 else (y, m - 1)
        return pd.concat(
            [_fake_month_df(prev_y, prev_m), _fake_month_df(y, m, days=(5,))],
            ignore_index=True,
        )

    df = update_taiex_tr_cache(
        date(2024, 2, 1), date(2024, 2, 29), cache,
        fetch_month_fn=poisoned_fetch, request_delay=0,
    )
    assert {(d.year, d.month) for d in df["date"]} == {(2024, 2)}
    assert df["date"].tolist() == [date(2024, 2, 5)]


def test_cache_empty_when_all_fetch_empty_raises(tmp_path):
    cache = tmp_path / "taiex_tr.parquet"
    with pytest.raises(TaiexTRError):
        update_taiex_tr_cache(
            date(2024, 1, 1), date(2024, 2, 28), cache,
            fetch_month_fn=lambda y, m: pd.DataFrame(columns=["date", "tr_index"]),
            request_delay=0,
        )


# ── 3. compute_vs_taiex_tr ──────────────────────────────────────────────────


def _equity_curve(start: date, end: date, eq_start: float, eq_end: float) -> list:
    return [
        {"date": start.isoformat(), "equity": eq_start},
        {"date": end.isoformat(), "equity": eq_end},
    ]


def test_vs_taiex_tr_annualized_math(tmp_path):
    """TR 10000→12100（2 年 +21% → 年化 10%）；策略 10000→14400（年化 20%）。"""
    cache = tmp_path / "taiex_tr.parquet"
    pd.DataFrame(
        {
            "date": [date(2022, 1, 3), date(2024, 1, 2)],
            "tr_index": [10000.0, 12100.0],
        }
    ).to_parquet(cache, index=False)

    out = compute_vs_taiex_tr(
        _equity_curve(date(2022, 1, 3), date(2024, 1, 2), 10000.0, 14400.0),
        cache,
        allow_fetch=False,
    )
    assert out is not None
    assert out["taiex_tr_total_return"] == pytest.approx(0.21, abs=1e-4)
    assert out["taiex_tr_annualized_return"] == pytest.approx(0.10, abs=5e-3)
    assert out["strategy_annualized_return"] == pytest.approx(0.20, abs=5e-3)
    # excess 由未捨入值計算後再 round(4)，與「捨入後相減」差 ≤ 捨入粒度
    assert out["excess_annualized_vs_tr"] == pytest.approx(
        out["strategy_annualized_return"] - out["taiex_tr_annualized_return"], abs=2e-4
    )
    assert out["window"]["tr_start_date"] == "2022-01-03"
    assert out["window"]["tr_end_date"] == "2024-01-02"


def test_vs_taiex_tr_returns_none_on_missing_cache_and_failed_fetch(tmp_path):
    """fetch 失敗 + 無快取 → None（不 raise、不阻斷回測）。"""

    def failing_fetch(y, m):
        raise TaiexTRError("simulated network failure")

    out = compute_vs_taiex_tr(
        _equity_curve(date(2022, 1, 3), date(2024, 1, 2), 10000.0, 14400.0),
        tmp_path / "missing.parquet",
        allow_fetch=True,
        fetch_month_fn=failing_fetch,
        request_delay=0,
    )
    assert out is None


def test_vs_taiex_tr_returns_none_when_cache_too_stale(tmp_path):
    """快取最新日落後回測終點 >45 天 → 對照失真，回傳 None。"""
    cache = tmp_path / "taiex_tr.parquet"
    pd.DataFrame(
        {"date": [date(2022, 1, 3), date(2022, 6, 1)], "tr_index": [10000.0, 10500.0]}
    ).to_parquet(cache, index=False)

    out = compute_vs_taiex_tr(
        _equity_curve(date(2022, 1, 3), date(2024, 1, 2), 10000.0, 14400.0),
        cache,
        allow_fetch=False,
    )
    assert out is None


def test_vs_taiex_tr_returns_none_on_short_curve(tmp_path):
    assert compute_vs_taiex_tr([], tmp_path / "x.parquet", allow_fetch=False) is None
    assert (
        compute_vs_taiex_tr(
            [{"date": "2024-01-02", "equity": 10000.0}],
            tmp_path / "x.parquet",
            allow_fetch=False,
        )
        is None
    )
