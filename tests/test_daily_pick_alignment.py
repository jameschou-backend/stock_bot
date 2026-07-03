"""P0 回歸測試：daily_pick 特徵對齊（2026-07-03 健檢 P0-1）。

歷史 bug：_choose_pick_date 內用 merge 做流動性過濾，merge 會把 index 重置成
RangeIndex(0..k-1)；run() 再以 chosen_df.index 對 feature_df 做 .loc 對齊，
結果取到「11 天候選特徵矩陣的前 k 列」——模型用別檔股票、別的日期的特徵打分。

此測試直接呼叫 _choose_pick_date，驗證回傳的 chosen_df.index 仍是
原始 feature_df 的列標籤（可用來正確 .loc 回原矩陣）。
"""

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from skills.daily_pick import _choose_pick_date


def _build_inputs():
    """建構跨 2 個交易日、多檔股票的候選矩陣（模擬 fallback 視窗）。

    列順序刻意讓「最舊日期在前」，重現歷史 bug 的觸發條件：
    位置 index 0..k-1 會指到舊日期的列。
    """
    d_old = date(2026, 6, 17)
    d_new = date(2026, 7, 2)
    rows = [
        # (index 位置隱含 0..5)
        {"stock_id": "2059", "trading_date": d_old, "amt_20": 4_150_105_720.75},
        {"stock_id": "8071", "trading_date": d_old, "amt_20": 999.0},
        {"stock_id": "1101", "trading_date": d_old, "amt_20": 888.0},
        {"stock_id": "2059", "trading_date": d_new, "amt_20": 111.0},
        {"stock_id": "8071", "trading_date": d_new, "amt_20": 191_469_494.0},
        {"stock_id": "1101", "trading_date": d_new, "amt_20": 777.0},
    ]
    feature_df = pd.DataFrame(rows)

    price_rows = []
    for d in (d_old, d_new):
        for sid in ("2059", "8071", "1101"):
            # 足量歷史列，避免 apply_liquidity_filter 的 min_records 保護剔除
            for k in range(12):
                price_rows.append(
                    {
                        "stock_id": sid,
                        "trading_date": d,
                        "close": 100.0,
                        "volume": 1_000_000,
                        "amount": 5e9,
                    }
                )
    price_df = pd.DataFrame(price_rows).drop_duplicates(
        subset=["stock_id", "trading_date"]
    )
    # 補足每檔 >=10 筆歷史（不同日期）供流動性過濾
    hist_rows = []
    for sid in ("2059", "8071", "1101"):
        for day in pd.date_range("2026-06-01", periods=12, freq="B"):
            hist_rows.append(
                {
                    "stock_id": sid,
                    "trading_date": day.date(),
                    "close": 100.0,
                    "volume": 1_000_000,
                    "amount": 5e9,
                }
            )
    price_df = pd.concat([pd.DataFrame(hist_rows), price_df], ignore_index=True)

    config = SimpleNamespace(min_avg_turnover=0.0, fallback_days=10)
    return feature_df, price_df, [d_new, d_old], config


def test_choose_pick_date_preserves_original_index():
    """chosen_df.index 必須是原始 feature_df 的列標籤（非重置後的 0..k-1）。"""
    feature_df, price_df, candidate_dates, config = _build_inputs()

    chosen_date, chosen_df, _logs = _choose_pick_date(
        candidate_dates, feature_df, price_df, topn=2, config=config, fallback_days=10
    )

    assert chosen_date == date(2026, 7, 2)
    assert not chosen_df.empty
    # 最新日的列在原矩陣中的標籤是 3/4/5；歷史 bug 會回傳 0..k-1
    assert set(chosen_df.index).issubset({3, 4, 5}), (
        f"chosen_df.index={list(chosen_df.index)} 不是原始列標籤 —— "
        "index 錯位會使模型用錯誤股票/日期的特徵打分（P0-1 迴歸）"
    )


def test_choose_pick_date_loc_alignment_roundtrip():
    """模擬 run() 的對齊流程：feature_df.loc[chosen_df.index] 必須取回同 (stock, date) 的特徵。"""
    feature_df, price_df, candidate_dates, config = _build_inputs()

    _date, chosen_df, _logs = _choose_pick_date(
        candidate_dates, feature_df, price_df, topn=2, config=config, fallback_days=10
    )

    realigned = feature_df.loc[chosen_df.index]
    for label in chosen_df.index:
        assert realigned.loc[label, "stock_id"] == chosen_df.loc[label, "stock_id"]
        assert realigned.loc[label, "trading_date"] == chosen_df.loc[label, "trading_date"]
        # 特徵值必須是「該股該日」的值（歷史 bug：8071 拿到 2059 在 6/17 的 amt_20）
        sid = chosen_df.loc[label, "stock_id"]
        expected = feature_df[
            (feature_df["stock_id"] == sid)
            & (feature_df["trading_date"] == date(2026, 7, 2))
        ]["amt_20"].iloc[0]
        assert realigned.loc[label, "amt_20"] == pytest.approx(expected)
