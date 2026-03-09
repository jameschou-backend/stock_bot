# NOT IN USE — 10年 walk-forward 驗證顯示 regime switching 在 sideways 期間（35%）造成
# 長期損耗（超額 -0.21%/期），整體 10y 表現劣於原始模型。保留供未來研究。
# 最後驗證結果：24m +24.71% 超額（2024-2026 熊市有效），10y +3.18% 超額（不推薦長期使用）。
"""市場環境（regime）判斷模組：在模型外部判斷市場狀態，驅動策略切換。

判斷規則（三態）：
  bull:     市場等權指數 > 200MA  AND  60日報酬 > 0  AND  20日報酬 > -5%
  bear:     市場等權指數 < 200MA  AND  60日報酬 < -10%
  sideways: 其餘

5日緩衝（majority vote）：取最近 buffer_days 天的眾數，避免頻繁切換。

使用方式：
  # 回測（批量預計算）
  regime_map = precompute_regimes(price_df, rebalance_dates)

  # 即時選股（每日判斷）
  regime = detect_regime_from_mkt_series(mkt_avg_close_series)
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ── 核心：向量化 regime 計算 ────────────────────────────────────────────


def _compute_regime_series(mkt: pd.Series, buffer_days: int = 5) -> pd.Series:
    """向量化計算每日 regime（含 5 日 majority vote 緩衝）。

    Args:
        mkt: 以 date 為 index 的市場等權均價 Series（升序）
        buffer_days: 多數決緩衝天數

    Returns:
        pd.Series of "bull" / "bear" / "sideways"，index 同 mkt
    """
    if len(mkt) < 2:
        return pd.Series("sideways", index=mkt.index, dtype=str)

    ma200 = mkt.rolling(200, min_periods=40).mean()
    ret60 = mkt.pct_change(60)
    ret20 = mkt.pct_change(20)
    above_200ma = mkt > ma200  # NaN → False（fallback sideways）

    # 向量化三態判斷
    bull_mask = above_200ma.fillna(False) & (ret60.fillna(0) > 0.0) & (ret20.fillna(0) > -0.05)
    bear_mask = (~above_200ma.fillna(True)) & (ret60.fillna(0) < -0.10)

    raw = pd.Series("sideways", index=mkt.index, dtype=object)
    raw[bull_mask] = "bull"
    raw[bear_mask] = "bear"

    return _apply_buffer(raw, buffer_days=buffer_days)


def _apply_buffer(raw: pd.Series, buffer_days: int = 5) -> pd.Series:
    """5日 majority vote 緩衝（向量化實作）。

    encode: bull=1, sideways=0, bear=-1
    rolling mode: 取 buffer_days 天內出現最多次的值
    """
    encode = {"bull": 1.0, "sideways": 0.0, "bear": -1.0}
    decode = {1: "bull", 0: "sideways", -1: "bear"}
    encoded = raw.map(encode).astype(float)

    def _rolling_mode(arr: np.ndarray) -> float:
        """返回最高頻率的編碼值：1(bull), 0(sideways), -1(bear)"""
        # arr 值為 {-1, 0, 1}，shift +1 → {0, 1, 2} 供 bincount
        int_arr = np.round(arr + 1).astype(int)
        int_arr = np.clip(int_arr, 0, 2)
        counts = np.bincount(int_arr, minlength=3)
        return float(np.argmax(counts)) - 1.0  # {0→-1, 1→0, 2→1}

    buffered = encoded.rolling(window=buffer_days, min_periods=1).apply(
        _rolling_mode, raw=True
    )
    result = buffered.round().astype(int).map(decode)
    return result.fillna("sideways")


# ── 公開 API ────────────────────────────────────────────────────────────


def precompute_regimes(
    price_df: pd.DataFrame,
    trading_dates: List[date],
    buffer_days: int = 5,
) -> Dict[date, str]:
    """批量預計算所有 trading_dates 的 regime（bull/bear/sideways）。

    供 backtest.py walk-forward 迴圈使用：先批量計算，避免迴圈內重複運算。

    Args:
        price_df: 含 trading_date, stock_id, close 的 raw_prices DataFrame
        trading_dates: 需要計算 regime 的日期清單（通常是再平衡日）
        buffer_days: 多數決緩衝天數（預設 5）

    Returns:
        Dict {date: "bull"/"bear"/"sideways"}
    """
    if price_df.empty or not trading_dates:
        return {d: "sideways" for d in trading_dates}

    # 建立等權市場指數（四碼台股平均收盤價）
    df = price_df.copy()
    df["stock_id"] = df["stock_id"].astype(str)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df = df[df["stock_id"].str.fullmatch(r"\d{4}", na=False)]
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    if df.empty:
        return {d: "sideways" for d in trading_dates}

    mkt = df.groupby("trading_date")["close"].mean().sort_index()
    if len(mkt) < 2:
        return {d: "sideways" for d in trading_dates}

    regime_series = _compute_regime_series(mkt, buffer_days=buffer_days)
    regime_by_date: Dict[date, str] = {idx: str(val) for idx, val in regime_series.items()}

    sorted_dates = sorted(regime_by_date.keys())
    result: Dict[date, str] = {}
    for d in trading_dates:
        if d in regime_by_date:
            result[d] = regime_by_date[d]
        else:
            # 往前找最近有資料的日期（backward fill）
            candidates = [x for x in sorted_dates if x <= d]
            result[d] = regime_by_date[candidates[-1]] if candidates else "sideways"
    return result


def detect_regime_from_mkt_series(
    mkt_avg_close: pd.Series,
    buffer_days: int = 5,
) -> str:
    """從市場均價 Series 判斷當日 regime（供 daily_pick.py 即時使用）。

    Args:
        mkt_avg_close: 以 date 為 index 的市場等權均價 Series（升序）
        buffer_days: 多數決緩衝天數

    Returns:
        "bull" / "bear" / "sideways"
    """
    if mkt_avg_close is None or len(mkt_avg_close) < 2:
        return "sideways"

    s = mkt_avg_close.dropna().sort_index().astype(float)
    if len(s) < 2:
        return "sideways"

    regime_series = _compute_regime_series(s, buffer_days=buffer_days)
    if regime_series.empty:
        return "sideways"
    return str(regime_series.iloc[-1])
