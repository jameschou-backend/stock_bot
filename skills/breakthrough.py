"""突破確認進場模組 — 共用於 skills/backtest.py（walk-forward 回測）和 skills/daily_pick.py（生產選股）。

突破條件（任一成立）：
  條件一（價格突破）：收盤 > 過去20日最高收盤 AND 當日量 > 20日均量 × 1.5
  條件二（籌碼突破）：外資連續買超 ≥ 3 天 AND 收盤 > 20日均線
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import pandas as pd

# ── 突破條件常數（與 backtest.py 保持一致）──
LOOKBACK: int = 20               # rolling 視窗長度
VOLUME_SURGE_RATIO: float = 1.5  # 量比門檻（今日量 / 20日均量）
FOREIGN_STREAK_MIN: int = 3      # 外資連買最低天數


def precompute_stats(
    price_df: pd.DataFrame,
    lookback: int = LOOKBACK,
) -> pd.DataFrame:
    """一次性預計算所有股票的 rolling 突破指標（供 walk-forward 回測使用）。

    在主迴圈前預計算，避免每個再平衡期重算（117 次 rolling → 1 次）。

    Returns:
        DataFrame: stock_id / trading_date / close / volume /
                   close_max_20 / vol_avg_20 / ma_20
    """
    if price_df.empty:
        return pd.DataFrame(
            columns=["stock_id", "trading_date", "close", "volume",
                     "close_max_20", "vol_avg_20", "ma_20"]
        )

    df = price_df[["stock_id", "trading_date", "close", "volume"]].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["stock_id"] = df["stock_id"].astype(str)
    df = df.sort_values(["stock_id", "trading_date"])

    grp = df.groupby("stock_id", group_keys=False)
    df["close_max_20"] = grp["close"].transform(
        lambda x: x.shift(1).rolling(lookback, min_periods=lookback).max()
    )
    df["vol_avg_20"] = grp["volume"].transform(
        lambda x: x.shift(1).rolling(lookback, min_periods=lookback).mean()
    )
    df["ma_20"] = grp["close"].transform(
        lambda x: x.shift(1).rolling(lookback, min_periods=lookback).mean()
    )
    return df[["stock_id", "trading_date", "close", "volume",
               "close_max_20", "vol_avg_20", "ma_20"]]


def compute_breakthrough_map(
    candidate_sids: List[str],
    window_dates: List[date],
    bt_stats_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    rb_date: Optional[date] = None,
    volume_surge_ratio: float = VOLUME_SURGE_RATIO,
    foreign_streak_min: int = FOREIGN_STREAK_MIN,
) -> Dict[str, date]:
    """向量化批次計算所有候選股在視窗內的第一個突破日（供 walk-forward 回測使用）。

    接收預計算的 bt_stats_df，只需在視窗日期內做篩選，省去每期重算 rolling 的費用。

    Args:
        candidate_sids: 候選股票清單
        window_dates: 突破等待視窗日期（再平衡日後最多 breakthrough_max_wait 天）
        bt_stats_df: 由 precompute_stats() 產生的 rolling 統計表
        feat_df: 特徵表（須含 foreign_buy_consecutive_days 欄位）
        rb_date: 再平衡日（保留參數供呼叫端傳入，目前未使用）
        volume_surge_ratio: 量比門檻
        foreign_streak_min: 外資連買最低天數

    Returns:
        {stock_id: 最早突破日} — 無突破的股票不出現在 dict 中
    """
    if not candidate_sids or not window_dates or bt_stats_df.empty:
        return {}

    sids_set = set(str(s) for s in candidate_sids)
    window_set = set(window_dates)

    window_data = bt_stats_df[
        bt_stats_df["stock_id"].isin(sids_set)
        & bt_stats_df["trading_date"].isin(window_set)
    ].copy()

    if window_data.empty:
        return {}

    # ── 條件一：價格 + 成交量突破 ──
    window_data["cond1"] = (
        window_data["close_max_20"].notna()
        & (window_data["close_max_20"] > 0)
        & (window_data["close"] > window_data["close_max_20"])
        & window_data["vol_avg_20"].notna()
        & (window_data["vol_avg_20"] > 0)
        & (window_data["volume"] > window_data["vol_avg_20"] * volume_surge_ratio)
    )

    # ── 條件二：籌碼突破（外資連買 ≥ foreign_streak_min 天 + 收盤 > 20日均線）──
    # 使用 feat_df 中各 window 日期的 foreign_buy_consecutive_days。
    # 此為當日可觀察數據（進場決策日的實際外資籌碼狀態），不構成未來洩漏。
    if (
        feat_df is not None
        and not feat_df.empty
        and "foreign_buy_consecutive_days" in feat_df.columns
        and "stock_id" in feat_df.columns
        and "trading_date" in feat_df.columns
    ):
        wf = feat_df[
            feat_df["stock_id"].astype(str).isin(sids_set)
            & feat_df["trading_date"].isin(window_set)
        ][["stock_id", "trading_date", "foreign_buy_consecutive_days"]].copy()
        wf["stock_id"] = wf["stock_id"].astype(str)
        wf["foreign_buy_consecutive_days"] = pd.to_numeric(
            wf["foreign_buy_consecutive_days"], errors="coerce"
        ).fillna(0)
        window_data = window_data.merge(wf, on=["stock_id", "trading_date"], how="left")
        window_data["foreign_buy_consecutive_days"] = (
            window_data["foreign_buy_consecutive_days"].fillna(0)
        )
        window_data["cond2"] = (
            (window_data["foreign_buy_consecutive_days"] >= foreign_streak_min)
            & window_data["ma_20"].notna()
            & (window_data["ma_20"] > 0)
            & (window_data["close"] > window_data["ma_20"])
        )
    else:
        window_data["cond2"] = False

    window_data["breakthrough"] = window_data["cond1"] | window_data["cond2"]

    bt_rows = window_data[window_data["breakthrough"]][["stock_id", "trading_date"]].copy()
    if bt_rows.empty:
        return {}
    bt_rows = bt_rows.sort_values("trading_date")
    return bt_rows.groupby("stock_id")["trading_date"].first().to_dict()


def check_today(
    price_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    stock_ids: List[str],
    target_date: date,
    lookback: int = LOOKBACK,
    volume_surge_ratio: float = VOLUME_SURGE_RATIO,
    foreign_streak_min: int = FOREIGN_STREAK_MIN,
) -> Dict[str, Dict]:
    """檢查今日各股的突破狀態（供 daily_pick.py 生產流程使用）。

    與 compute_breakthrough_map() 不同，此函數只檢查「今天」是否符合突破條件，
    不涉及未來等待視窗，用於即時標記選股清單的可進場狀態。

    Args:
        price_df: 至少含 target_date 前 lookback 個交易日的價格資料
                  （欄位：stock_id, trading_date, close, volume）
        feature_df: 包含 stock_id 和 foreign_buy_consecutive_days 的特徵表
                    （可為 None 或空白，此時跳過條件二）
        stock_ids: 要檢查的股票清單
        target_date: 選股決策日
        lookback: rolling 視窗長度（預設 20 日）
        volume_surge_ratio: 量比門檻（預設 1.5）
        foreign_streak_min: 外資連買最低天數（預設 3 天）

    Returns:
        {stock_id: {
            "ready": bool,                  # 今日已突破，可立即進場
            "type": "price" | "institutional" | None,
            "days_waiting": 0,              # 今日開始等，固定為 0
            "close": float,
            "close_max_20": float,          # 前20日最高收盤（today 不含）
            "vol_today": float,
            "vol_avg_20": float,            # 前20日均量
            "ma_20": float,                 # 前20日均線
            "foreign_streak": int,          # 外資連續買超天數
            "pct_to_price_bt": float,       # 距條件一還差多少 %（>0 = 尚未突破）
            "vol_ratio": float,             # 今日量 / 20日均量
        }}
    """
    result: Dict[str, Dict] = {}

    if price_df.empty or not stock_ids:
        return result

    price_df = price_df.copy()
    price_df["stock_id"] = price_df["stock_id"].astype(str)
    price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
    price_df["volume"] = pd.to_numeric(price_df["volume"], errors="coerce")
    price_df["trading_date"] = pd.to_datetime(price_df["trading_date"]).dt.date

    # 建立 foreign_buy_consecutive_days 查找表（O(1) 逐股查詢）
    feat_map: Dict[str, float] = {}
    if (
        feature_df is not None
        and not feature_df.empty
        and "foreign_buy_consecutive_days" in feature_df.columns
        and "stock_id" in feature_df.columns
    ):
        for _, frow in feature_df.iterrows():
            sid = str(frow.get("stock_id", ""))
            val = frow.get("foreign_buy_consecutive_days", 0)
            feat_map[sid] = float(val) if pd.notna(val) else 0.0

    for sid in [str(s) for s in stock_ids]:
        stock_prices = price_df[price_df["stock_id"] == sid].sort_values("trading_date")

        # 今日資料（若 target_date 無資料，取最新一筆）
        today_row = stock_prices[stock_prices["trading_date"] == target_date]
        if today_row.empty:
            today_row = stock_prices.tail(1)
        if today_row.empty:
            result[sid] = {
                "ready": False, "type": None, "days_waiting": 0,
                "close": 0.0, "close_max_20": 0.0,
                "vol_today": 0.0, "vol_avg_20": 0.0, "ma_20": 0.0,
                "foreign_streak": 0, "pct_to_price_bt": float("nan"),
                "vol_ratio": float("nan"),
            }
            continue

        today_close = float(today_row["close"].iloc[0])
        today_volume = float(today_row["volume"].iloc[0])

        # 前 lookback 個交易日（不含今日）
        prior = stock_prices[stock_prices["trading_date"] < target_date].tail(lookback)

        close_max_20 = 0.0
        vol_avg_20 = 0.0
        ma_20 = 0.0

        if len(prior) >= lookback:
            close_max_20 = float(prior["close"].max())
            vol_avg_20 = float(prior["volume"].mean())
            ma_20 = float(prior["close"].mean())

        foreign_streak = feat_map.get(sid, 0.0)

        # 條件一：價格突破（收盤創20日新高 + 量放大1.5倍）
        cond1 = bool(
            close_max_20 > 0
            and today_close > close_max_20
            and vol_avg_20 > 0
            and today_volume > vol_avg_20 * volume_surge_ratio
        )

        # 條件二：籌碼突破（外資連買≥3天 + 收盤>MA20）
        cond2 = bool(
            foreign_streak >= foreign_streak_min
            and ma_20 > 0
            and today_close > ma_20
        )

        bt_type: Optional[str] = None
        if cond1:
            bt_type = "price"
        elif cond2:
            bt_type = "institutional"

        pct_to_price_bt = (
            float(close_max_20 / today_close - 1)
            if today_close > 0 and close_max_20 > 0
            else float("nan")
        )
        vol_ratio = float(today_volume / vol_avg_20) if vol_avg_20 > 0 else float("nan")

        result[sid] = {
            "ready": bool(cond1 or cond2),
            "type": bt_type,
            "days_waiting": 0,
            "close": today_close,
            "close_max_20": close_max_20,
            "vol_today": today_volume,
            "vol_avg_20": vol_avg_20,
            "ma_20": ma_20,
            "foreign_streak": int(foreign_streak),
            "pct_to_price_bt": pct_to_price_bt,
            "vol_ratio": vol_ratio,
        }

    return result
