from __future__ import annotations

from datetime import date
from typing import Dict

import pandas as pd

from app.db import get_session
from app.models import RawInstitutional, RawPrice
from skills.regime import get_regime_detector


def load_price_df(start_date: date, end_date: date) -> pd.DataFrame:
    with get_session() as session:
        q = (
            session.query(
                RawPrice.stock_id,
                RawPrice.trading_date,
                RawPrice.open,
                RawPrice.high,
                RawPrice.low,
                RawPrice.close,
                RawPrice.volume,
                RawInstitutional.foreign_net,
                RawInstitutional.trust_net,
                RawInstitutional.dealer_net,
            )
            .outerjoin(
                RawInstitutional,
                (RawPrice.stock_id == RawInstitutional.stock_id)
                & (RawPrice.trading_date == RawInstitutional.trading_date),
            )
            .filter(RawPrice.trading_date.between(start_date, end_date))
        )
        df = pd.read_sql(q.statement, session.get_bind())
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["foreign_net", "trust_net", "dealer_net"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["stock_id", "trading_date"])
    outputs = []
    for _, g in df.groupby("stock_id"):
        g = g.copy()
        close = g["close"]
        volume = g["volume"]
        g["ret_20"] = close.pct_change(20)
        g["ret_5"] = close.pct_change(5)
        g["ma_20"] = close.rolling(20).mean()
        g["ma_60"] = close.rolling(60).mean()
        g["volume_20"] = volume.rolling(20).mean()
        g["volume_60_mean"] = volume.rolling(60).mean()
        g["high_60"] = close.rolling(60).max()
        g["low_20"] = close.rolling(20).min()
        g["volume_max_10"] = volume.rolling(10).max()

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        g["rsi_14"] = (100 - (100 / (1 + rs))).clip(0, 100)

        ma = close.rolling(20).mean()
        std = close.rolling(20).std()
        g["bb_lower"] = ma - 2 * std

        g["vol_60"] = close.pct_change(1).rolling(60).std()
        g["avg_turnover_20"] = (close * volume).rolling(20).mean() / 1e8

        # 法人籌碼特徵（課程策略）
        g["foreign_net_3"] = g["foreign_net"].rolling(3).sum()
        g["trust_net_3"] = g["trust_net"].rolling(3).sum()
        g["foreign_trust_same_side"] = (
            ((g["foreign_net"] > 0) & (g["trust_net"] > 0))
            | ((g["foreign_net"] < 0) & (g["trust_net"] < 0))
        ).astype(int)
        outputs.append(g)
    merged = pd.concat(outputs, ignore_index=True)

    merged["vol_pct_60"] = (
        merged.groupby("trading_date")["vol_60"]
        .rank(pct=True)
        .fillna(0.5)
    )
    merged = merged.replace([float("inf"), float("-inf")], pd.NA)
    merged = merged[merged["close"] > 0].copy()
    return merged


def detect_regime(price_df: pd.DataFrame, config) -> str:
    market = (
        price_df.groupby("trading_date")["close"]
        .mean()
        .reset_index()
        .rename(columns={"close": "avg_close"})
    )
    detector = get_regime_detector(config)
    return detector.detect(market, config).get("regime", "BULL")


def resolve_weights(regime: str, config, config_json: Dict) -> Dict[str, float]:
    if regime == "BEAR":
        weights = config_json.get("weights_bear") or getattr(config, "strategy_weights_bear", None)
    else:
        weights = config_json.get("weights_bull") or getattr(config, "strategy_weights_bull", None)
    if not weights:
        weights = {"MomentumTrend": 0.6, "MeanReversion": 0.2, "DefensiveLowVol": 0.2}
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}
