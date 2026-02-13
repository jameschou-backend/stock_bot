from __future__ import annotations

from datetime import date
from typing import Dict

import pandas as pd

from app.db import get_session
from app.models import RawPrice
from skills.regime import get_regime_detector


def load_price_df(start_date: date, end_date: date) -> pd.DataFrame:
    with get_session() as session:
        df = pd.read_sql(
            session.query(
                RawPrice.stock_id,
                RawPrice.trading_date,
                RawPrice.open,
                RawPrice.high,
                RawPrice.low,
                RawPrice.close,
                RawPrice.volume,
            )
            .filter(RawPrice.trading_date.between(start_date, end_date))
            .statement,
            session.get_bind(),
        )
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["stock_id", "trading_date"])
    outputs = []
    for _, g in df.groupby("stock_id"):
        g = g.copy()
        close = g["close"]
        volume = g["volume"]
        g["ret_20"] = close.pct_change(20)
        g["ma_20"] = close.rolling(20).mean()
        g["ma_60"] = close.rolling(60).mean()
        g["volume_20"] = volume.rolling(20).mean()
        g["volume_60_mean"] = volume.rolling(60).mean()

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
        outputs.append(g)
    merged = pd.concat(outputs, ignore_index=True)

    merged["vol_pct_60"] = (
        merged.groupby("trading_date")["vol_60"]
        .rank(pct=True)
        .fillna(0.5)
    )
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
