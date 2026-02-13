from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class BaseRegimeDetector(ABC):
    @abstractmethod
    def detect(self, market_price_df: pd.DataFrame, config) -> Dict[str, object]:
        """回傳 {regime, score, meta}。"""
        raise NotImplementedError


class MovingAverageRegimeDetector(BaseRegimeDetector):
    def detect(self, market_price_df: pd.DataFrame, config) -> Dict[str, object]:
        ma_days = int(getattr(config, "market_filter_ma_days", 60))
        if market_price_df.empty:
            return {
                "regime": "BULL",
                "score": 0.0,
                "meta": {
                    "ma_days": ma_days,
                    "current_price": None,
                    "ma_value": None,
                    "diff_pct": None,
                },
            }

        df = market_price_df.copy().sort_values("trading_date")
        price_col = "avg_close" if "avg_close" in df.columns else "close"
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        df = df.dropna(subset=[price_col])
        if len(df) < ma_days:
            return {
                "regime": "BULL",
                "score": 0.0,
                "meta": {
                    "ma_days": ma_days,
                    "current_price": float(df[price_col].iloc[-1]) if not df.empty else None,
                    "ma_value": None,
                    "diff_pct": None,
                },
            }

        df["ma"] = df[price_col].rolling(ma_days).mean()
        latest = df.iloc[-1]
        current_price = float(latest[price_col])
        ma_value = float(latest["ma"])
        diff_pct = (current_price / ma_value - 1.0) if ma_value != 0 else 0.0
        regime = "BEAR" if current_price < ma_value else "BULL"
        return {
            "regime": regime,
            "score": float(diff_pct),
            "meta": {
                "ma_days": ma_days,
                "current_price": current_price,
                "ma_value": ma_value,
                "diff_pct": float(diff_pct),
            },
        }


def get_regime_detector(config) -> BaseRegimeDetector:
    detector_name = str(getattr(config, "regime_detector", "ma")).lower()
    if detector_name == "ma":
        return MovingAverageRegimeDetector()
    raise ValueError(f"Unsupported regime detector: {detector_name}. Set REGIME_DETECTOR=ma")
