from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd

from skills.regime import MovingAverageRegimeDetector, get_regime_detector


def _old_is_bear_like(market_df: pd.DataFrame, ma_days: int) -> bool:
    if len(market_df) < ma_days:
        return False
    tmp = market_df.copy().sort_values("trading_date")
    tmp["ma"] = tmp["avg_close"].astype(float).rolling(ma_days).mean()
    latest = tmp.iloc[-1]
    return float(latest["avg_close"]) < float(latest["ma"])


def test_ma_regime_detector_matches_previous_rule():
    ma_days = 3
    base = date(2026, 2, 1)
    market_df = pd.DataFrame(
        {
            "trading_date": [base + timedelta(days=i) for i in range(6)],
            "avg_close": [100.0, 101.0, 102.0, 101.0, 99.0, 96.0],
        }
    )
    cfg = SimpleNamespace(market_filter_ma_days=ma_days, regime_detector="ma")
    detector = get_regime_detector(cfg)

    out = detector.detect(market_df, cfg)
    assert out["regime"] in {"BULL", "BEAR"}
    assert (out["regime"] == "BEAR") == _old_is_bear_like(market_df, ma_days)


def test_ma_detector_meta_fields_exist():
    cfg = SimpleNamespace(market_filter_ma_days=2, regime_detector="ma")
    detector = MovingAverageRegimeDetector()
    market_df = pd.DataFrame(
        {
            "trading_date": [date(2026, 2, 1), date(2026, 2, 2)],
            "avg_close": [100.0, 98.0],
        }
    )
    out = detector.detect(market_df, cfg)
    meta = out["meta"]
    assert "ma_days" in meta
    assert "current_price" in meta
    assert "ma_value" in meta
    assert "diff_pct" in meta
