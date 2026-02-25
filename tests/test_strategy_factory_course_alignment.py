from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from skills.strategy_factory.data import compute_indicators, detect_regime
from skills.strategy_factory.strategies import (
    CourseBreakout,
    CoursePullback,
    CourseVolumeMomentum,
)


@dataclass
class _Cfg:
    regime_detector: str = "ma"
    market_filter_ma_days: int = 2


def _price_df_for_regime(date2_prices: list[float]) -> pd.DataFrame:
    stock_ids = ["1101", "1102", "1103", "1104"]
    rows = []
    for sid in stock_ids:
        rows.append({"stock_id": sid, "trading_date": pd.Timestamp("2026-02-10"), "close": 10.0})
    for sid, px in zip(stock_ids, date2_prices):
        rows.append({"stock_id": sid, "trading_date": pd.Timestamp("2026-02-11"), "close": px})
    return pd.DataFrame(rows)


def test_compute_indicators_adds_course_columns():
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "stock_id": ["2330"] * len(dates),
            "trading_date": dates,
            "open": [100.0 + i for i in range(len(dates))],
            "high": [101.0 + i for i in range(len(dates))],
            "low": [99.0 + i for i in range(len(dates))],
            "close": [100.0 + i for i in range(len(dates))],
            "volume": [1_000_000 + i * 1000 for i in range(len(dates))],
            "foreign_net": [0.0] * len(dates),
            "trust_net": [0.0] * len(dates),
            "dealer_net": [0.0] * len(dates),
        }
    )
    out = compute_indicators(df)
    for col in ["ma_5", "ma_10", "high_400", "volume_max_10"]:
        assert col in out.columns


def test_detect_regime_uses_breadth_bull():
    df = _price_df_for_regime([11.0, 11.0, 9.0, 11.0])  # up=3 down=1
    regime = detect_regime(df, _Cfg())
    assert regime == "BULL"


def test_detect_regime_uses_breadth_bear():
    df = _price_df_for_regime([9.0, 9.0, 11.0, 9.0])  # up=1 down=3
    regime = detect_regime(df, _Cfg())
    assert regime == "BEAR"


def test_detect_regime_neutral_breadth_falls_back_to_ma():
    # up=2 down=2 => ratio=1.0, fallback to MA detector.
    # avg_close: 10 -> 9 => below 2-day MA(9.5) => BEAR
    df = _price_df_for_regime([11.0, 11.0, 7.0, 7.0])
    regime = detect_regime(df, _Cfg())
    assert regime == "BEAR"


def test_course_strategies_use_updated_rules():
    vm = CourseVolumeMomentum()
    bo = CourseBreakout()
    pb = CoursePullback()

    vm_exit_names = [r.name for r in vm.exit_rules]
    bo_entry_names = [r.name for r in bo.entry_rules]
    pb_entry_names = [r.name for r in pb.entry_rules]

    assert "close_below_ma_5" in vm_exit_names
    assert "close_below_ma_10" in vm_exit_names
    assert "close_near_high_400" in bo_entry_names
    assert "volume_gte_volume_max_10" in bo_entry_names
    assert any(name.startswith("close_near_ma_10_") for name in pb_entry_names)
