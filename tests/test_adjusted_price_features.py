from __future__ import annotations

import numpy as np
import pandas as pd

from skills.build_features import _compute_features


def _make_df() -> pd.DataFrame:
    periods = 40
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")
    true_close = np.linspace(100, 139, periods)
    raw_close = true_close.copy()
    raw_close[20:] = raw_close[20:] - 15.0  # 模擬除權造成的假跳空
    return pd.DataFrame(
        {
            "stock_id": ["2330"] * periods,
            "trading_date": dates,
            "open": raw_close,
            "high": raw_close * 1.01,
            "low": raw_close * 0.99,
            "close": raw_close,
            "adj_close": true_close,
            "volume": [1000] * periods,
            "foreign_net": [0] * periods,
            "trust_net": [0] * periods,
            "dealer_net": [0] * periods,
        }
    )


def test_adjusted_price_avoids_fake_gap_pollution():
    df = _make_df()
    out_adj = _compute_features(df, use_adjusted_price=True).sort_values("trading_date")
    out_raw = _compute_features(df, use_adjusted_price=False).sort_values("trading_date")

    last_adj = out_adj.iloc[-1]
    last_raw = out_raw.iloc[-1]

    expected_ret_20 = df["adj_close"].iloc[-1] / df["adj_close"].iloc[-21] - 1
    expected_ma_20 = df["adj_close"].rolling(20).mean().iloc[-1]

    assert np.isclose(last_adj["ret_20"], expected_ret_20, atol=1e-10)
    assert np.isclose(last_adj["ma_20"], expected_ma_20, atol=1e-10)
    # 未還原價應與還原價有明顯差異，代表假跳空會污染訊號
    assert abs(last_raw["ret_20"] - last_adj["ret_20"]) > 0.05
