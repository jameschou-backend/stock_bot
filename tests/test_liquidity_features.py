from __future__ import annotations

import numpy as np
import pandas as pd

from skills.build_features import _compute_features


def test_liquidity_feature_columns():
    periods = 30
    df = pd.DataFrame(
        {
            "stock_id": ["2330"] * periods,
            "trading_date": pd.date_range("2025-01-01", periods=periods, freq="D"),
            "open": [100.0] * periods,
            "high": [101.0] * periods,
            "low": [99.0] * periods,
            "close": [100.0] * periods,
            "volume": [1000 + i for i in range(periods)],
            "foreign_net": [0] * periods,
            "trust_net": [0] * periods,
            "dealer_net": [0] * periods,
        }
    )
    out = _compute_features(df).sort_values("trading_date")
    last = out.iloc[-1]

    expected_amt = 100.0 * (1000 + periods - 1)
    expected_amt_20 = np.mean([100.0 * (1000 + i) for i in range(periods - 20, periods)])
    assert np.isclose(last["amt"], expected_amt)
    assert np.isclose(last["amt_20"], expected_amt_20)
    assert np.isclose(last["amt_ratio_20"], expected_amt / expected_amt_20)
