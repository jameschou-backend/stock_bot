import numpy as np
import pandas as pd

from skills.build_features import _compute_features
from skills.build_labels import _compute_labels


def test_compute_features_basic():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    close = 100 * (1.01 ** np.arange(60))
    df = pd.DataFrame(
        {
            "stock_id": ["2330"] * 60,
            "trading_date": dates,
            "close": close,
            "volume": [1000] * 60,
            "foreign_net": [10] * 60,
            "trust_net": [5] * 60,
            "dealer_net": [2] * 60,
        }
    )

    out = _compute_features(df)
    last = out.iloc[-1]

    expected_ret_5 = close[-1] / close[-6] - 1
    expected_ma_5 = pd.Series(close).rolling(5).mean().iloc[-1]
    expected_ma_20 = pd.Series(close).rolling(20).mean().iloc[-1]

    assert np.isclose(last["ret_5"], expected_ret_5)
    assert np.isclose(last["ma_5"], expected_ma_5)
    assert np.isclose(last["ma_20"], expected_ma_20)
    assert np.isclose(last["bias_20"], close[-1] / expected_ma_20 - 1)
    assert np.isclose(last["foreign_net_5"], 50)
    assert np.isclose(last["trust_net_20"], 100)
    assert abs(last["vol_20"]) < 1e-12


def test_compute_labels_basic():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    close = [10, 11, 12, 13, 14]
    df = pd.DataFrame(
        {
            "stock_id": ["2330"] * 5,
            "trading_date": dates,
            "close": close,
        }
    )

    out = _compute_labels(df, horizon=2)
    out = out.sort_values("trading_date").reset_index(drop=True)

    expected = close[2] / close[0] - 1
    assert np.isclose(out.loc[0, "future_ret_h"], expected)
