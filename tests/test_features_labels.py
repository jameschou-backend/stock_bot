import numpy as np
import pandas as pd

from skills.build_features import (
    CORE_FEATURE_COLUMNS,
    EXTENDED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    _compute_features,
)
from skills.build_labels import _compute_labels


def _make_sample_df(periods=80):
    """建立包含所有必要欄位的測試 DataFrame"""
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    close = 100 * (1.01 ** np.arange(periods))
    high = close * 1.02
    low = close * 0.98
    return pd.DataFrame(
        {
            "stock_id": ["2330"] * periods,
            "trading_date": dates,
            "open": close * 0.99,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1000] * periods,
            "foreign_net": [10] * periods,
            "trust_net": [5] * periods,
            "dealer_net": [2] * periods,
            "margin_purchase_balance": [50000 + i * 100 for i in range(periods)],
            "short_sale_balance": [5000 + i * 10 for i in range(periods)],
        }
    )


def test_compute_features_basic():
    """測試基礎特徵計算正確性"""
    df = _make_sample_df(periods=80)
    close = df["close"].values

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


def test_compute_features_extended():
    """測試擴充特徵（RSI, MACD, KD, 融資融券, 大盤相對強弱）"""
    df = _make_sample_df(periods=80)
    out = _compute_features(df)
    last = out.iloc[-1]

    # RSI 應在 0-100 之間
    assert 0 <= last["rsi_14"] <= 100, f"RSI out of range: {last['rsi_14']}"

    # MACD histogram 應有值
    assert np.isfinite(last["macd_hist"]), f"MACD hist not finite: {last['macd_hist']}"

    # KD 應在 0-100 之間
    assert 0 <= last["kd_k"] <= 100, f"KD_K out of range: {last['kd_k']}"
    assert 0 <= last["kd_d"] <= 100, f"KD_D out of range: {last['kd_d']}"

    # 融資融券特徵應有值
    assert np.isfinite(last["margin_balance_chg_5"]), "margin_balance_chg_5 not finite"
    assert np.isfinite(last["margin_short_ratio"]), "margin_short_ratio not finite"

    # market_rel_ret_20：單一股票時，相對大盤報酬應為 0
    assert np.isclose(last["market_rel_ret_20"], 0, atol=1e-10), (
        f"Single stock market_rel_ret should be 0, got {last['market_rel_ret_20']}"
    )


def test_feature_columns_complete():
    """確認所有宣告的特徵欄位都有計算到"""
    df = _make_sample_df(periods=80)
    out = _compute_features(df)
    last = out.iloc[-1]

    for col in FEATURE_COLUMNS:
        assert col in out.columns, f"Feature column '{col}' missing from output"
        # 在足夠資料的最後一行，核心特徵應有值
        if col in CORE_FEATURE_COLUMNS:
            assert np.isfinite(last[col]), f"Core feature '{col}' is not finite: {last[col]}"


def test_compute_features_no_margin_data():
    """沒有融資融券資料時，融資融券特徵應為 NaN"""
    df = _make_sample_df(periods=80)
    df = df.drop(columns=["margin_purchase_balance", "short_sale_balance"])
    out = _compute_features(df)
    last = out.iloc[-1]

    # 融資融券特徵應為 NaN
    assert np.isnan(last["margin_balance_chg_5"])
    assert np.isnan(last["margin_short_ratio"])

    # 其他特徵應正常
    assert np.isfinite(last["ret_5"])
    assert np.isfinite(last["rsi_14"])


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
