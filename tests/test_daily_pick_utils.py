from datetime import date

import pandas as pd

from skills import daily_pick


def test_impute_features_fills_nan_and_all_nan():
    df = pd.DataFrame(
        {
            "a": [1.0, None, 3.0],
            "b": [None, None, None],
        }
    )
    filled, stats = daily_pick._impute_features(df)
    assert filled["a"].isna().sum() == 0
    assert filled["b"].isna().sum() == 0
    assert stats["filled_cells"] == 4
    assert "b" in stats["all_nan_cols"]


def test_choose_pick_date_fallback():
    d1 = date(2026, 2, 3)
    d2 = date(2026, 2, 2)
    feature_df = pd.DataFrame(
        {
            "stock_id": ["1234", "1234"],
            "trading_date": [d1, d2],
        }
    )
    price_df = pd.DataFrame(
        {
            "stock_id": ["1234", "1234"],
            "trading_date": [d2, d2],
            "close": [10.0, 10.0],
            "volume": [1000, 1000],
        }
    )
    pick_date, chosen, logs = daily_pick._choose_pick_date(
        [d1, d2], feature_df, price_df, topn=1, min_avg_volume=0, fallback_days=2
    )
    assert pick_date == d2
    assert len(chosen) == 1
    assert logs["fallback_days"] == 1
