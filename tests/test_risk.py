from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

from skills import risk


def test_apply_liquidity_filter_threshold():
    cfg = SimpleNamespace(min_avg_turnover=0.5)  # 0.5 億
    price_df = pd.DataFrame(
        {
            "stock_id": ["2330", "2330", "2317", "2317"],
            "trading_date": [date(2026, 2, 10), date(2026, 2, 11), date(2026, 2, 10), date(2026, 2, 11)],
            "close": [100.0, 100.0, 10.0, 10.0],
            "volume": [800000, 700000, 100000, 90000],
        }
    )
    out = risk.apply_liquidity_filter(price_df, cfg)
    ids = set(out["stock_id"].astype(str).tolist())
    assert "2330" in ids
    assert "2317" not in ids


def test_pick_topn_orders_by_score_desc():
    scores_df = pd.DataFrame(
        {
            "stock_id": ["1101", "2330", "2317"],
            "score": [0.1, 0.9, 0.3],
        }
    )
    out = risk.pick_topn(scores_df, 2)
    assert out["stock_id"].tolist() == ["2330", "2317"]


def test_apply_stoploss_triggers_on_threshold():
    positions = pd.DataFrame(
        {
            "stock_id": ["2330"],
            "entry_date": [date(2026, 2, 10)],
            "planned_exit_date": [date(2026, 2, 13)],
            "entry_price": [100.0],
        }
    )
    price_df = pd.DataFrame(
        {
            "stock_id": ["2330", "2330", "2330", "2330"],
            "trading_date": [date(2026, 2, 10), date(2026, 2, 11), date(2026, 2, 12), date(2026, 2, 13)],
            "close": [100.0, 96.0, 92.0, 95.0],
        }
    )
    out = risk.apply_stoploss(positions, price_df, stoploss_pct=-0.07)
    assert len(out) == 1
    assert bool(out.loc[0, "stoploss_triggered"]) is True
    assert float(out.loc[0, "exit_price"]) == 92.0
