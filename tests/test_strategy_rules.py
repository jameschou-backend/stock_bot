import pandas as pd

from skills.strategy_factory.base import FunctionalRule, RuleContext, apply_rules_all, apply_rules_any
from skills.strategy_factory import rules_exit as rx


def test_rules_all_any():
    df = pd.DataFrame(
        {
            "stock_id": ["2330", "2317"],
            "trading_date": ["2026-02-10", "2026-02-10"],
            "close": [100, 50],
        }
    )
    ctx = RuleContext(df=df, now=pd.Timestamp("2026-02-10"))
    r1 = FunctionalRule("gt_80", lambda c: c.df["close"] > 80)
    r2 = FunctionalRule("gt_40", lambda c: c.df["close"] > 40)

    all_mask = apply_rules_all([r1, r2], ctx)
    any_mask = apply_rules_any([r1, r2], ctx)

    assert all_mask.tolist() == [True, False]
    assert any_mask.tolist() == [True, True]


def test_exit_rules_with_positions():
    df = pd.DataFrame(
        {
            "stock_id": ["2330"],
            "trading_date": ["2026-02-10"],
            "close": [90],
            "ma_20": [95],
        }
    )
    class Pos:
        avg_cost = 100
        max_price = 105
        entry_date = pd.Timestamp("2026-02-01")
    ctx = RuleContext(df=df, now=pd.Timestamp("2026-02-10"), extra={"positions": {"2330": Pos()}})
    assert bool(rx.exit_stop_loss_fixed(-0.07).apply(ctx).iloc[0]) is True
    assert bool(rx.exit_trailing_stop(0.1).apply(ctx).iloc[0]) is True
    assert bool(rx.exit_below_ma("ma_20").apply(ctx).iloc[0]) is True
