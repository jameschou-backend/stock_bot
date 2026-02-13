import pandas as pd

from skills.strategy_factory.base import FunctionalRule
from skills.strategy_factory.engine import BacktestConfig, BacktestEngine, StrategyAllocation


class DummyStrategy:
    name = "Dummy"
    entry_rules = [FunctionalRule("always", lambda ctx: pd.Series([True] * len(ctx.df), index=ctx.df.index))]
    filter_rules = []
    exit_rules = []


def test_backtest_engine_runs():
    df = pd.DataFrame(
        {
            "stock_id": ["2330", "2317", "2330", "2317"],
            "trading_date": ["2026-02-10", "2026-02-10", "2026-02-11", "2026-02-11"],
            "close": [100, 50, 101, 49],
        }
    )
    cfg = BacktestConfig(start_date=pd.Timestamp("2026-02-10").date(), end_date=pd.Timestamp("2026-02-11").date())
    engine = BacktestEngine(cfg)
    result = engine.run(df, [StrategyAllocation(strategy=DummyStrategy(), weight=1.0)])
    assert len(result["equity_curve"]) == 2
    assert len(result["trades"]) >= 1
