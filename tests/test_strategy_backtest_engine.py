import pandas as pd

from skills.strategy_factory.base import FunctionalRule
from skills.strategy_factory.engine import BacktestConfig, BacktestEngine, StrategyAllocation


class DummyStrategy:
    name = "Dummy"
    entry_rules = [FunctionalRule("always", lambda ctx: pd.Series([True] * len(ctx.df), index=ctx.df.index))]
    filter_rules = []
    exit_rules = []


class DummyManyCandidatesStrategy:
    name = "DummyMany"
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


def test_backtest_engine_respects_max_positions_for_pending_buys():
    day1 = ["2026-02-10"] * 10
    day2 = ["2026-02-11"] * 10
    stock_ids = [f"{2300 + i}" for i in range(10)]
    df = pd.DataFrame(
        {
            "stock_id": stock_ids + stock_ids,
            "trading_date": day1 + day2,
            "close": [100.0] * 20,
        }
    )
    cfg = BacktestConfig(
        start_date=pd.Timestamp("2026-02-10").date(),
        end_date=pd.Timestamp("2026-02-11").date(),
        max_positions=3,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(df, [StrategyAllocation(strategy=DummyManyCandidatesStrategy(), weight=1.0)])
    buy_rows = [t for t in result["trades"] if t["action"] == "BUY"]
    assert len(buy_rows) <= 3
    assert max(point["positions"] for point in result["equity_curve"]) <= 3


def test_backtest_engine_skip_zero_price_without_crash():
    df = pd.DataFrame(
        {
            "stock_id": ["2330", "2330"],
            "trading_date": ["2026-02-10", "2026-02-11"],
            "close": [0, 101],
        }
    )
    cfg = BacktestConfig(start_date=pd.Timestamp("2026-02-10").date(), end_date=pd.Timestamp("2026-02-11").date())
    engine = BacktestEngine(cfg)
    result = engine.run(df, [StrategyAllocation(strategy=DummyStrategy(), weight=1.0)])
    assert len(result["equity_curve"]) == 2
