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


class DummyAlwaysExitStrategy:
    name = "DummyExit"
    entry_rules = [FunctionalRule("always", lambda ctx: pd.Series([True] * len(ctx.df), index=ctx.df.index))]
    filter_rules = []
    exit_rules = [FunctionalRule("always_exit", lambda ctx: pd.Series([True] * len(ctx.df), index=ctx.df.index))]


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


def test_backtest_engine_enforces_min_hold_days_for_exit_rules():
    df = pd.DataFrame(
        {
            "stock_id": ["2330", "2330"],
            "trading_date": ["2026-02-10", "2026-02-11"],
            "close": [100.0, 99.0],
        }
    )
    cfg = BacktestConfig(
        start_date=pd.Timestamp("2026-02-10").date(),
        end_date=pd.Timestamp("2026-02-11").date(),
        min_hold_days=1,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(df, [StrategyAllocation(strategy=DummyAlwaysExitStrategy(), weight=1.0)])
    buy_rows = [t for t in result["trades"] if t["action"] == "BUY"]
    sell_rows = [t for t in result["trades"] if t["action"] == "SELL"]
    assert len(buy_rows) == 1
    assert len(sell_rows) == 0


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


def test_backtest_engine_price_tick_rounding_and_slippage():
    cfg = BacktestConfig(start_date=pd.Timestamp("2026-02-10").date(), end_date=pd.Timestamp("2026-02-11").date())
    engine = BacktestEngine(cfg)
    # <100: 第二位小數
    assert engine._round_to_valid_price(16.3363, "BUY") == 16.34
    assert engine._round_to_valid_price(16.3363, "SELL") == 16.33
    # 100~999: 第一位小數（對應 442.442 -> 442.5）
    assert engine._round_to_valid_price(442.442, "BUY") == 442.5
    assert engine._round_to_valid_price(442.442, "SELL") == 442.4
    # >=1000: 每 5 元一檔
    assert engine._round_to_valid_price(1001.0, "BUY") == 1005.0
    assert engine._round_to_valid_price(1001.0, "SELL") == 1000.0


def test_backtest_engine_fee_ceil_and_min_fee():
    cfg = BacktestConfig(start_date=pd.Timestamp("2026-02-10").date(), end_date=pd.Timestamp("2026-02-11").date())
    engine = BacktestEngine(cfg)
    # 未達 20 元，應收 20
    assert engine._calc_fee(1, 100.0) == 20.0
    # 62 * 436.563 * 0.001425 = 38.5703... -> 無條件進位 39
    assert engine._calc_fee(62, 436.563) == 39.0


def test_backtest_engine_qty_normalization_prefers_board_lot():
    cfg = BacktestConfig(start_date=pd.Timestamp("2026-02-10").date(), end_date=pd.Timestamp("2026-02-11").date())
    engine = BacktestEngine(cfg)
    assert engine._normalize_qty(99) == 0
    assert engine._normalize_qty(100) == 100
    assert engine._normalize_qty(950) == 900
    assert engine._normalize_qty(1000) == 1000
    assert engine._normalize_qty(1999) == 1000


def test_backtest_engine_buy_qty_uses_lot_or_100_shares():
    df = pd.DataFrame(
        {
            "stock_id": ["2330", "2330"],
            "trading_date": ["2026-02-10", "2026-02-11"],
            "close": [100.0, 100.0],
        }
    )
    cfg = BacktestConfig(
        start_date=pd.Timestamp("2026-02-10").date(),
        end_date=pd.Timestamp("2026-02-11").date(),
        max_positions=1,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(df, [StrategyAllocation(strategy=DummyStrategy(), weight=1.0)])
    buys = [t for t in result["trades"] if t["action"] == "BUY"]
    assert len(buys) == 1
    qty = int(buys[0]["qty"])
    assert qty % 100 == 0


def test_backtest_engine_position_size_multiplier_changes_buy_qty():
    df = pd.DataFrame(
        {
            "stock_id": ["2330", "2330"],
            "trading_date": ["2026-02-10", "2026-02-11"],
            "close": [100.0, 100.0],
        }
    )
    cfg_full = BacktestConfig(
        start_date=pd.Timestamp("2026-02-10").date(),
        end_date=pd.Timestamp("2026-02-11").date(),
        max_positions=1,
        risk_per_trade=0.02,
        position_size_multiplier=1.0,
        board_lot_shares=100,
        min_trade_shares=100,
    )
    engine_full = BacktestEngine(cfg_full)
    result_full = engine_full.run(df, [StrategyAllocation(strategy=DummyStrategy(), weight=1.0)])
    full_qty = int([t for t in result_full["trades"] if t["action"] == "BUY"][0]["qty"])

    cfg_half = BacktestConfig(
        start_date=pd.Timestamp("2026-02-10").date(),
        end_date=pd.Timestamp("2026-02-11").date(),
        max_positions=1,
        risk_per_trade=0.02,
        position_size_multiplier=0.5,
        board_lot_shares=100,
        min_trade_shares=100,
    )
    engine_half = BacktestEngine(cfg_half)
    result_half = engine_half.run(df, [StrategyAllocation(strategy=DummyStrategy(), weight=1.0)])
    half_qty = int([t for t in result_half["trades"] if t["action"] == "BUY"][0]["qty"])

    assert half_qty < full_qty
    assert half_qty == full_qty // 2


def test_backtest_engine_target_exposure_pct_caps_buy_qty():
    df = pd.DataFrame(
        {
            "stock_id": ["2330", "2330"],
            "trading_date": ["2026-02-10", "2026-02-11"],
            "close": [100.0, 100.0],
        }
    )
    cfg_full = BacktestConfig(
        start_date=pd.Timestamp("2026-02-10").date(),
        end_date=pd.Timestamp("2026-02-11").date(),
        max_positions=1,
        risk_per_trade=0.2,
        target_exposure_pct=1.0,
        board_lot_shares=100,
        min_trade_shares=100,
    )
    engine_full = BacktestEngine(cfg_full)
    result_full = engine_full.run(df, [StrategyAllocation(strategy=DummyStrategy(), weight=1.0)])
    full_qty = int([t for t in result_full["trades"] if t["action"] == "BUY"][0]["qty"])

    cfg_low = BacktestConfig(
        start_date=pd.Timestamp("2026-02-10").date(),
        end_date=pd.Timestamp("2026-02-11").date(),
        max_positions=1,
        risk_per_trade=0.2,
        target_exposure_pct=0.3,
        board_lot_shares=100,
        min_trade_shares=100,
    )
    engine_low = BacktestEngine(cfg_low)
    result_low = engine_low.run(df, [StrategyAllocation(strategy=DummyStrategy(), weight=1.0)])
    low_qty = int([t for t in result_low["trades"] if t["action"] == "BUY"][0]["qty"])

    assert low_qty < full_qty
