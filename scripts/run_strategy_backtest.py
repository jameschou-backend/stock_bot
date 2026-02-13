#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict

from app.config import load_config
from skills.strategy_factory.engine import BacktestConfig, BacktestEngine, StrategyAllocation
from skills.strategy_factory.registry import get, register_defaults
from skills.strategy_factory.data import compute_indicators, detect_regime, load_price_df, resolve_weights


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    args = parser.parse_args()

    config = load_config()
    register_defaults()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    raw = load_price_df(start_date, end_date)
    df = compute_indicators(raw)
    regime = detect_regime(df, config)
    weights = resolve_weights(regime, config, {})

    allocations = [
        StrategyAllocation(strategy=get(name), weight=weight)
        for name, weight in weights.items()
    ]
    bt_cfg = BacktestConfig(start_date=start_date, end_date=end_date)
    engine = BacktestEngine(bt_cfg)
    result = engine.run(df, allocations)

    out = ROOT / "artifacts" / "strategy_backtest.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"regime={regime} output={out}")


if __name__ == "__main__":
    main()
