#!/usr/bin/env python
"""Walk-forward backtest CLI.

Usage:
    python scripts/run_backtest.py                    # 預設 24 個月回測
    python scripts/run_backtest.py --months 36        # 36 個月
    python scripts/run_backtest.py --topn 10          # 只選 10 檔
    python scripts/run_backtest.py --no-stoploss      # 不設停損
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.db import get_session
from skills.backtest import run_backtest


def main():
    parser = argparse.ArgumentParser(description="Walk-forward 回測")
    parser.add_argument("--months", type=int, default=24, help="回測月數 (預設 24)")
    parser.add_argument("--retrain-freq", type=int, default=3, help="模型重訓頻率（月）(預設 3)")
    parser.add_argument("--topn", type=int, default=None, help="每期選股數量 (預設使用 config)")
    parser.add_argument("--stoploss", type=float, default=None, help="停損比例 (如 -0.07)")
    parser.add_argument("--no-stoploss", action="store_true", help="不設停損")
    parser.add_argument("--cost", type=float, default=None, help="來回交易成本 (如 0.00585)")
    parser.add_argument("--output", type=str, default=None, help="結果輸出 JSON 路徑")
    args = parser.parse_args()

    config = load_config()
    topn = args.topn or config.topn
    stoploss = 0.0 if args.no_stoploss else (args.stoploss if args.stoploss is not None else config.stoploss_pct)
    cost = args.cost if args.cost is not None else config.transaction_cost_pct

    with get_session() as session:
        result = run_backtest(
            config=config,
            db_session=session,
            backtest_months=args.months,
            retrain_freq_months=args.retrain_freq,
            topn=topn,
            stoploss_pct=stoploss,
            transaction_cost_pct=cost,
        )

    # 輸出 JSON
    output_path = args.output
    if output_path is None:
        artifacts_dir = ROOT / "artifacts" / "backtest"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(artifacts_dir / f"backtest_{ts}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果已儲存: {output_path}")


if __name__ == "__main__":
    main()
