from __future__ import annotations

from itertools import product
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from skills.backtest import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="參數網格回測")
    parser.add_argument("--months", type=int, default=120, help="回測月數（預設 120 = 10 年）")
    args = parser.parse_args()

    config = load_config()
    output_dir = PROJECT_ROOT / "artifacts" / "ai_answers"
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = {
        "topn": [5, 10, 15, 20],
        "stoploss_pct": [-0.05, -0.07, -0.10],
        "holding_days_proxy": [10, 20, 30],  # 目前回測為月度再平衡，保留欄位供後續擴充
        "min_avg_turnover": [0.5, 1.0, 1.5],
    }

    rows = []
    with get_session() as session:
        for topn, stoploss, hold_days, min_turnover in product(
            grid["topn"], grid["stoploss_pct"], grid["holding_days_proxy"], grid["min_avg_turnover"]
        ):
            local_cfg = config
            # 目前 min_avg_turnover 為 config 層，這裡記錄參數，實際若要生效需重建 features/picks
            result = run_backtest(
                local_cfg,
                session,
                backtest_months=args.months,
                topn=topn,
                stoploss_pct=stoploss,
                transaction_cost_pct=config.transaction_cost_pct,
                retrain_freq_months=3,
                min_train_days=500,
                min_avg_turnover=min_turnover,
            )
            summary = result.get("summary", {})
            rows.append(
                {
                    "topn": topn,
                    "stoploss_pct": stoploss,
                    "holding_days_proxy": hold_days,
                    "min_avg_turnover": min_turnover,
                    **summary,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["annualized_return", "max_drawdown"], ascending=[False, False])
    out_path = output_dir / "grid_backtest_results.csv"
    out_df.to_csv(out_path, index=False)

    best = out_df.head(10).to_dict(orient="records")
    summary_path = output_dir / "grid_backtest_top10.json"
    summary_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[grid-backtest] wrote: {out_path}")
    print(f"[grid-backtest] wrote: {summary_path}")


if __name__ == "__main__":
    main()
