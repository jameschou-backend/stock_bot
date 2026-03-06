#!/usr/bin/env python
"""三種倉位配置方法 24 個月回測公平對比腳本。

統一參數：
    - trailing_stop_pct = -0.12（原最佳配置，非 -0.15）
    - stoploss_pct      = -0.07
    - TopN              = 20
    - 週再平衡 (W)

用法：
    python scripts/compare_position_sizing.py
    python scripts/compare_position_sizing.py --months 24 --fast
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

# ── 公平對比統一配置（直接作為 run_backtest 參數，三種方法完全一致）──
FAIR_TRAILING_STOP: float = -0.12      # 原最佳配置，非預設 -0.15
FAIR_STOPLOSS: float = -0.07
FAIR_TOPN: int = 20
FAIR_POSITION_SIZING: str = "vol_inverse"   # 舊參數固定，position_sizing_method 決定實際方法


def main():
    parser = argparse.ArgumentParser(description="三種倉位方法公平對比回測")
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    config = load_config()
    methods = ["vol_inverse", "mean_variance", "risk_parity"]
    results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"倉位方法：{method}  （trailing={FAIR_TRAILING_STOP:.0%}）")
        print("=" * 60)
        with get_session() as session:
            res = run_backtest(
                config=config,
                db_session=session,
                backtest_months=args.months,
                topn=FAIR_TOPN,
                stoploss_pct=FAIR_STOPLOSS,
                trailing_stop_pct=FAIR_TRAILING_STOP,
                position_sizing=FAIR_POSITION_SIZING,
                position_sizing_method=method,
                fast_mode=args.fast,
            )
        s = res.get("summary", {})
        s["_method"] = method
        s["_fair_trailing_stop"] = FAIR_TRAILING_STOP
        results[method] = s

    # ── 輸出對比表格 ──
    print("\n" + "=" * 78)
    print(f"三種倉位方法回測公平對比（{args.months} 個月 Walk-Forward）")
    print(f"統一參數：trailing={FAIR_TRAILING_STOP:.0%}  stoploss={FAIR_STOPLOSS:.0%}  TopN={FAIR_TOPN}")
    print("=" * 78)
    header = f"{'方法':<16} {'累積報酬':>10} {'年化報酬':>10} {'Sharpe':>8} {'MDD':>9} {'Calmar':>8}"
    print(header)
    print("-" * 78)
    for method in methods:
        s = results[method]
        tot = s.get("total_return", float("nan"))
        ann = s.get("annualized_return", float("nan"))
        sr  = s.get("sharpe_ratio", float("nan"))
        mdd = s.get("max_drawdown", float("nan"))
        cal = s.get("calmar_ratio") or float("nan")
        print(
            f"{method:<16} "
            f"{tot:>10.2%} "
            f"{ann:>10.2%} "
            f"{sr:>8.3f} "
            f"{mdd:>9.2%} "
            f"{cal:>8.3f}"
        )
    print("=" * 78)

    # ── 儲存 JSON ──
    artifacts_dir = ROOT / "artifacts" / "backtest"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = artifacts_dir / f"position_sizing_compare_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n對比結果已儲存: {out_path}")


if __name__ == "__main__":
    main()
