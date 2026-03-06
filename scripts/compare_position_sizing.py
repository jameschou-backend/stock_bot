#!/usr/bin/env python
"""三種倉位配置方法 24 個月回測對比腳本。

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


def main():
    parser = argparse.ArgumentParser(description="三種倉位方法對比回測")
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    config = load_config()
    methods = ["vol_inverse", "mean_variance", "risk_parity"]
    results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"倉位方法：{method}")
        print("=" * 60)
        with get_session() as session:
            res = run_backtest(
                config=config,
                db_session=session,
                backtest_months=args.months,
                position_sizing="vol_inverse",   # 舊參數保持不變
                position_sizing_method=method,   # 新參數決定實際方法
                fast_mode=args.fast,
            )
        results[method] = res.get("summary", {})

    # ── 輸出對比表格 ──
    print("\n" + "=" * 70)
    print("三種倉位方法回測對比（24 個月 Walk-Forward）")
    print("=" * 70)
    header = f"{'方法':<16} {'Sharpe':>8} {'MDD':>8} {'年化報酬':>10} {'累積報酬':>10}"
    print(header)
    print("-" * 70)
    for method in methods:
        s = results[method]
        sharpe = s.get("sharpe_ratio", "N/A")
        mdd = s.get("max_drawdown", "N/A")
        ann = s.get("annualized_return", "N/A")
        tot = s.get("total_return", "N/A")
        print(
            f"{method:<16} "
            f"{sharpe:>8.3f} "
            f"{mdd:>8.2%} "
            f"{ann:>10.2%} "
            f"{tot:>10.2%}"
        )
    print("=" * 70)

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
