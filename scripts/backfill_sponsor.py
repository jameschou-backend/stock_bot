"""Sponsor 資料集回補腳本

回補 FinMind Sponsor 計劃專屬資料集：
  fear_greed   — CNN 恐懼貪婪指數（最輕量，建議先測試）
  gov_bank     — 官股銀行買賣超
  holding_dist — 持股分級週報
  broker_trades — 分點券商聚合（資料量大）
  kbar_features — 分鐘K線日內特徵（資料量最大）
  all          — 依序執行全部（優先序 P5→P4→P2→P1→P3）

用法：
  python scripts/backfill_sponsor.py --dataset fear_greed
  python scripts/backfill_sponsor.py --dataset all
  make backfill-sponsor
"""
from __future__ import annotations

import argparse
import sys
import time

# 確保 project root 在 path 中
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_config
from app.db import get_session


DATASETS = {
    "fear_greed": ("skills.ingest_fear_greed", "ingest_fear_greed"),
    "gov_bank": ("skills.ingest_gov_bank", "ingest_gov_bank"),
    "holding_dist": ("skills.ingest_holding_dist", "ingest_holding_dist"),
    "broker_trades": ("skills.ingest_broker_trades", "ingest_broker_trades"),
    "kbar_features": ("skills.ingest_kbar_features", "ingest_kbar_features"),
    # 2026-04-23 新增：價值因子 / 借券 / 季報
    "per": ("skills.ingest_per", "ingest_per"),
    "securities_lending": ("skills.ingest_securities_lending", "ingest_securities_lending"),
    "quarterly_fundamental": ("skills.ingest_quarterly_fundamental", "ingest_quarterly_fundamental"),
}

# 執行順序（all 模式）：依資料量由小到大，避免大任務失敗影響小任務
ALL_ORDER = [
    "fear_greed", "gov_bank", "holding_dist",
    "broker_trades", "kbar_features",
    "per", "securities_lending", "quarterly_fundamental",
]


def _run_one(name: str, config) -> dict:
    import importlib
    module_path, module_name = DATASETS[name]
    mod = importlib.import_module(module_path)
    t0 = time.perf_counter()
    with get_session() as session:
        result = mod.run(config, session)
    elapsed = time.perf_counter() - t0
    rows = result.get("rows", 0) if isinstance(result, dict) else "?"
    print(f"  ✅ {name}: {rows} 筆（{elapsed:.1f}s）", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="回補 FinMind Sponsor 資料集")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        required=True,
        help="要回補的資料集（all=全部依序執行）",
    )
    args = parser.parse_args()

    config = load_config()

    if args.dataset == "all":
        print(f"[backfill_sponsor] 依序回補 {len(ALL_ORDER)} 個 Sponsor 資料集", flush=True)
        for name in ALL_ORDER:
            print(f"\n[backfill_sponsor] → {name}", flush=True)
            try:
                _run_one(name, config)
            except Exception as exc:
                print(f"  ⚠️  {name} 失敗（{exc}），繼續下一個", flush=True)
        print("\n[backfill_sponsor] 完成", flush=True)
    else:
        print(f"[backfill_sponsor] 回補 {args.dataset}", flush=True)
        _run_one(args.dataset, config)


if __name__ == "__main__":
    main()
