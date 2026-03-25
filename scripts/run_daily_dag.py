"""DAG 版每日 pipeline 入口。

等同 run_daily.py，但使用 DAGExecutor 並行執行獨立的 ingest 節點，
預估縮短執行時間 30-40%（ingest 層並行化）。

使用方式：
    python scripts/run_daily_dag.py [--skip-ingest] [--dry-run]

選項：
    --skip-ingest   跳過所有 ingest 節點，直接從 data_quality 開始
                    （資料已是最新時使用，可節省 API 呼叫）
    --dry-run       顯示 DAG 結構與執行順序但不實際執行
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── 確保專案根目錄在 sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_daily_dag")


def _dry_run(skip_ingest: bool) -> None:
    """顯示 DAG 結構（節點清單與執行層次）而不實際執行。"""
    from pipelines.dag_daily import build_dag
    executor = build_dag(skip_ingest=skip_ingest)
    levels = executor._topological_levels()
    print(f"\n{'='*60}")
    print(f"DAG 結構（{'skip-ingest' if skip_ingest else 'full'}）")
    print(f"{'='*60}")
    print(f"  節點總數: {len(executor.nodes)}")
    print(f"  執行層數: {len(levels)}")
    print()
    for i, level in enumerate(levels, 1):
        parallel = "（可並行）" if len(level) > 1 else ""
        print(f"  Layer {i}{parallel}:")
        for name in level:
            node = executor.nodes[name]
            flags = []
            if node.optional:
                flags.append("optional")
            if node.condition is not None:
                flags.append("conditional")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            deps_str = f"  deps={node.deps}" if node.deps else ""
            print(f"    - {name}{flag_str}{deps_str}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DAG 版每日 pipeline（並行 ingest）"
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="跳過所有 ingest 節點（資料已最新時使用）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只顯示 DAG 結構，不實際執行",
    )
    args = parser.parse_args()

    if args.dry_run:
        _dry_run(skip_ingest=args.skip_ingest)
        return

    from app.config import load_config
    from pipelines.dag_daily import build_dag

    config = load_config()
    executor = build_dag(skip_ingest=args.skip_ingest)

    logger.info(
        "開始執行 DAG pipeline（skip_ingest=%s）…", args.skip_ingest
    )
    results = executor.run(config)

    # 印出執行摘要
    ok = sum(1 for r in results.values() if r.status.value == "SUCCESS")
    total = len(results)
    logger.info("DAG pipeline 完成：%d/%d 節點成功", ok, total)


if __name__ == "__main__":
    main()
