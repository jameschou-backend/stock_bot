"""One-shot backfill institutional / margin_short / per via TWSE.

繞過 FinMind batch query bug。利用既有 ingest 模組的 _run_twse() 邏輯
（TWSE legacy T86 / MI_MARGN / BWIBBU_d 接受 date 參數，一日一 call）。

用法：
    # 補全部三個（從各表的 latest+1 開始）
    python scripts/backfill_via_twse.py

    # 只補單一資料
    python scripts/backfill_via_twse.py --only institutional

執行速度：每個 dataset 約 1-2 分鐘（12 個交易日 × 2 個 endpoint × 1.5s/req）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# 強制走 TWSE 後端（即使 .env 沒設）
os.environ.setdefault("INGEST_INSTITUTIONAL_SOURCE", "twse")
os.environ.setdefault("INGEST_MARGIN_SHORT_SOURCE", "twse")
os.environ.setdefault("INGEST_PER_SOURCE", "twse")

from app.config import load_config
from app.db import get_session
from skills import ingest_institutional, ingest_margin_short, ingest_per


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        choices=["institutional", "margin_short", "per"],
        default=None,
        help="只跑單一資料；不給則全跑",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config()

    targets = [
        ("institutional", ingest_institutional),
        ("margin_short", ingest_margin_short),
        ("per", ingest_per),
    ]
    if args.only:
        targets = [(n, m) for n, m in targets if n == args.only]

    all_ok = True
    for name, mod in targets:
        print(f"=== Backfill {name} via TWSE ===")
        # 強制 module 內部的 _resolve_source 讀到 'twse'
        env_key = mod.SOURCE_ENV
        os.environ[env_key] = "twse"
        resolved = mod._resolve_source()
        print(f"  source resolved: {resolved}")
        try:
            with get_session() as s:
                result = mod.run(config, s)
                print(f"  result: {result}")
        except Exception as exc:
            all_ok = False
            print(f"  ❌ ERROR: {exc}")
            import traceback
            traceback.print_exc()
        print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
