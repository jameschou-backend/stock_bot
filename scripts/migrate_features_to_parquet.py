"""一次性遷移腳本：MySQL Feature table → 年份 Parquet Feature Store。

執行方式：
    python scripts/migrate_features_to_parquet.py [--force]

選項：
    --force     強制覆蓋已存在的年份 Parquet 檔案（預設跳過已存在年份）

說明：
    每年約 400k 行，序列讀取避免 OOM。
    遷移完成後 build_features / train_ranker / daily_pick 會自動改走 Parquet 路徑。
    MySQL Feature 表保留完整（dual-write），可隨時 rollback。
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
logger = logging.getLogger("migrate_features")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate features MySQL → Parquet")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing year Parquet files (default: skip)",
    )
    args = parser.parse_args()

    from app.db import get_session
    from skills.feature_store import FeatureStore

    fs = FeatureStore()
    logger.info(
        "FeatureStore 儲存路徑: %s（force=%s）", fs.store_dir, args.force
    )

    with get_session() as session:
        fs.migrate_from_mysql(
            session,
            skip_existing_years=not args.force,
        )

    # 驗證結果
    max_date = fs.get_max_date()
    all_files = fs._all_year_paths()
    logger.info(
        "遷移完成：%d 個年份檔案，最新日期 = %s",
        len(all_files),
        max_date,
    )
    for p in all_files:
        size_mb = p.stat().st_size / 1024 / 1024
        logger.info("  %s  (%.1f MB)", p.name, size_mb)


if __name__ == "__main__":
    main()
