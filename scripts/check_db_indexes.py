#!/usr/bin/env python3
"""DB Index Checker

檢查 features / raw_prices / picks / raw_institutional / jobs 表的重要索引是否存在，
並輸出缺少的 CREATE INDEX 語句。

用法：
    python scripts/check_db_indexes.py
    make check-index
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text

from app.db import get_engine

# ── 期望存在的索引 (table, leading_columns) ──────────────────────────────────
# 以 leading_columns 中的欄位為首碼即視為「存在」
EXPECTED_INDEXES: list[tuple[str, list[str]]] = [
    ("raw_prices",          ["stock_id", "trading_date"]),
    ("raw_prices",          ["trading_date"]),
    ("raw_prices",          ["trading_date", "stock_id"]),  # covering: GROUP BY trading_date COUNT(DISTINCT stock_id)
    ("features",            ["stock_id", "trading_date"]),
    ("features",            ["trading_date"]),
    ("picks",               ["pick_date"]),
    ("picks",               ["stock_id"]),
    ("raw_institutional",   ["stock_id", "trading_date"]),
    ("raw_institutional",   ["trading_date"]),
    ("raw_institutional",   ["trading_date", "stock_id"]),  # covering: GROUP BY queries
    ("raw_margin_short",    ["trading_date", "stock_id"]),  # covering: GROUP BY queries
    ("jobs",                ["started_at"]),
    ("jobs",                ["status"]),
]


def _fetch_existing_indexes(engine) -> dict[str, list[tuple[str, ...]]]:
    """回傳 {table_name: [ordered_columns_tuple, ...]} 的現有索引清單。"""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT TABLE_NAME,
                   INDEX_NAME,
                   GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) AS cols
            FROM INFORMATION_SCHEMA.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
            GROUP BY TABLE_NAME, INDEX_NAME
        """)).fetchall()

    result: dict[str, list[tuple[str, ...]]] = {}
    for table, _idx_name, cols_str in rows:
        cols = tuple(c.strip().lower() for c in cols_str.split(","))
        result.setdefault(table, []).append(cols)
    return result


def check_indexes() -> int:
    engine = get_engine()
    existing = _fetch_existing_indexes(engine)

    print("\n=== DB Index Check ===\n")
    missing: list[tuple[str, list[str]]] = []

    for table, expected_cols in EXPECTED_INDEXES:
        expected = tuple(c.lower() for c in expected_cols)
        table_indexes = existing.get(table, [])
        # An index satisfies if its leading columns match expected_cols
        found = any(idx[: len(expected)] == expected for idx in table_indexes)
        status = "✅" if found else "❌ MISSING"
        print(f"  {status}  {table}  ({', '.join(expected_cols)})")
        if not found:
            missing.append((table, expected_cols))

    ok_count = len(EXPECTED_INDEXES) - len(missing)
    print(f"\n結果：{ok_count} OK, {len(missing)} MISSING\n")

    if missing:
        print("建議執行的 CREATE INDEX 語句：")
        for table, cols in missing:
            idx_name = f"idx_{table}_{'_'.join(cols)}"
            col_str = ", ".join(cols)
            print(f"  CREATE INDEX {idx_name} ON {table} ({col_str});")
        print()
        return 1

    print("所有重要索引均已存在。")
    return 0


if __name__ == "__main__":
    sys.exit(check_indexes())
