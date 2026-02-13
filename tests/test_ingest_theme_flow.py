from __future__ import annotations

import math

import pandas as pd

from skills.ingest_theme_flow import _build_theme_flow, _chunk_records, _to_mysql_safe_records


def test_to_mysql_safe_records_replaces_nan_with_none():
    rows = []
    for i in range(1, 22):
        rows.append(
            {
                "stock_id": "1101",
                "trading_date": f"2026-01-{i:02d}",
                "close": 100 + i,
                "volume": 1000 + i * 10,
                "industry_category": "水泥工業",
            }
        )
    for i in range(1, 22):
        rows.append(
            {
                "stock_id": "1216",
                "trading_date": f"2026-01-{i:02d}",
                "close": 80 + i,
                "volume": 1200 + i * 8,
                "industry_category": "食品工業",
            }
        )

    raw = pd.DataFrame(rows)
    raw.loc[0, "industry_category"] = None
    raw.loc[5, "close"] = None

    theme_df = _build_theme_flow(raw)
    records = _to_mysql_safe_records(theme_df)

    assert records, "records should not be empty"
    for row in records:
        for value in row.values():
            assert not (isinstance(value, float) and math.isnan(value))


def test_chunk_records_splits_into_batches():
    records = [{"i": i} for i in range(5)]
    chunks = list(_chunk_records(records, 2))
    assert [len(c) for c in chunks] == [2, 2, 1]
