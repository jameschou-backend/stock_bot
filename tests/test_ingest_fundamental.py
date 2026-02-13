from __future__ import annotations

import math

import pandas as pd

from datetime import date

from skills.ingest_fundamental import (
    _first_day_next_month,
    _normalize_fundamentals,
    _normalize_stock_ids,
    _to_mysql_safe_records,
)


def test_to_mysql_safe_records_replaces_nan_with_none():
    raw = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-02-01"],
            "stock_id": ["1101", "1101"],
            "revenue": [1000, 1200],
        }
    )
    norm = _normalize_fundamentals(raw)
    records = _to_mysql_safe_records(norm)

    assert records, "records should not be empty"
    for row in records:
        for key in ["revenue_last_month", "revenue_last_year", "revenue_mom", "revenue_yoy"]:
            value = row.get(key)
            assert not (isinstance(value, float) and math.isnan(value)), f"{key} still has NaN"


def test_first_day_next_month_handles_year_boundary():
    assert _first_day_next_month(date(2026, 2, 11)) == date(2026, 3, 1)
    assert _first_day_next_month(date(2026, 12, 1)) == date(2027, 1, 1)


def test_normalize_stock_ids_keeps_only_tw_4_digit_codes():
    stock_ids = ["2330", "0050", "abc", "1101", "2330", "12345"]
    assert _normalize_stock_ids(stock_ids) == ["0050", "1101", "2330"]
