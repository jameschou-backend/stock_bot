"""Tests for skills/feature_store.py."""
from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skills.feature_store import FeatureStore


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_df(stock_ids, dates, feat_val=1.0) -> pd.DataFrame:
    rows = []
    for sid in stock_ids:
        for d in dates:
            rows.append({"stock_id": sid, "trading_date": d, "feat_a": feat_val, "feat_b": feat_val * 2})
    return pd.DataFrame(rows)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_write_and_read_basic():
    with tempfile.TemporaryDirectory() as tmp:
        fs = FeatureStore(store_dir=Path(tmp))
        df = _make_df(["0050", "2330"], [date(2024, 1, 2), date(2024, 1, 3)])
        fs.write(df)

        result = fs.read(date(2024, 1, 1), date(2024, 12, 31))
        assert len(result) == 4
        assert set(result["stock_id"]) == {"0050", "2330"}
        assert "feat_a" in result.columns


def test_write_creates_year_partition():
    with tempfile.TemporaryDirectory() as tmp:
        fs = FeatureStore(store_dir=Path(tmp))
        df = _make_df(["0050"], [date(2023, 12, 31), date(2024, 1, 1)])
        fs.write(df)

        files = list(Path(tmp).glob("features_*.parquet"))
        years = {int(f.stem.split("_")[1]) for f in files}
        assert 2023 in years
        assert 2024 in years


def test_get_max_date_empty():
    with tempfile.TemporaryDirectory() as tmp:
        fs = FeatureStore(store_dir=Path(tmp))
        assert fs.get_max_date() is None


def test_get_max_date():
    with tempfile.TemporaryDirectory() as tmp:
        fs = FeatureStore(store_dir=Path(tmp))
        df = _make_df(["0050"], [date(2024, 3, 1), date(2024, 3, 5)])
        fs.write(df)
        assert fs.get_max_date() == date(2024, 3, 5)


def test_upsert_overwrites_existing():
    with tempfile.TemporaryDirectory() as tmp:
        fs = FeatureStore(store_dir=Path(tmp))
        df1 = _make_df(["0050"], [date(2024, 1, 2)], feat_val=1.0)
        fs.write(df1)

        df2 = _make_df(["0050"], [date(2024, 1, 2)], feat_val=99.0)
        fs.write(df2)

        result = fs.read(date(2024, 1, 2), date(2024, 1, 2))
        assert len(result) == 1
        assert float(result["feat_a"].iloc[0]) == 99.0


def test_delete_from_partial_year():
    with tempfile.TemporaryDirectory() as tmp:
        fs = FeatureStore(store_dir=Path(tmp))
        df = _make_df(["0050"], [date(2024, 1, 2), date(2024, 6, 1), date(2024, 12, 1)])
        fs.write(df)

        fs.delete_from(date(2024, 6, 1))

        result = fs.read(date(2024, 1, 1), date(2025, 1, 1))
        assert len(result) == 1
        assert result["trading_date"].iloc[0] == date(2024, 1, 2)


def test_delete_from_full_year():
    with tempfile.TemporaryDirectory() as tmp:
        fs = FeatureStore(store_dir=Path(tmp))
        df = _make_df(["0050"], [date(2023, 6, 1), date(2024, 1, 2)])
        fs.write(df)

        fs.delete_from(date(2024, 1, 1))

        result = fs.read(date(2024, 1, 1), date(2025, 1, 1))
        assert len(result) == 0

        # 2023 data intact
        result_2023 = fs.read(date(2023, 1, 1), date(2023, 12, 31))
        assert len(result_2023) == 1


def test_get_distinct_dates():
    with tempfile.TemporaryDirectory() as tmp:
        fs = FeatureStore(store_dir=Path(tmp))
        dates = [date(2024, 1, d) for d in [2, 3, 4, 5, 8]]
        df = _make_df(["0050", "2330"], dates)
        fs.write(df)

        last3 = fs.get_distinct_dates(3)
        assert last3 == [date(2024, 1, 8), date(2024, 1, 5), date(2024, 1, 4)]


def test_read_feature_columns_projection():
    with tempfile.TemporaryDirectory() as tmp:
        fs = FeatureStore(store_dir=Path(tmp))
        df = _make_df(["0050"], [date(2024, 1, 2)])
        fs.write(df)

        result = fs.read(date(2024, 1, 1), date(2024, 12, 31), feature_columns=["feat_a"])
        assert "feat_a" in result.columns
        assert "feat_b" not in result.columns


def test_read_empty_when_no_files():
    with tempfile.TemporaryDirectory() as tmp:
        fs = FeatureStore(store_dir=Path(tmp))
        result = fs.read(date(2024, 1, 1), date(2024, 12, 31))
        assert result.empty


def test_schema_evolution_cross_year_read():
    """跨年讀取時 DuckDB union_by_name=true 自動以 NULL 填補舊年缺失欄。"""
    with tempfile.TemporaryDirectory() as tmp:
        fs = FeatureStore(store_dir=Path(tmp))
        # 舊資料（只有 feat_a）
        df_old = pd.DataFrame([
            {"stock_id": "0050", "trading_date": date(2023, 6, 1), "feat_a": 1.0}
        ])
        fs.write(df_old)

        # 新資料（feat_a + feat_new）
        df_new = pd.DataFrame([
            {"stock_id": "0050", "trading_date": date(2024, 1, 2), "feat_a": 2.0, "feat_new": 99.0}
        ])
        fs.write(df_new)

        # 跨年讀取：2023 row 的 feat_new 應為 NaN（DuckDB union_by_name 填 NULL）
        result_all = fs.read(date(2023, 1, 1), date(2024, 12, 31))
        assert "feat_new" in result_all.columns
        row_2023 = result_all[result_all["trading_date"] == date(2023, 6, 1)]
        assert len(row_2023) == 1
        assert pd.isna(row_2023["feat_new"].iloc[0])

        # 2024 row 的 feat_new 應有值
        row_2024 = result_all[result_all["trading_date"] == date(2024, 1, 2)]
        assert float(row_2024["feat_new"].iloc[0]) == 99.0


def test_schema_evolution_single_year_no_new_col():
    """讀取單一舊年份時不含後期新增的欄（此為預期行為，cross-year read 才會有）。"""
    with tempfile.TemporaryDirectory() as tmp:
        fs = FeatureStore(store_dir=Path(tmp))
        df_old = pd.DataFrame([
            {"stock_id": "0050", "trading_date": date(2023, 6, 1), "feat_a": 1.0}
        ])
        fs.write(df_old)

        df_new = pd.DataFrame([
            {"stock_id": "0050", "trading_date": date(2024, 1, 2), "feat_a": 2.0, "feat_new": 99.0}
        ])
        fs.write(df_new)

        # 單獨讀取 2023：不含 feat_new（只有 2023 parquet 檔）
        result_2023 = fs.read(date(2023, 1, 1), date(2023, 12, 31))
        assert "feat_a" in result_2023.columns
        assert len(result_2023) == 1


def test_atomic_write_no_tmp_file_left():
    """成功 write 後不應留下 .tmp 檔案。"""
    with tempfile.TemporaryDirectory() as tmp:
        fs = FeatureStore(store_dir=Path(tmp))
        df = _make_df(["0050"], [date(2024, 1, 2)])
        fs.write(df)

        tmp_files = list(Path(tmp).glob("*.tmp"))
        assert len(tmp_files) == 0
