"""Year-partitioned Parquet feature store.

取代 MySQL Feature table 成為特徵資料的讀取主路徑。

儲存格式：
    artifacts/features/features_YYYY.parquet  （每年一個檔案，預先解析好的數值欄位）

優點：
  - 無 JSON 解析開銷（已預先 flatten）
  - DuckDB predicate pushdown：掃 10 年資料只讀目標年份檔案
  - 原子寫入（.tmp → rename）：不產生部分讀取問題
  - Schema 演進：新欄位 reindex 後舊資料以 NaN 填補

Public API:
    FeatureStore().write(df)                     — 追加 / upsert 行
    FeatureStore().read(start, end)              — DuckDB 日期範圍查詢
    FeatureStore().get_max_date()                — 最新 trading_date
    FeatureStore().get_distinct_dates(last_n)    — 最近 N 個交易日（降冪）
    FeatureStore().delete_from(from_date)        — 刪除 >= from_date 的資料
    FeatureStore().migrate_from_mysql(session)   — 一次性 MySQL → Parquet 遷移
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import List, Optional

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 可由環境變數覆蓋（測試時設 FEATURE_STORE_DIR=tests/fixtures/features）
_STORE_DIR = Path(os.environ.get("FEATURE_STORE_DIR", "artifacts/features"))

_META_COLS = frozenset(["stock_id", "trading_date"])


class FeatureStore:
    """Year-partitioned Parquet feature store.

    Thread-safety: 讀取可並發；寫入使用 atomic rename，避免 partial-read。
    """

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self.store_dir = Path(store_dir) if store_dir else _STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ── path helpers ─────────────────────────────────────────────────────────

    def _year_path(self, year: int) -> Path:
        return self.store_dir / f"features_{year}.parquet"

    def _all_year_paths(self) -> List[Path]:
        """Return sorted list of existing year parquet files."""
        return sorted(self.store_dir.glob("features_*.parquet"))

    # ── write ─────────────────────────────────────────────────────────────────

    def write(self, df: pd.DataFrame) -> None:
        """Upsert rows into year-partitioned Parquet files.

        Args:
            df: DataFrame with columns [stock_id, trading_date, *feature_cols].
                trading_date 可為 date、datetime 或 ISO 字串。
                新行在 (stock_id, trading_date) 衝突時覆蓋舊行。
        """
        if df.empty:
            return
        df = df.copy()
        df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
        df["_year"] = [d.year for d in df["trading_date"]]

        for year, year_df in df.groupby("_year"):
            year_df = year_df.drop(columns=["_year"]).reset_index(drop=True)
            self._write_year(int(year), year_df)

    def _write_year(self, year: int, new_df: pd.DataFrame) -> None:
        """Merge new_df into the year Parquet file, then rewrite atomically."""
        path = self._year_path(year)
        if path.exists():
            existing = pd.read_parquet(path)
            existing["trading_date"] = pd.to_datetime(existing["trading_date"]).dt.date

            # Schema evolution: new_df may introduce new columns
            all_cols = list(dict.fromkeys(list(existing.columns) + list(new_df.columns)))
            existing = existing.reindex(columns=all_cols)
            new_df = new_df.reindex(columns=all_cols)

            # Concatenate; new_df rows win on duplicates (keep="last")
            combined = (
                pd.concat([existing, new_df], ignore_index=True)
                .drop_duplicates(subset=["stock_id", "trading_date"], keep="last")
                .sort_values(["trading_date", "stock_id"])
                .reset_index(drop=True)
            )
        else:
            combined = (
                new_df.sort_values(["trading_date", "stock_id"])
                .reset_index(drop=True)
            )

        # Atomic write: write to .tmp then rename
        tmp_path = path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp_path, index=False)
        tmp_path.rename(path)
        logger.debug(
            "[FeatureStore] wrote year=%d: %d rows → %s", year, len(combined), path.name
        )

    # ── read ──────────────────────────────────────────────────────────────────

    def read(
        self,
        start_date: date,
        end_date: date,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Read features for [start_date, end_date] using DuckDB predicate pushdown.

        Args:
            start_date: inclusive lower bound。
            end_date:   inclusive upper bound。
            feature_columns: 若指定，僅回傳這些特徵欄（+ stock_id, trading_date）。

        Returns:
            DataFrame。無資料時回傳空 DataFrame。
        """
        start_year = start_date.year
        end_year = end_date.year
        paths = [
            p for p in self._all_year_paths()
            if start_year <= int(p.stem.split("_")[1]) <= end_year
        ]
        if not paths:
            return pd.DataFrame()

        col_expr = "*"
        if feature_columns is not None:
            all_cols = sorted(_META_COLS | set(feature_columns))
            col_expr = ", ".join(f'"{c}"' for c in all_cols)

        path_list = [str(p) for p in paths]
        con = duckdb.connect()
        try:
            # union_by_name=true：跨年份 schema 不同時（新增欄位），
            # DuckDB 自動以 NULL 填補缺失欄，實現零成本 schema 演進
            df = con.execute(
                f"SELECT {col_expr} FROM read_parquet(?, union_by_name=true) "
                f"WHERE trading_date >= ? AND trading_date <= ?",
                [path_list, str(start_date), str(end_date)],
            ).df()
        finally:
            con.close()

        if df.empty:
            return df
        df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
        return df

    # ── get_max_date ──────────────────────────────────────────────────────────

    def get_max_date(self) -> Optional[date]:
        """Return the latest trading_date across all year files, or None if no files."""
        paths = self._all_year_paths()
        if not paths:
            return None
        # Only the last year file needs scanning
        last_path = paths[-1]
        con = duckdb.connect()
        try:
            result = con.execute(
                "SELECT MAX(trading_date) FROM read_parquet(?, union_by_name=true)",
                [str(last_path)],
            ).fetchone()
        finally:
            con.close()
        if result is None or result[0] is None:
            return None
        val = result[0]
        if isinstance(val, str):
            return date.fromisoformat(val)
        return val

    def get_distinct_dates(self, last_n: int) -> List[date]:
        """Return the last N distinct trading dates in descending order."""
        paths = self._all_year_paths()
        if not paths:
            return []
        path_list = sorted([str(p) for p in paths])
        con = duckdb.connect()
        try:
            rows = con.execute(
                "SELECT DISTINCT trading_date FROM read_parquet(?, union_by_name=true) "
                "ORDER BY trading_date DESC LIMIT ?",
                [path_list, last_n],
            ).fetchall()
        finally:
            con.close()
        results: List[date] = []
        for (val,) in rows:
            if isinstance(val, str):
                results.append(date.fromisoformat(val))
            else:
                results.append(val)
        return results

    # ── delete_from ───────────────────────────────────────────────────────────

    def delete_from(self, from_date: date) -> None:
        """Delete all rows with trading_date >= from_date.

        - 整年 >= from_date：直接刪除檔案
        - 跨年邊界（from_date 的年份）：部分重寫
        """
        from_year = from_date.year
        for path in self._all_year_paths():
            year = int(path.stem.split("_")[1])
            if year < from_year:
                continue
            if year > from_year:
                path.unlink()
                logger.info(
                    "[FeatureStore] deleted %s (year %d entirely >= %s)",
                    path.name, year, from_date,
                )
                continue
            # year == from_year: partial rewrite
            df = pd.read_parquet(path)
            df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
            df = df[df["trading_date"] < from_date]
            if df.empty:
                path.unlink()
                logger.info(
                    "[FeatureStore] deleted %s (no rows remain after filtering < %s)",
                    path.name, from_date,
                )
            else:
                tmp_path = path.with_suffix(".parquet.tmp")
                df.to_parquet(tmp_path, index=False)
                tmp_path.rename(path)
                logger.info(
                    "[FeatureStore] rewrote %s: %d rows remain (deleted >= %s)",
                    path.name, len(df), from_date,
                )

    # ── migrate_from_mysql ────────────────────────────────────────────────────

    def migrate_from_mysql(self, db_session, skip_existing_years: bool = True) -> None:
        """One-time migration: MySQL Feature table → year-partitioned Parquet.

        年份串行讀取（每年約 400k 行），避免一次載入全部 10 年 OOM。

        Args:
            db_session: SQLAlchemy session。
            skip_existing_years: True（預設）→ 跳過已有 Parquet 的年份，
                                  False → 強制覆蓋所有年份。
        """
        from sqlalchemy import select, func
        from app.models import Feature
        from skills.feature_utils import parse_features_json

        min_date = db_session.execute(select(func.min(Feature.trading_date))).scalar()
        max_date = db_session.execute(select(func.max(Feature.trading_date))).scalar()
        if min_date is None:
            logger.info("[FeatureStore.migrate] MySQL Feature table is empty — skipping.")
            return

        start_year = min_date.year
        end_year = max_date.year
        logger.info(
            "[FeatureStore.migrate] migrating %d ~ %d from MySQL …", start_year, end_year
        )

        for year in range(start_year, end_year + 1):
            year_path = self._year_path(year)
            if skip_existing_years and year_path.exists():
                logger.info(
                    "[FeatureStore.migrate] year=%d — %s already exists, skipping.",
                    year, year_path.name,
                )
                continue

            logger.info("[FeatureStore.migrate] year=%d — reading from MySQL …", year)
            t0 = time.time()
            stmt = (
                select(Feature.stock_id, Feature.trading_date, Feature.features_json)
                .where(Feature.trading_date >= date(year, 1, 1))
                .where(Feature.trading_date <= date(year, 12, 31))
                .order_by(Feature.trading_date, Feature.stock_id)
            )
            raw = pd.read_sql(stmt, db_session.get_bind())
            if raw.empty:
                logger.info("[FeatureStore.migrate] year=%d — no MySQL data, skipping.", year)
                continue
            raw["trading_date"] = pd.to_datetime(raw["trading_date"]).dt.date

            parsed = parse_features_json(raw["features_json"])
            parsed = parsed.replace([np.inf, -np.inf], np.nan)
            feat_df = pd.concat(
                [
                    raw[["stock_id", "trading_date"]].reset_index(drop=True),
                    parsed.reset_index(drop=True),
                ],
                axis=1,
            )
            del raw, parsed

            tmp_path = year_path.with_suffix(".parquet.tmp")
            feat_df.to_parquet(tmp_path, index=False)
            tmp_path.rename(year_path)
            logger.info(
                "[FeatureStore.migrate] year=%d — %d rows → %s (%.1fs)",
                year, len(feat_df), year_path.name, time.time() - t0,
            )
            del feat_df
