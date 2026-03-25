"""DuckDB-accelerated data access layer with data-date-aware Parquet cache.

全量 10 年資料從 MySQL 載入一次，存為固定路徑 Parquet。
後續查詢改用 DuckDB 的謂語下推（predicate pushdown）按日期範圍高效讀取，
無論回測期間長短（1 年/5 年/10 年）均不需要重讀 MySQL。

Public API:
    get_prices(db_session, start_date, end_date)           -> pd.DataFrame
    get_features(db_session, start_date, end_date, ...)    -> pd.DataFrame
    get_labels(db_session, start_date, end_date)           -> pd.DataFrame
    invalidate()                                           -> None（強制重建快取）
    warm_up(db_session)                                    -> None（預建所有快取）

快取路徑:
    artifacts/cache/prices.parquet
    artifacts/cache/features.parquet
    artifacts/cache/labels.parquet

快取更新策略：比對 cache 的 max(trading_date) 與資料來源的 max date，
來源有新資料就重建（不依賴檔案時間），pipeline 跑完後 daily-c 立即拿到最新資料。
"""
from __future__ import annotations

import gc
import time
from datetime import date
from pathlib import Path
from typing import List, Optional

import duckdb
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Feature, Label, RawPrice
from skills.feature_utils import (
    parse_features_json as _parse_features_json_shared,
    filter_schema_valid_rows as _filter_schema_valid_rows_shared,
)

# ── 路徑常數 ─────────────────────────────────────────────────────────────────
CACHE_DIR        = Path("artifacts/cache")
PRICES_PARQUET   = CACHE_DIR / "prices.parquet"
FEATURES_PARQUET = CACHE_DIR / "features.parquet"
LABELS_PARQUET   = CACHE_DIR / "labels.parquet"

_TTL_SECONDS = 86_400  # 24 小時（fallback 用，正常走資料日期比對）


# ── 快取有效性 ────────────────────────────────────────────────────────────────
def _parquet_max_date(path: Path) -> Optional[str]:
    """快速讀取 parquet 內的 max(trading_date)，利用 DuckDB 欄位統計避免全掃。"""
    if not path.exists():
        return None
    try:
        con = duckdb.connect()
        result = con.execute(
            "SELECT max(trading_date) FROM read_parquet(?)", [str(path)]
        ).fetchone()[0]
        con.close()
        return str(result) if result is not None else None
    except Exception:
        return None


def _is_fresh(path: Path) -> bool:
    """Fallback：檔案存在且未超過 TTL（資料日期比對失敗時使用）。"""
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < _TTL_SECONDS


# ── 特徵解析（委派給 feature_utils，統一實作）────────────────────────────────
def _parse_features_json(series: pd.Series) -> pd.DataFrame:
    """委派給 feature_utils.parse_features_json（含 orjson 加速）。"""
    return _parse_features_json_shared(series)


def _parse_and_schema_filter(raw_feat_df: pd.DataFrame) -> pd.DataFrame:
    """解析 features_json → 個別欄位，套用 schema filter（>50% 非 NaN），float32。"""
    parsed = _parse_features_json(raw_feat_df["features_json"])
    parsed = parsed.replace([np.inf, -np.inf], np.nan)
    feat_df = raw_feat_df[["stock_id", "trading_date"]].reset_index(drop=True)
    feat_df = pd.concat([feat_df, parsed.reset_index(drop=True)], axis=1)
    del parsed

    num_cols = [c for c in feat_df.columns if c not in ("stock_id", "trading_date")]
    if num_cols:
        thr = max(1, int(len(num_cols) * 0.50))
        mask = feat_df[num_cols].notna().sum(axis=1) >= thr
        n_drop = int((~mask).sum())
        if n_drop:
            print(f"  [data_store] schema filter: dropped {n_drop:,} stale rows", flush=True)
        feat_df = feat_df.loc[mask].reset_index(drop=True)
        feat_df[num_cols] = feat_df[num_cols].astype("float32")

    return feat_df


# ── 快取建立 ──────────────────────────────────────────────────────────────────
def _build_prices(db_session: Session) -> None:
    print("  [data_store] building prices.parquet (full history) …", flush=True)
    t0 = time.time()
    stmt = (
        select(
            RawPrice.stock_id, RawPrice.trading_date,
            RawPrice.open, RawPrice.high, RawPrice.low,
            RawPrice.close, RawPrice.volume,
        )
        .order_by(RawPrice.trading_date, RawPrice.stock_id)
    )
    df = pd.read_sql(stmt, db_session.get_bind())
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_parquet(PRICES_PARQUET, index=False)
    print(
        f"  [data_store] prices saved: {len(df):,} rows "
        f"({time.time()-t0:.1f}s) → {PRICES_PARQUET.name}",
        flush=True,
    )


def _build_features(db_session: Session) -> None:
    """Build the flat features cache.

    優先從 FeatureStore（年份 Parquet）讀取已解析資料，
    大幅省去 JSON 解析開銷（4M 行 × 62 特徵 ≈ 節省 60-90s）。
    若 FeatureStore 無資料（首次使用前），fallback 至 MySQL。
    """
    print("  [data_store] building features.parquet …", flush=True)
    t0 = time.time()

    # ── 嘗試從 FeatureStore（年份 Parquet）直接讀取 ──
    try:
        from skills.feature_store import FeatureStore
        from datetime import date as _date
        _fs = FeatureStore()
        _max_date = _fs.get_max_date()
        if _max_date is not None:
            print(
                f"  [data_store] FeatureStore found (max={_max_date}), "
                "reading from year-partitioned Parquet …",
                flush=True,
            )
            # 讀取全部歷史（2000-01-01 → _max_date 涵蓋全部年份）
            feat_df = _fs.read(_date(2000, 1, 1), _max_date)
            feat_df["trading_date"] = pd.to_datetime(feat_df["trading_date"]).dt.date
            # 確保 float32 型別（與原快取行為一致）
            num_cols = [c for c in feat_df.columns if c not in ("stock_id", "trading_date")]
            if num_cols:
                feat_df[num_cols] = feat_df[num_cols].astype("float32")
            feat_df = feat_df.sort_values(
                ["trading_date", "stock_id"]
            ).reset_index(drop=True)
            feat_df.to_parquet(FEATURES_PARQUET, index=False)
            print(
                f"  [data_store] features saved (from FeatureStore): "
                f"{len(feat_df):,} rows × {len(num_cols)} features "
                f"({time.time()-t0:.1f}s total) → {FEATURES_PARQUET.name}",
                flush=True,
            )
            return
    except Exception as _exc:
        print(
            f"  [data_store] FeatureStore read failed ({_exc}), "
            "falling back to MySQL …",
            flush=True,
        )

    # ── Fallback：MySQL（FeatureStore 尚未初始化時使用）──
    print("  [data_store] reading features from MySQL (fallback) …", flush=True)
    stmt = (
        select(Feature.stock_id, Feature.trading_date, Feature.features_json)
        .order_by(Feature.trading_date, Feature.stock_id)
    )
    raw = pd.read_sql(stmt, db_session.get_bind())
    raw["trading_date"] = pd.to_datetime(raw["trading_date"]).dt.date
    print(f"  [data_store] db_load done ({time.time()-t0:.1f}s), parsing …", flush=True)

    t1 = time.time()
    feat_df = _parse_and_schema_filter(raw)
    del raw
    gc.collect()
    print(f"  [data_store] parse done ({time.time()-t1:.1f}s), saving …", flush=True)

    feat_df.to_parquet(FEATURES_PARQUET, index=False)
    print(
        f"  [data_store] features saved (from MySQL): {len(feat_df):,} rows × "
        f"{len(feat_df.columns)-2} features "
        f"({time.time()-t0:.1f}s total) → {FEATURES_PARQUET.name}",
        flush=True,
    )


def _build_labels(db_session: Session) -> None:
    print("  [data_store] building labels.parquet (full history) …", flush=True)
    t0 = time.time()
    stmt = (
        select(Label.stock_id, Label.trading_date, Label.future_ret_h)
        .order_by(Label.trading_date, Label.stock_id)
    )
    df = pd.read_sql(stmt, db_session.get_bind())
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df.to_parquet(LABELS_PARQUET, index=False)
    print(
        f"  [data_store] labels saved: {len(df):,} rows "
        f"({time.time()-t0:.1f}s) → {LABELS_PARQUET.name}",
        flush=True,
    )


def _ensure(db_session: Session) -> None:
    """確保三個 parquet 存在且資料日期與來源一致，否則重建。

    比對邏輯：
      - prices / labels：比對 cache max_date vs DB max(trading_date)
      - features：比對 cache max_date vs FeatureStore max_date（年份 Parquet）
    只要來源有新資料（source_max > cache_max），就觸發重建，
    不依賴檔案時間——pipeline 跑完後 daily-c 立即拿到最新資料。
    """
    from sqlalchemy import text as _text
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Prices ──
    _cache_px = _parquet_max_date(PRICES_PARQUET)
    try:
        _src_px = str(db_session.execute(_text("SELECT max(trading_date) FROM raw_prices")).scalar() or "")
    except Exception:
        _src_px = None
    if not _cache_px or (_src_px and _cache_px < _src_px):
        if _cache_px and _src_px:
            print(f"  [data_store] prices cache stale ({_cache_px} < {_src_px}), rebuilding …", flush=True)
        _build_prices(db_session)

    # ── Features ──
    _cache_feat = _parquet_max_date(FEATURES_PARQUET)
    try:
        from skills.feature_store import FeatureStore as _FS
        _src_feat_raw = _FS().get_max_date()
        _src_feat = str(_src_feat_raw) if _src_feat_raw is not None else None
    except Exception:
        _src_feat = None
    if not _cache_feat or (_src_feat and _cache_feat < _src_feat):
        if _cache_feat and _src_feat:
            print(f"  [data_store] features cache stale ({_cache_feat} < {_src_feat}), rebuilding …", flush=True)
        _build_features(db_session)

    # ── Labels ──
    _cache_lbl = _parquet_max_date(LABELS_PARQUET)
    try:
        _src_lbl = str(db_session.execute(_text("SELECT max(trading_date) FROM labels")).scalar() or "")
    except Exception:
        _src_lbl = None
    if not _cache_lbl or (_src_lbl and _cache_lbl < _src_lbl):
        if _cache_lbl and _src_lbl:
            print(f"  [data_store] labels cache stale ({_cache_lbl} < {_src_lbl}), rebuilding …", flush=True)
        _build_labels(db_session)


# ── DuckDB 謂語下推查詢 ───────────────────────────────────────────────────────
def _duck_query(
    parquet_path: Path,
    start_date: date,
    end_date: date,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """DuckDB predicate-pushdown date-range query.

    DuckDB 的 read_parquet 支援 row-group 跳過與謂語下推，
    對大型 parquet（4M 行 × 58 欄）的 1 年切片查詢比 pd.read_parquet+filter 快 3-5×。

    Args:
        columns: 若指定，只回傳這些欄（+ stock_id, trading_date）。
                 None 表示回傳全部欄位。
    """
    col_expr = "*"
    if columns is not None:
        # 確保 meta 欄永遠保留
        all_cols = {"stock_id", "trading_date"} | set(columns)
        col_expr = ", ".join(f'"{c}"' for c in sorted(all_cols))

    con = duckdb.connect()
    try:
        df = con.execute(
            f"SELECT {col_expr} FROM read_parquet(?) "
            f"WHERE trading_date >= ? AND trading_date <= ?",
            [str(parquet_path), str(start_date), str(end_date)],
        ).df()
    finally:
        con.close()

    if "trading_date" in df.columns:
        df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date

    return df


# ── 公開 API ──────────────────────────────────────────────────────────────────
def get_prices(
    db_session: Session,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """OHLCV price data for the given date range.

    Columns: stock_id, trading_date, open, high, low, close, volume
    """
    _ensure(db_session)
    return _duck_query(PRICES_PARQUET, start_date, end_date)


def get_features(
    db_session: Session,
    start_date: date,
    end_date: date,
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Parsed feature data for the given date range.

    Returns columns already decoded from JSON, schema-filtered, stored as float32.
    feature_columns: optional list to restrict returned feature columns.
    """
    _ensure(db_session)
    # DuckDB 謂語下推（行）+ 選擇性欄位投影（列）
    duck_cols = feature_columns  # None → SELECT *
    df = _duck_query(FEATURES_PARQUET, start_date, end_date, columns=duck_cols)
    return df


def get_labels(
    db_session: Session,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Label data (future_ret_h) for the given date range.

    Columns: stock_id, trading_date, future_ret_h
    """
    _ensure(db_session)
    return _duck_query(LABELS_PARQUET, start_date, end_date)


def invalidate() -> None:
    """Delete all cached parquet files; next call will rebuild from MySQL."""
    for p in (PRICES_PARQUET, FEATURES_PARQUET, LABELS_PARQUET):
        if p.exists():
            p.unlink()
            print(f"  [data_store] invalidated: {p.name}", flush=True)


def warm_up(db_session: Session) -> None:
    """Force-build all caches (useful after pipeline update to pre-warm before backtest)."""
    invalidate()
    _ensure(db_session)
    print("  [data_store] warm-up complete.", flush=True)


def cache_info() -> dict:
    """Return dict with each parquet's path, size (MB), max_date, and age (minutes)."""
    info = {}
    now = time.time()
    for label, path in [("prices", PRICES_PARQUET), ("features", FEATURES_PARQUET), ("labels", LABELS_PARQUET)]:
        if path.exists():
            stat = path.stat()
            info[label] = {
                "path": str(path),
                "size_mb": round(stat.st_size / 1e6, 1),
                "age_min": round((now - stat.st_mtime) / 60, 1),
                "max_date": _parquet_max_date(path),
                "fresh": _is_fresh(path),
            }
        else:
            info[label] = {"path": str(path), "exists": False}
    return info
