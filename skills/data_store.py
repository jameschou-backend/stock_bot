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
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import List, Optional

import duckdb
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

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

# ── 快取有效性 ────────────────────────────────────────────────────────────────
# 注意：cache 失效判斷只用「資料日期比對」（_ensure 內部比對 max_date vs source max_date）
# 不再用 mtime+TTL，因為那會造成 pipeline 跑完但 cache 未過期時 daily-c 用舊資料。
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
            logger.info("[data_store] schema filter: dropped %s stale rows", f"{n_drop:,}")
        feat_df = feat_df.loc[mask].reset_index(drop=True)
        feat_df[num_cols] = feat_df[num_cols].astype("float32")

    return feat_df


# ── 快取建立 ──────────────────────────────────────────────────────────────────
def _build_prices(db_session: Session) -> None:
    logger.info("[data_store] building prices.parquet (full history) …")
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
    logger.info(
        "[data_store] prices saved: %s rows (%.1fs) -> %s",
        f"{len(df):,}", time.time() - t0, PRICES_PARQUET.name,
    )


def _build_features(db_session: Session) -> None:
    """Build the flat features cache.

    優先從 FeatureStore（年份 Parquet）讀取已解析資料，
    大幅省去 JSON 解析開銷（4M 行 × 62 特徵 ≈ 節省 60-90s）。
    若 FeatureStore 無資料（首次使用前），fallback 至 MySQL。
    """
    logger.info("[data_store] building features.parquet …")
    t0 = time.time()

    # ── 嘗試從 FeatureStore（年份 Parquet）直接讀取 ──
    try:
        from skills.feature_store import FeatureStore
        from datetime import date as _date
        _fs = FeatureStore()
        _max_date = _fs.get_max_date()
        if _max_date is not None:
            logger.info(
                "[data_store] FeatureStore found (max=%s), reading from year-partitioned Parquet …",
                _max_date,
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
            logger.info(
                "[data_store] features saved (from FeatureStore): %s rows x %d features (%.1fs total) -> %s",
                f"{len(feat_df):,}", len(num_cols), time.time() - t0, FEATURES_PARQUET.name,
            )
            return
    except Exception as _exc:
        logger.warning(
            "[data_store] FeatureStore read failed (%s), falling back to MySQL …",
            _exc,
        )

    # ── Fallback：MySQL（FeatureStore 尚未初始化時使用）──
    logger.info("[data_store] reading features from MySQL (fallback) …")
    stmt = (
        select(Feature.stock_id, Feature.trading_date, Feature.features_json)
        .order_by(Feature.trading_date, Feature.stock_id)
    )
    raw = pd.read_sql(stmt, db_session.get_bind())
    raw["trading_date"] = pd.to_datetime(raw["trading_date"]).dt.date
    logger.info("[data_store] db_load done (%.1fs), parsing …", time.time() - t0)

    t1 = time.time()
    feat_df = _parse_and_schema_filter(raw)
    del raw
    gc.collect()
    logger.info("[data_store] parse done (%.1fs), saving …", time.time() - t1)

    feat_df.to_parquet(FEATURES_PARQUET, index=False)
    logger.info(
        "[data_store] features saved (from MySQL): %s rows x %d features (%.1fs total) -> %s",
        f"{len(feat_df):,}", len(feat_df.columns) - 2, time.time() - t0, FEATURES_PARQUET.name,
    )


def _build_labels(db_session: Session) -> None:
    logger.info("[data_store] building labels.parquet (full history) …")
    t0 = time.time()
    stmt = (
        select(Label.stock_id, Label.trading_date, Label.future_ret_h)
        .order_by(Label.trading_date, Label.stock_id)
    )
    df = pd.read_sql(stmt, db_session.get_bind())
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df.to_parquet(LABELS_PARQUET, index=False)
    logger.info(
        "[data_store] labels saved: %s rows (%.1fs) -> %s",
        f"{len(df):,}", time.time() - t0, LABELS_PARQUET.name,
    )


def _parquet_row_count(path: Path) -> Optional[int]:
    """讀 parquet 列數（DuckDB metadata，不掃資料）。"""
    if not path.exists():
        return None
    try:
        con = duckdb.connect()
        n = con.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
        con.close()
        return int(n) if n is not None else None
    except Exception:
        return None


# process 級 memoization：同一個 process 內每種 cache 只檢查/重建一次。
# 背景（2026-07-03 健檢 P1-4）：rolling 回測逐 fold 呼叫 get_*，若 pipeline
# 併發更新 DB，同一個 run 的前後 fold 可能讀到兩個不同快照（比 2026-06-22
# 記錄的「兩臂不同快照」更糟）。首次檢查後鎖定，run 內保證單一快照。
_ENSURED_KINDS: set = set()


def reset_ensure_memo() -> None:
    """清除 process 級 staleness memoization（測試/長駐 process 換日用）。"""
    _ENSURED_KINDS.clear()


def snapshot_info() -> dict:
    """回傳目前三個 cache 的快照身分（max_date + 列數）。

    實驗記錄用：把此 dict 寫進回測結果 JSON，跨 run 比對即可確認兩臂
    是否使用同一份資料快照（2026-06-22 可重現性危害的防護之一）。
    """
    info: dict = {}
    for kind, path in (
        ("prices", PRICES_PARQUET),
        ("features", FEATURES_PARQUET),
        ("labels", LABELS_PARQUET),
    ):
        info[kind] = {
            "max_date": _parquet_max_date(path),
            "rows": _parquet_row_count(path),
        }
    return info


def _ensure(db_session: Session) -> None:
    """確保三個 parquet 存在且資料日期與來源一致，否則重建。

    比對邏輯（stale 任一成立即重建）：
      - cache 不存在
      - cache max_date < 來源 max_date（有新交易日）
      - cache 列數 != 來源列數（**歷史內容重建**：force_recompute、下市股回補、
        adj factor 重灌後全量重建等，max_date 不變但內容已換——只比 max_date
        會默默供應舊快照，2026-07-03 健檢 P1-4）
    prices / labels 來源 = MySQL 全表；features 來源 = FeatureStore（年份 Parquet）。
    同一 process 內每種 cache 只檢查一次（_ENSURED_KINDS），保證 run 內單一快照。
    """
    from sqlalchemy import text as _text
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 凍結模式（實驗可重現性）──
    # 設 DATA_STORE_FREEZE=1 時，只用既有 cache、絕不重建，確保對照實驗期間
    # （即使 daily pipeline / cron 併發更新 DB）cache 不變、結果可重現。
    # 背景：2026-06-22 發現 rep1 因橫跨 cache 重建讀到舊快照，與其餘 run 差 0.23 Sharpe。
    # 不做 silent fallback：cache 缺檔時明確報錯，要求先建 cache 或解除凍結。
    if os.getenv("DATA_STORE_FREEZE", "").strip().lower() in ("1", "true", "yes", "on"):
        _missing = [str(p) for p in (PRICES_PARQUET, FEATURES_PARQUET, LABELS_PARQUET) if not p.exists()]
        if _missing:
            raise RuntimeError(
                f"DATA_STORE_FREEZE 啟用但 cache 不存在: {_missing}；"
                "請先在未凍結狀態跑一次建立 cache，或解除 DATA_STORE_FREEZE。"
            )
        logger.info("[data_store] DATA_STORE_FREEZE 啟用：使用既有 cache，跳過 staleness 檢查與重建")
        return

    def _is_stale(kind: str, path: Path, src_max: Optional[str], src_rows: Optional[int]) -> bool:
        cache_max = _parquet_max_date(path)
        if not cache_max:
            return True
        if src_max and cache_max < src_max:
            logger.info("[data_store] %s cache stale (max %s < %s), rebuilding …", kind, cache_max, src_max)
            return True
        if src_rows is not None:
            cache_rows = _parquet_row_count(path)
            if cache_rows is not None and cache_rows != src_rows:
                logger.info(
                    "[data_store] %s cache stale (rows %s != source %s — 歷史內容已重建), rebuilding …",
                    kind, f"{cache_rows:,}", f"{src_rows:,}",
                )
                return True
        return False

    # ── Prices ──
    if "prices" not in _ENSURED_KINDS:
        try:
            _src_px = str(db_session.execute(_text("SELECT max(trading_date) FROM raw_prices")).scalar() or "")
            _src_px_rows = int(db_session.execute(_text("SELECT count(*) FROM raw_prices")).scalar() or 0)
        except Exception:
            _src_px, _src_px_rows = None, None
        if _is_stale("prices", PRICES_PARQUET, _src_px, _src_px_rows):
            _build_prices(db_session)
        _ENSURED_KINDS.add("prices")

    # ── Features ──
    if "features" not in _ENSURED_KINDS:
        try:
            from skills.feature_store import FeatureStore as _FS
            _fs = _FS()
            _src_feat_raw = _fs.get_max_date()
            _src_feat = str(_src_feat_raw) if _src_feat_raw is not None else None
            _src_feat_rows = _fs.row_count()
        except Exception:
            _src_feat, _src_feat_rows = None, None
        if _is_stale("features", FEATURES_PARQUET, _src_feat, _src_feat_rows):
            _build_features(db_session)
        _ENSURED_KINDS.add("features")

    # ── Labels ──
    if "labels" not in _ENSURED_KINDS:
        try:
            _src_lbl = str(db_session.execute(_text("SELECT max(trading_date) FROM labels")).scalar() or "")
            _src_lbl_rows = int(db_session.execute(_text("SELECT count(*) FROM labels")).scalar() or 0)
        except Exception:
            _src_lbl, _src_lbl_rows = None, None
        if _is_stale("labels", LABELS_PARQUET, _src_lbl, _src_lbl_rows):
            _build_labels(db_session)
        _ENSURED_KINDS.add("labels")


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
            logger.info("[data_store] invalidated: %s", p.name)


def warm_up(db_session: Session) -> None:
    """Force-build all caches (useful after pipeline update to pre-warm before backtest)."""
    invalidate()
    _ensure(db_session)
    logger.info("[data_store] warm-up complete.")


def cache_info(db_session: Optional[Session] = None) -> dict:
    """Return dict with each parquet's path, size (MB), max_date, and sync status.

    Args:
        db_session: 若提供，會比對 DB 來源 max_date 並回報 `synced` 旗標
                    （True = cache 與來源同步、安全使用；False = cache 落後、下次 _ensure 會重建）。
                    不提供時 synced 欄位不存在（純 cache 內容報告）。
    """
    from sqlalchemy import text as _text

    info = {}
    now = time.time()
    sources = {}
    if db_session is not None:
        try:
            sources["prices"] = str(db_session.execute(_text("SELECT max(trading_date) FROM raw_prices")).scalar() or "")
        except Exception:
            sources["prices"] = None
        try:
            from skills.feature_store import FeatureStore as _FS
            mx = _FS().get_max_date()
            sources["features"] = str(mx) if mx is not None else None
        except Exception:
            sources["features"] = None
        try:
            sources["labels"] = str(db_session.execute(_text("SELECT max(trading_date) FROM labels")).scalar() or "")
        except Exception:
            sources["labels"] = None

    for label, path in [("prices", PRICES_PARQUET), ("features", FEATURES_PARQUET), ("labels", LABELS_PARQUET)]:
        if path.exists():
            stat = path.stat()
            cache_max = _parquet_max_date(path)
            entry = {
                "path": str(path),
                "size_mb": round(stat.st_size / 1e6, 1),
                "age_min": round((now - stat.st_mtime) / 60, 1),
                "max_date": cache_max,
            }
            if db_session is not None:
                src = sources.get(label)
                entry["source_max_date"] = src
                entry["synced"] = bool(cache_max and src and cache_max >= src)
            info[label] = entry
        else:
            info[label] = {"path": str(path), "exists": False}
    return info
