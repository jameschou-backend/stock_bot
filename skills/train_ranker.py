"""模型訓練模組：讀取 features/labels 表，使用 LightGBM（優先）或 sklearn GBR 訓練排名模型，
評估 IC/TopK 命中率，並將模型 artifact 存至磁碟、版本資訊寫入 model_versions 表。
"""
from __future__ import annotations

import logging
from datetime import timedelta
import hashlib
from pathlib import Path
from typing import Dict, Iterable

logger = logging.getLogger(__name__)

try:
    import joblib
except ModuleNotFoundError as exc:
    raise RuntimeError("Missing dependency 'joblib'. Install with `pip install -r requirements.txt`.") from exc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import Feature, Label, ModelVersion
from skills.build_features import (
    FEATURE_COLUMNS as _CANONICAL_FEATURES,
    PRUNED_FEATURE_COLS as _PRUNED_FEATURES,
)
from skills.feature_utils import (
    parse_features_json as _parse_features_json_shared,
    filter_schema_valid_rows as _filter_schema_valid_rows,
)
from skills.model_params import RANKER_PROD_PARAMS

# ── 嘗試載入 LightGBM（優先），否則回退 sklearn ──
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
from sklearn.ensemble import GradientBoostingRegressor


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "models"

# 交易日制 label buffer：與 backtest.py 的 label_horizon_buffer=20 對齊（20 交易日 horizon）
LABEL_HORIZON_BUFFER_DAYS = 20


def _parse_features(series: pd.Series) -> pd.DataFrame:
    """委派給 feature_utils.parse_features_json（統一實作）。"""
    return _parse_features_json_shared(series)


def resolve_train_end(all_trading_dates, max_label_date, buffer_days: int = LABEL_HORIZON_BUFFER_DAYS):
    """交易日制 label buffer 截止日（純函數，抽出供行為鎖定測試）。

    - 交易日足夠（len > buffer_days）：取倒數第 (buffer_days+1) 個「交易日」，
      確保訓練集最後一筆 label（close[T + 20 交易日]）不超出資料末端。
    - 交易日不足：保守 fallback 至 max_label_date - buffer_days「日曆天」
      （日曆天 < 交易日跨度，train_end 只會更早、不會更晚，方向安全）。

    Args:
        all_trading_dates: 遞增排序的 distinct label 交易日 list。
        max_label_date: labels 表最大交易日。
    """
    if len(all_trading_dates) > buffer_days:
        return all_trading_dates[-(buffer_days + 1)]
    return max_label_date - timedelta(days=buffer_days)


def _resolve_feature_columns(config) -> list:
    """P1-5(a) 回測=部署對齊：解析訓練特徵集。

    預設 PRUNED_FEATURE_COLS（58 特徵，含 enrich-only 欄 close_fracdiff_0_50），
    與生產回測 CLI `--pruned-features` 一致；config.train_use_pruned_features=False
    時退回全 FEATURE_COLUMNS（87，診斷/實驗用）。
    """
    if bool(getattr(config, "train_use_pruned_features", True)):
        return list(_PRUNED_FEATURES)
    return list(_CANONICAL_FEATURES)


def _liquidity_sample_weight(amt20: pd.Series) -> np.ndarray:
    """P1-5(b)：流動性加權 sample_weight ∝ log(1+amt_20)，平均歸一至 1.0。

    與 skills/backtest.py `_train_model_for_period` 的 --liq-weighted 實作一致
    （fillna(0).clip(lower=0) → log1p → 除以平均），讓模型學偏大型股模式。
    """
    _vals = pd.to_numeric(amt20, errors="coerce").fillna(0).clip(lower=0).values.astype(float)
    _w = np.log1p(_vals)
    _mean = _w.mean() if len(_w) else 0.0
    if _mean > 0:
        _w = _w / _mean
    return _w


def _build_model(train_X, train_y, sample_weight=None):
    """建立並訓練模型。優先使用 LightGBM，否則回退 sklearn GBR。

    P1-5 回測=部署對齊：LGBM 參數引用 skills.model_params.RANKER_PROD_PARAMS，
    與 backtest._train_model（regression 路徑）同一組：500 樹、無 early stopping。
    舊版 800 樹 + early_stopping(50) 造成「部署模型 ≠ 回測驗證過的模型」，已向回測收斂。
    無 early stopping → 訓練不再需要 val set（val 段僅用於事後評估指標）。
    """
    if _HAS_LGBM:
        model = lgb.LGBMRegressor(**RANKER_PROD_PARAMS)
        model.fit(train_X, train_y, sample_weight=sample_weight)
        engine = "lightgbm"
    else:
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )
        model.fit(train_X, train_y, sample_weight=sample_weight)
        engine = "sklearn_gbr"
    return model, engine


def _safe_float(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def _compute_validation_metrics(
    val_dates: Iterable,
    val_y: np.ndarray,
    preds: np.ndarray,
    topk_list: Iterable[int],
) -> Dict[str, object]:
    ic = None
    if len(preds):
        ic_value = spearmanr(preds, val_y).correlation
        ic = _safe_float(ic_value)

    topk_metrics: Dict[str, float | None] = {}
    hitrate_metrics: Dict[str, float | None] = {}

    if len(preds):
        val_frame = pd.DataFrame(
            {
                "trading_date": list(val_dates),
                "pred": preds,
                "y": val_y,
            }
        )
        for k in topk_list:
            key = f"k{int(k)}"
            topk_rows = (
                val_frame.sort_values(["trading_date", "pred"], ascending=[True, False])
                .groupby("trading_date")
                .head(k)
            )
            per_day_topk = topk_rows.groupby("trading_date")["y"].mean()
            per_day_hitrate = topk_rows.groupby("trading_date")["y"].apply(lambda s: float((s > 0).mean()))
            topk_metrics[key] = _safe_float(per_day_topk.mean()) if not per_day_topk.empty else None
            hitrate_metrics[key] = _safe_float(per_day_hitrate.mean()) if not per_day_hitrate.empty else None
    else:
        for k in topk_list:
            key = f"k{int(k)}"
            topk_metrics[key] = None
            hitrate_metrics[key] = None

    pred_stats = {
        "mean": _safe_float(np.mean(preds)) if len(preds) else None,
        "std": _safe_float(np.std(preds)) if len(preds) else None,
        "min": _safe_float(np.min(preds)) if len(preds) else None,
        "max": _safe_float(np.max(preds)) if len(preds) else None,
    }

    return {
        "v": 1,
        "ic_spearman": ic,
        "topk": topk_metrics,
        "hitrate": hitrate_metrics,
        "pred_stats": pred_stats,
    }


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "train_ranker")
    try:
        max_label_date = db_session.query(func.max(Label.trading_date)).scalar()
        if max_label_date is None:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        # 交易日制 buffer：取最後一個 label 日往前第 20 個「交易日」作為訓練截止，確保訓練集
        # 最後一筆 label（close[T + 20 交易日]）不超出資料末端。先前用日曆天蓋不住 20 交易日
        # （≈28-29 日曆天）的 horizon → 殘餘前向洩漏（與 backtest.py 同步修正）。
        # 註：值仍 20，但語義從「日曆天」改為「交易日」（邏輯抽至 resolve_train_end 純函數）。
        _td_rows = (
            db_session.query(Label.trading_date).distinct().order_by(Label.trading_date).all()
        )
        _all_tds = [r[0] for r in _td_rows]
        train_end = resolve_train_end(_all_tds, max_label_date, LABEL_HORIZON_BUFFER_DAYS)
        train_start = train_end - timedelta(days=365 * config.train_lookback_years)
        val_start = (pd.Timestamp(train_end) - pd.DateOffset(months=6)).date()

        # ── 讀取特徵：優先 Parquet FeatureStore，fallback MySQL ──
        feature_df: pd.DataFrame
        _used_parquet = False
        try:
            from skills.feature_store import FeatureStore as _FeatureStore
            _fs = _FeatureStore()
            if _fs.get_max_date() is not None:
                feature_df = _fs.read(train_start, train_end)
                feature_df["trading_date"] = pd.to_datetime(
                    feature_df["trading_date"]
                ).dt.date
                _used_parquet = True
        except Exception as _exc:
            logger.warning(
                "[train_ranker] FeatureStore read failed (%s), falling back to MySQL …",
                _exc,
            )

        if not _used_parquet:
            feature_stmt = (
                select(Feature.stock_id, Feature.trading_date, Feature.features_json)
                .where(Feature.trading_date.between(train_start, train_end))
                .order_by(Feature.stock_id, Feature.trading_date)
            )
            feature_df = pd.read_sql(feature_stmt, db_session.get_bind())

        label_stmt = (
            select(Label.stock_id, Label.trading_date, Label.future_ret_h)
            .where(Label.trading_date.between(train_start, train_end))
            .order_by(Label.stock_id, Label.trading_date)
        )
        label_df = pd.read_sql(label_stmt, db_session.get_bind())
        if feature_df.empty or label_df.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        df = feature_df.merge(label_df, on=["stock_id", "trading_date"], how="inner")
        if df.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}
        df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date

        # ── 流動性過濾（與 backtest.py 評分/訓練階段保持一致）──
        min_avg_turnover = float(kwargs.get("min_avg_turnover", 0.0))
        if min_avg_turnover > 0:
            from app.models import RawPrice as _RawPrice
            _price_stmt = (
                select(_RawPrice.stock_id, _RawPrice.trading_date, _RawPrice.close, _RawPrice.volume)
                .where(_RawPrice.trading_date.between(train_start, train_end))
                .order_by(_RawPrice.stock_id, _RawPrice.trading_date)
            )
            _price_df = pd.read_sql(_price_stmt, db_session.get_bind())
            if not _price_df.empty:
                _price_df["close"] = pd.to_numeric(_price_df["close"], errors="coerce")
                _price_df["volume"] = pd.to_numeric(_price_df["volume"], errors="coerce")
                _price_df = _price_df.dropna(subset=["close", "volume"])
                _price_df["stock_id"] = _price_df["stock_id"].astype(str)
                _price_df = _price_df.sort_values(["stock_id", "trading_date"])
                # `min_avg_turnover` 此處為「億元」單位（向後相容語意），× 1e8 轉成元
                # 新 code 請用 skills.risk.resolve_liquidity_threshold_twd(config)
                _threshold = min_avg_turnover * 1e8
                _price_df["_tv"] = _price_df["close"] * _price_df["volume"]
                _price_df["_avg_tv20"] = _price_df.groupby("stock_id")["_tv"].transform(
                    lambda s: s.rolling(20, min_periods=1).mean()
                )
                _price_df["trading_date"] = pd.to_datetime(_price_df["trading_date"]).dt.date
                _eligible = _price_df[_price_df["_avg_tv20"] >= _threshold]
                _before = len(df)
                # 向量化成員判定（2026-07-03 健檢效能審計發現 9）：
                # 等價於原 `[(td, str(sid)) in eligible_set for ...]` Python 迴圈——
                # 右表 (stock_id, _td64) 唯一，left merge 保序保列數，比對同一組 pair。
                _pairs = _eligible[["stock_id", "trading_date"]].drop_duplicates().copy()
                _pairs["stock_id"] = _pairs["stock_id"].astype(str)
                _pairs["_td64"] = pd.to_datetime(_pairs["trading_date"])
                _pairs = _pairs[["stock_id", "_td64"]]
                _pairs["_liq_ok"] = True
                _probe = pd.DataFrame({
                    "stock_id": df["stock_id"].astype(str).values,
                    "_td64": pd.to_datetime(df["trading_date"]).values,
                })
                _liq_ok = _probe.merge(_pairs, on=["stock_id", "_td64"], how="left")["_liq_ok"].notna().to_numpy()
                df = df[_liq_ok].copy()
                df = df.reset_index(drop=True)
                logger.info(
                    "[train_ranker] 流動性過濾 (>=%.0f億): %s -> %s 筆",
                    min_avg_turnover, f"{_before:,}", f"{len(df):,}",
                )
                del _price_df, _eligible, _pairs, _probe

        # ── 特徵矩陣：Parquet 路徑已預解析，MySQL 路徑需 JSON parse ──
        if _used_parquet:
            # Parquet 已預先 flatten，直接取出特徵欄（排除 meta 欄和 label 欄）
            _non_feat = {"stock_id", "trading_date", "future_ret_h", "features_json"}
            _feat_cols_available = [c for c in df.columns if c not in _non_feat]
            feature_matrix = df[_feat_cols_available].copy().reset_index(drop=True)
        else:
            feature_matrix = _parse_features(df["features_json"])
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)

        # ── P1-5(a)：限縮到訓練特徵集，確保訓練/預測特徵集一致 ──
        # 預設 PRUNED_FEATURE_COLS（58，與生產回測 --pruned-features 一致）；
        # 補缺失欄（新特徵在舊歷史資料 / MySQL fallback 中不存在，填 NaN → 後續用 train 段中位數填補）
        _train_feature_cols = _resolve_feature_columns(config)
        for col in _train_feature_cols:
            if col not in feature_matrix.columns:
                feature_matrix[col] = np.nan
        feature_matrix = feature_matrix[list(_train_feature_cols)]

        # ── Schema 遷移保護：過濾掉 canonical feature 覆蓋率不足的舊版資料 ──
        # 根因：730d 重算後 DB 中舊 schema（19 features）資料對 48 canonical features
        # 只有約 17% 覆蓋率，median imputation 會嚴重污染訓練。
        # 門檻 50%：舊行 8/48≈17% < 50% → 過濾；新行 48/48=100% > 50% → 保留。
        # 委派給 feature_utils.filter_schema_valid_rows（統一實作）。
        feature_matrix, _n_schema_dropped = _filter_schema_valid_rows(feature_matrix, coverage_threshold=0.50)
        if _n_schema_dropped > 0:
            logger.info(
                "[train_ranker] Schema filter: 過濾 %s 筆舊版特徵資料 (coverage < 50%%, canonical_features=%d)",
                f"{_n_schema_dropped:,}", len(_CANONICAL_FEATURES),
            )
            df = df.loc[feature_matrix.index].copy()
        if feature_matrix.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0, "reason": "no valid schema rows"})
            return {"rows": 0}

        # ── P1-5(b)：先擷取原始 amt_20（imputation 前），供流動性加權使用 ──
        # 與 backtest 一致：權重用 raw amt_20（fillna(0)），不用 imputed 值。
        _amt20_raw = feature_matrix["amt_20"].copy() if "amt_20" in feature_matrix.columns else None

        # ── P2-4 修正：imputation 中位數只用 train 段計算 ──
        # 舊版在 train/val 切分「前」以全體資料算 median → val 段分佈洩入填補值。
        # 現在先決定時間切分點，再以 train 段 median 填補 train+val 兩段；
        # train 段全 NaN 的欄（新特徵尚無歷史）填 0（保留舊版 all-NaN→0 行為）。
        _impute_train_mask = df["trading_date"] < val_start
        for col in feature_matrix.columns:
            _med = feature_matrix.loc[_impute_train_mask, col].median()
            if pd.isna(_med):
                _med = 0
            feature_matrix[col] = feature_matrix[col].fillna(_med)

        # 移除仍有 NaN 的行（理論上不該剩餘）
        valid_mask = feature_matrix.notna().all(axis=1)
        feature_matrix = feature_matrix.loc[valid_mask]
        df = df.loc[feature_matrix.index].copy()
        if _amt20_raw is not None:
            _amt20_raw = _amt20_raw.loc[feature_matrix.index]
        if feature_matrix.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        feature_names = list(feature_matrix.columns)
        feature_set_hash = hashlib.sha256(",".join(sorted(feature_names)).encode("utf-8")).hexdigest()[:16]
        engine_tag = "lgbm" if _HAS_LGBM else "gbr"
        model_id = f"ranker_{engine_tag}_{train_start:%Y%m%d}_{train_end:%Y%m%d}_{feature_set_hash}"

        existing = db_session.query(ModelVersion).filter(ModelVersion.model_id == model_id).first()
        if existing:
            finish_job(db_session, job_id, "success", logs={"rows": len(df), "model_id": model_id})
            return {"rows": len(df), "model_id": model_id, "existing": True}

        # ── 時間序列切分：嚴格以時間劃分 train / val ──
        # （P1-5：無 early stopping，val 段只用於下方評估指標，不參與訓練）
        train_mask = df["trading_date"] < val_start
        train_X = feature_matrix.loc[train_mask].values
        train_y = df.loc[train_mask, "future_ret_h"].astype(float).values
        val_X = feature_matrix.loc[~train_mask].values
        val_y = df.loc[~train_mask, "future_ret_h"].astype(float).values

        if train_X.shape[0] == 0:
            raise ValueError("Not enough training data to fit model")

        # ── P1-5(b)：流動性加權 sample_weight（train 段），與生產回測 --liq-weighted 一致 ──
        train_sample_weight = None
        if bool(getattr(config, "train_liq_weighting", True)) and _amt20_raw is not None:
            train_sample_weight = _liquidity_sample_weight(_amt20_raw.loc[train_mask])

        model, engine = _build_model(train_X, train_y, sample_weight=train_sample_weight)

        # ── 驗證集評估 ──
        preds = model.predict(val_X) if len(val_X) else np.array([])
        eval_topk_list = getattr(config, "eval_topk_list", (10, 20))
        metrics_core = _compute_validation_metrics(
            val_dates=df.loc[~train_mask, "trading_date"].values,
            val_y=val_y,
            preds=preds,
            topk_list=eval_topk_list,
        )

        # ── 兩段式訓練：holdout 模型只做上方評估指標，正式部署模型用全部資料重訓 ──
        # 回測每期訓練到 cutoff 為止（無 holdout）；若部署模型也扣掉最近 6 個月，
        # 會系統性少學最新 regime，與回測驗證的行為不一致（2026-07-03 對齊尾項）。
        full_sample_weight = None
        if bool(getattr(config, "train_liq_weighting", True)) and _amt20_raw is not None:
            full_sample_weight = _liquidity_sample_weight(_amt20_raw)
        model, engine = _build_model(
            feature_matrix.values,
            df["future_ret_h"].astype(float).values,
            sample_weight=full_sample_weight,
        )

        # ── 特徵重要度（LightGBM 原生支援）──
        importance = {}
        if _HAS_LGBM and hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            for i, name in enumerate(feature_names):
                importance[name] = int(imp[i])

        metrics = {
            **metrics_core,
            "train_rows": int(train_X.shape[0]),
            "val_rows": int(val_X.shape[0]),
            # 正式模型 = 全資料重訓（holdout 模型僅產出上方 val 指標）
            "final_train_rows": int(feature_matrix.shape[0]),
            "engine": engine,
            "feature_count": len(feature_names),
            # P1-5 對齊資訊（追溯部署模型與回測配置是否一致）
            "feature_set": "pruned" if bool(getattr(config, "train_use_pruned_features", True)) else "full",
            "liq_weighting": bool(train_sample_weight is not None),
            "feature_importance_top10": dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            ) if importance else None,
        }

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        artifact_path = ARTIFACT_DIR / f"{model_id}.pkl"
        joblib.dump({"model": model, "feature_names": feature_names}, artifact_path)

        model_version = ModelVersion(
            model_id=model_id,
            train_start=train_start,
            train_end=train_end,
            feature_set_hash=feature_set_hash,
            params_json=model.get_params() if hasattr(model, "get_params") else {},
            metrics_json=metrics,
            artifact_path=str(artifact_path),
        )
        db_session.add(model_version)

        logs = {"model_id": model_id, **metrics}
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs={"error": str(exc)})
        raise
