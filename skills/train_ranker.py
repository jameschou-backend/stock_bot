from __future__ import annotations

from datetime import timedelta
import hashlib
from pathlib import Path
from typing import Dict

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

# ── 嘗試載入 LightGBM（優先），否則回退 sklearn ──
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
from sklearn.ensemble import GradientBoostingRegressor


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "models"


def _parse_features(series: pd.Series) -> pd.DataFrame:
    def ensure_dict(value):
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        import json
        return json.loads(value)

    parsed = [ensure_dict(v) for v in series]
    return pd.json_normalize(parsed)


def _build_model(train_X, train_y, val_X, val_y):
    """建立並訓練模型。優先使用 LightGBM，否則回退 sklearn GBR。"""
    if _HAS_LGBM:
        model = lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=50,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        # 使用 early stopping 避免 overfit
        model.fit(
            train_X, train_y,
            eval_set=[(val_X, val_y)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        engine = "lightgbm"
    else:
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )
        model.fit(train_X, train_y)
        engine = "sklearn_gbr"
    return model, engine


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "train_ranker")
    try:
        max_label_date = db_session.query(func.max(Label.trading_date)).scalar()
        if max_label_date is None:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        train_end = max_label_date
        train_start = train_end - timedelta(days=365 * config.train_lookback_years)
        val_start = (pd.Timestamp(train_end) - pd.DateOffset(months=6)).date()

        feature_stmt = (
            select(Feature.stock_id, Feature.trading_date, Feature.features_json)
            .where(Feature.trading_date.between(train_start, train_end))
            .order_by(Feature.stock_id, Feature.trading_date)
        )
        label_stmt = (
            select(Label.stock_id, Label.trading_date, Label.future_ret_h)
            .where(Label.trading_date.between(train_start, train_end))
            .order_by(Label.stock_id, Label.trading_date)
        )
        feature_df = pd.read_sql(feature_stmt, db_session.get_bind())
        label_df = pd.read_sql(label_stmt, db_session.get_bind())
        if feature_df.empty or label_df.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        df = feature_df.merge(label_df, on=["stock_id", "trading_date"], how="inner")
        if df.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}
        df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date

        feature_matrix = _parse_features(df["features_json"])
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)

        # 允許部分 NaN（用中位數填補），而非全部 dropna
        for col in feature_matrix.columns:
            if feature_matrix[col].isna().all():
                feature_matrix[col] = 0
            else:
                feature_matrix[col] = feature_matrix[col].fillna(feature_matrix[col].median())

        # 移除仍有 NaN 的行（理論上不該剩餘）
        valid_mask = feature_matrix.notna().all(axis=1)
        feature_matrix = feature_matrix.loc[valid_mask]
        df = df.loc[feature_matrix.index].copy()
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
        train_mask = df["trading_date"] < val_start
        train_X = feature_matrix.loc[train_mask].values
        train_y = df.loc[train_mask, "future_ret_h"].astype(float).values
        val_X = feature_matrix.loc[~train_mask].values
        val_y = df.loc[~train_mask, "future_ret_h"].astype(float).values

        if train_X.shape[0] == 0:
            raise ValueError("Not enough training data to fit model")

        model, engine = _build_model(train_X, train_y, val_X, val_y)

        # ── 驗證集評估 ──
        preds = model.predict(val_X) if len(val_X) else np.array([])
        ic = spearmanr(preds, val_y).correlation if len(preds) else None

        topn = config.topn
        topn_mean = None
        if len(preds):
            val_frame = pd.DataFrame({"trading_date": df.loc[~train_mask, "trading_date"].values, "pred": preds, "y": val_y})

            def topn_mean_for_day(group: pd.DataFrame) -> float:
                return group.sort_values("pred", ascending=False).head(topn)["y"].mean()

            per_day = val_frame.groupby("trading_date").apply(topn_mean_for_day)
            topn_mean = float(per_day.mean()) if not per_day.empty else None

        # ── 特徵重要度（LightGBM 原生支援）──
        importance = {}
        if _HAS_LGBM and hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            for i, name in enumerate(feature_names):
                importance[name] = int(imp[i])

        metrics = {
            "ic_spearman": None if ic is None else float(ic),
            "topn_mean_future_ret": topn_mean,
            "train_rows": int(train_X.shape[0]),
            "val_rows": int(val_X.shape[0]),
            "engine": engine,
            "feature_count": len(feature_names),
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
