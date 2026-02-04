from __future__ import annotations

from typing import Dict, List

try:
    import joblib
except ModuleNotFoundError as exc:
    raise RuntimeError("Missing dependency 'joblib'. Install with `pip install -r requirements.txt`.") from exc
import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import Feature, ModelVersion, Pick
from skills.build_features import FEATURE_COLUMNS


REASON_FEATURES = FEATURE_COLUMNS[:8]


def _load_latest_model(session: Session) -> ModelVersion | None:
    return (
        session.query(ModelVersion)
        .order_by(ModelVersion.created_at.desc())
        .limit(1)
        .one_or_none()
    )


def _parse_features(series: pd.Series) -> pd.DataFrame:
    import json

    parsed = [json.loads(v) if isinstance(v, str) else v for v in series]
    return pd.json_normalize(parsed)


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "daily_pick")
    try:
        latest_date = db_session.query(func.max(Feature.trading_date)).scalar()
        if latest_date is None:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        model_version = _load_latest_model(db_session)
        if model_version is None:
            raise ValueError("No trained model found")

        artifact = joblib.load(model_version.artifact_path)
        model = artifact["model"]
        feature_names = artifact["feature_names"]

        stmt = (
            select(Feature.stock_id, Feature.trading_date, Feature.features_json)
            .where(Feature.trading_date == latest_date)
            .order_by(Feature.stock_id)
        )
        df = pd.read_sql(stmt, db_session.get_bind())
        if df.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        feature_df = _parse_features(df["features_json"])
        for col in feature_names:
            if col not in feature_df.columns:
                feature_df[col] = np.nan
        feature_df = feature_df[feature_names]
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        valid_mask = ~feature_df.isna().any(axis=1)
        if not valid_mask.any():
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        df = df.loc[valid_mask].reset_index(drop=True)
        feature_df = feature_df.loc[valid_mask].reset_index(drop=True)

        scores = model.predict(feature_df.values)
        df["score"] = scores

        percentile_map = {}
        for feat in REASON_FEATURES:
            if feat not in feature_df.columns:
                continue
            vals = pd.to_numeric(feature_df[feat], errors="coerce")
            percentile_map[feat] = vals.rank(pct=True)

        df = df.sort_values("score", ascending=False).head(config.topn)
        records: List[Dict] = []
        for _, row in df.iterrows():
            reasons = {}
            for feat in REASON_FEATURES:
                if feat not in feature_df.columns:
                    continue
                value = float(feature_df.loc[row.name, feat])
                pct = float(percentile_map[feat].loc[row.name]) if feat in percentile_map else None
                reasons[feat] = {"value": value, "percentile": pct}

            records.append(
                {
                    "pick_date": latest_date,
                    "stock_id": row["stock_id"],
                    "score": float(row["score"]),
                    "model_id": model_version.model_id,
                    "reason_json": reasons,
                }
            )

        if records:
            stmt = insert(Pick).values(records)
            stmt = stmt.on_duplicate_key_update(
                score=stmt.inserted.score,
                model_id=stmt.inserted.model_id,
                reason_json=stmt.inserted.reason_json,
            )
            db_session.execute(stmt)

        logs = {"rows": len(records), "pick_date": latest_date.isoformat(), "model_id": model_version.model_id}
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs={"error": str(exc)})
        raise
