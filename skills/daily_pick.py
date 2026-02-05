from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Tuple

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
from app.models import Feature, ModelVersion, Pick, RawPrice, Stock
from skills.build_features import FEATURE_COLUMNS


REASON_FEATURES = FEATURE_COLUMNS[:8]


def _get_valid_stock_universe(session: Session) -> set:
    """取得有效股票 universe（排除 ETF、權證等）
    
    過濾條件：
    - security_type = 'stock' 
    - is_listed = True
    
    若 stocks 表為空，回傳空集合（不會阻擋流程，但會記錄 warning）
    """
    stmt = (
        select(Stock.stock_id)
        .where(Stock.security_type == "stock")
        .where(Stock.is_listed == True)
    )
    rows = session.execute(stmt).fetchall()
    return {row[0] for row in rows}


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


def _impute_features(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    feature_df = feature_df.copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    nan_mask = feature_df.isna()
    total_cells = int(nan_mask.size)
    filled_cells = int(nan_mask.sum().sum())
    all_nan_cols = [col for col in feature_df.columns if feature_df[col].isna().all()]

    medians = feature_df.median(skipna=True)
    for col in feature_df.columns:
        if col in all_nan_cols:
            feature_df[col] = feature_df[col].fillna(0)
        else:
            feature_df[col] = feature_df[col].fillna(medians[col])

    fill_ratio = filled_cells / total_cells if total_cells else 0.0
    stats = {
        "filled_cells": filled_cells,
        "total_cells": total_cells,
        "fill_ratio": round(fill_ratio, 6),
        "all_nan_cols": all_nan_cols,
    }
    return feature_df, stats


def _load_price_universe(
    db_session: Session,
    target_date,
    stock_ids: List[str],
    min_avg_volume: int,
) -> pd.DataFrame:
    if not stock_ids:
        return pd.DataFrame()

    start_date = target_date - timedelta(days=30)
    stmt = (
        select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close, RawPrice.volume)
        .where(RawPrice.trading_date.between(start_date, target_date))
        .where(RawPrice.stock_id.in_(stock_ids))
        .order_by(RawPrice.stock_id, RawPrice.trading_date)
    )
    df = pd.read_sql(stmt, db_session.get_bind())
    if df.empty:
        return df

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["close", "volume"])
    return df


def _choose_pick_date(
    candidate_dates: List[date],
    feature_df: pd.DataFrame,
    price_df: pd.DataFrame,
    topn: int,
    min_avg_volume: int,
    fallback_days: int,
) -> Tuple[date | None, pd.DataFrame, Dict[str, object]]:
    best_date = None
    best_df = pd.DataFrame()
    best_valid = 0
    fallback_used = 0

    for idx, target_date in enumerate(candidate_dates[: fallback_days + 1]):
        target = target_date
        date_features = feature_df[feature_df["trading_date"] == target].copy()
        if date_features.empty:
            continue

        date_features = date_features[date_features["stock_id"].str.fullmatch(r"\d{4}")]
        if date_features.empty:
            continue
        date_features = date_features.drop_duplicates(subset=["stock_id", "trading_date"])

        date_prices = price_df[price_df["trading_date"] == target].copy()
        if date_prices.empty:
            continue

        latest_price = date_prices.dropna(subset=["close", "volume"])[["stock_id", "close", "volume"]]
        if latest_price.empty:
            continue

        avg_volume = (
            price_df[price_df["trading_date"] <= target]
            .sort_values(["stock_id", "trading_date"])
            .groupby("stock_id")
            .tail(20)
            .groupby("stock_id")["volume"]
            .mean()
        )
        if min_avg_volume > 0:
            avg_volume = avg_volume[avg_volume >= min_avg_volume]

        eligible = latest_price.merge(avg_volume.rename("avg_volume"), on="stock_id", how="inner")
        eligible = eligible.drop_duplicates(subset=["stock_id"])
        if eligible.empty:
            continue

        date_features = date_features.merge(eligible[["stock_id"]], on="stock_id", how="inner")
        valid_count = len(date_features)
        if valid_count > 0 and best_date is None:
            best_date = target_date
            best_df = date_features
            best_valid = valid_count
            fallback_used = idx

        if valid_count >= topn:
            return target_date, date_features, {
                "fallback_days": idx,
                "valid_candidates": valid_count,
                "topn_returned": min(valid_count, topn),
            }

    if best_date is None:
        return None, pd.DataFrame(), {
            "fallback_days": None,
            "valid_candidates": 0,
            "topn_returned": 0,
        }

    return best_date, best_df, {
        "fallback_days": fallback_used,
        "valid_candidates": best_valid,
        "topn_returned": min(best_valid, topn),
    }


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "daily_pick")
    coverage_stats: Dict[str, object] = {}
    
    try:
        candidate_dates = (
            db_session.query(Feature.trading_date)
            .distinct()
            .order_by(Feature.trading_date.desc())
            .limit(config.fallback_days + 1)
            .all()
        )
        candidate_dates = [row[0] for row in candidate_dates]
        coverage_stats["candidate_dates_count"] = len(candidate_dates)
        
        if not candidate_dates:
            finish_job(db_session, job_id, "success", logs={
                "rows": 0, 
                "reason": "no feature dates",
                **coverage_stats,
            })
            return {"rows": 0}
        
        coverage_stats["latest_feature_date"] = candidate_dates[0].isoformat()
        coverage_stats["oldest_candidate_date"] = candidate_dates[-1].isoformat()

        model_version = _load_latest_model(db_session)
        if model_version is None:
            raise ValueError("No trained model found")

        artifact = joblib.load(model_version.artifact_path)
        model = artifact["model"]
        feature_names = artifact["feature_names"]
        
        # 取得有效股票 universe（排除 ETF、權證等）
        valid_stocks = _get_valid_stock_universe(db_session)
        coverage_stats["valid_stock_universe_count"] = len(valid_stocks)
        
        if not valid_stocks:
            # stocks 表為空時，不過濾（向後相容）
            coverage_stats["stock_universe_filter"] = "disabled (stocks table empty)"
            stmt = (
                select(Feature.stock_id, Feature.trading_date, Feature.features_json)
                .where(Feature.trading_date.in_(candidate_dates))
                .order_by(Feature.stock_id, Feature.trading_date)
            )
        else:
            coverage_stats["stock_universe_filter"] = "enabled"
            stmt = (
                select(Feature.stock_id, Feature.trading_date, Feature.features_json)
                .where(Feature.trading_date.in_(candidate_dates))
                .where(Feature.stock_id.in_(valid_stocks))
                .order_by(Feature.stock_id, Feature.trading_date)
            )
        
        df = pd.read_sql(stmt, db_session.get_bind())
        
        coverage_stats["total_feature_rows"] = len(df)
        coverage_stats["unique_stocks_with_features"] = df["stock_id"].nunique() if not df.empty else 0
        
        if df.empty:
            finish_job(db_session, job_id, "success", logs={
                "rows": 0,
                "reason": "no features for candidate dates",
                **coverage_stats,
            })
            return {"rows": 0}

        feature_df = _parse_features(df["features_json"])
        
        # 記錄缺失欄位
        missing_cols = [col for col in feature_names if col not in feature_df.columns]
        coverage_stats["missing_feature_columns"] = missing_cols
        
        for col in feature_names:
            if col not in feature_df.columns:
                feature_df[col] = np.nan
        feature_df = feature_df[feature_names]

        price_df = _load_price_universe(
            db_session,
            max(candidate_dates),
            df["stock_id"].astype(str).unique().tolist(),
            config.min_avg_volume,
        )
        
        coverage_stats["price_universe_rows"] = len(price_df)
        coverage_stats["price_universe_stocks"] = price_df["stock_id"].nunique() if not price_df.empty else 0

        chosen_date, chosen_df, fallback_logs = _choose_pick_date(
            candidate_dates,
            df,
            price_df,
            config.topn,
            config.min_avg_volume,
            config.fallback_days,
        )
        if chosen_date is None:
            reason = "no valid candidates after fallback"
            finish_job(db_session, job_id, "failed", error_text=reason, logs={
                "error": reason, 
                **fallback_logs,
                **coverage_stats,
            })
            raise ValueError(reason)

        selected_idx = chosen_df.index
        df = chosen_df.reset_index(drop=True)
        feature_df = feature_df.loc[selected_idx].reset_index(drop=True)
        feature_df, impute_stats = _impute_features(feature_df)

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
                    "pick_date": chosen_date,
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

        logs = {
            "rows": len(records),
            "pick_date": chosen_date.isoformat(),
            "model_id": model_version.model_id,
            **fallback_logs,
            **impute_stats,
            **coverage_stats,
            "min_avg_volume": config.min_avg_volume,
        }
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs={
            "error": str(exc),
            **coverage_stats,
        })
        raise
