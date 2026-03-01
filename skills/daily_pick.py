from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import joblib
except ModuleNotFoundError as exc:
    raise RuntimeError("Missing dependency 'joblib'. Install with `pip install -r requirements.txt`.") from exc
import numpy as np
import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import Feature, Job, ModelVersion, Pick, RawPrice
from skills.build_features import FEATURE_COLUMNS
from skills import regime, risk
from skills import tradability_filter
from skills import multi_agent_selector


# 選取前 8 個特徵作為 reason 說明
REASON_FEATURES = FEATURE_COLUMNS[:8]
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def _load_market_price_df(db_session: Session, target_date: date, ma_days: int) -> pd.DataFrame:
    start = target_date - timedelta(days=ma_days * 2)
    stmt = (
        select(RawPrice.trading_date, func.avg(RawPrice.close).label("avg_close"))
        .where(RawPrice.trading_date.between(start, target_date))
        .group_by(RawPrice.trading_date)
        .order_by(RawPrice.trading_date)
    )
    return pd.read_sql(stmt, db_session.get_bind())


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
    config,
    fallback_days: int,
) -> Tuple[date | None, pd.DataFrame, Dict[str, object]]:
    """選擇最佳選股日期。

    20 日平均成交值門檻由 risk.apply_liquidity_filter 控制。
    """
    best_date = None
    best_df = pd.DataFrame()
    best_valid = 0
    fallback_used = 0
    best_meta: Dict[str, object] = {}

    for idx, target_date in enumerate(candidate_dates[: fallback_days + 1]):
        target = target_date
        date_features = feature_df[feature_df["trading_date"] == target].copy()
        if date_features.empty:
            continue

        date_features = date_features[date_features["stock_id"].str.fullmatch(r"\d{4}")]
        if date_features.empty:
            continue
        date_features = date_features.drop_duplicates(subset=["stock_id", "trading_date"])
        pre_liquidity_count = len(date_features)

        date_prices = price_df[price_df["trading_date"] == target].copy()
        if date_prices.empty:
            continue

        latest_price = date_prices.dropna(subset=["close", "volume"])[["stock_id", "close", "volume"]]
        if latest_price.empty:
            continue

        eligible_turnover = risk.apply_liquidity_filter(
            price_df[price_df["trading_date"] <= target],
            config,
        )
        eligible = latest_price.merge(eligible_turnover[["stock_id"]], on="stock_id", how="inner")
        eligible = eligible.drop_duplicates(subset=["stock_id"])
        if eligible.empty:
            continue

        date_features = date_features.merge(eligible[["stock_id"]], on="stock_id", how="inner")
        valid_count = len(date_features)
        meta = {
            "candidate_count_before_liquidity": pre_liquidity_count,
            "candidate_count_after_liquidity": valid_count,
            "liquidity_excluded_count": max(pre_liquidity_count - valid_count, 0),
            "liquidity_excluded_ratio": (
                max(pre_liquidity_count - valid_count, 0) / pre_liquidity_count
                if pre_liquidity_count
                else 0.0
            ),
        }
        if valid_count > 0 and best_date is None:
            best_date = target_date
            best_df = date_features
            best_valid = valid_count
            fallback_used = idx
            best_meta = meta

        if valid_count >= topn:
            return target_date, date_features, {
                "fallback_days": idx,
                "valid_candidates": valid_count,
                "topn_returned": min(valid_count, topn),
                **meta,
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
        **best_meta,
    }


def _load_data_quality_degraded_context(session: Session) -> Dict[str, object]:
    latest = (
        session.query(Job)
        .filter(Job.job_name == "data_quality_check")
        .order_by(Job.started_at.desc())
        .limit(1)
        .one_or_none()
    )
    if latest is None or not latest.logs_json:
        return {"degraded_mode": False, "degraded_datasets": []}
    logs = latest.logs_json if isinstance(latest.logs_json, dict) else {}
    return {
        "degraded_mode": bool(logs.get("degraded_mode", False)),
        "degraded_datasets": list(logs.get("degraded_datasets", [])),
        "dq_mode": logs.get("data_quality_mode"),
    }


def _research_score_candidates(feature_df: pd.DataFrame) -> pd.Series:
    # research 降級模式：只使用技術+流動性特徵做啟發式排序，避免依賴缺失的 institutional 資料
    keys = ["ret_20", "breakout_20", "amt_ratio_20"]
    data = feature_df.copy()
    z = pd.DataFrame(index=data.index)
    for col in keys:
        vals = pd.to_numeric(data.get(col), errors="coerce").fillna(0.0)
        std = float(vals.std())
        if std == 0:
            z[col] = 0.0
        else:
            z[col] = (vals - vals.mean()) / std
    return 0.5 * z["ret_20"] + 0.3 * z["breakout_20"] + 0.2 * z["amt_ratio_20"]


def _should_use_research_fallback(config, dq_ctx: Dict[str, object]) -> bool:
    degraded_datasets = set(str(x) for x in dq_ctx.get("degraded_datasets", []))
    return (
        str(getattr(config, "data_quality_mode", "strict")).lower() == "research"
        and bool(dq_ctx.get("degraded_mode", False))
        and "raw_institutional" in degraded_datasets
    )


def _build_agent_dump_summary(agent_dump: Dict[str, object]) -> Dict[str, object]:
    if not agent_dump:
        return {}
    return dict(agent_dump.get("summary", {}))


def _write_run_manifest(
    job_id: str,
    chosen_date: date,
    rows_df: pd.DataFrame,
    logs: Dict[str, object],
    selection_mode: str,
    score_mode: str,
    config,
    selection_meta: Dict[str, object],
    dq_ctx: Dict[str, object],
    weights_used: Dict[str, float] | None = None,
    agent_dump: Dict[str, object] | None = None,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    picks_payload = []
    sorted_df = rows_df.sort_values("score", ascending=False).reset_index(drop=True)
    for i, row in sorted_df.iterrows():
        picks_payload.append({"stock_id": str(row["stock_id"]), "rank": int(i + 1), "score": float(row["score"])})

    universe = {
        "valid_stock_universe_count": logs.get("valid_stock_universe_count"),
        "liquidity_excluded_ratio": logs.get("liquidity_excluded_ratio"),
        "tradability": selection_meta.get("tradability", {}),
        "missing_feature_columns": logs.get("missing_feature_columns", []),
    }
    manifest = {
        "job_id": job_id,
        "pick_date": chosen_date.isoformat(),
        "selection_mode": selection_mode,
        "score_mode": score_mode,
        "data_quality_mode": str(getattr(config, "data_quality_mode", "strict")),
        "degraded_mode": bool(dq_ctx.get("degraded_mode", False)),
        "degraded_datasets": list(dq_ctx.get("degraded_datasets", [])),
        "effective_topn": int(logs.get("effective_topn", getattr(config, "topn", len(picks_payload)))),
        "universe": universe,
        "weights_requested": getattr(config, "multi_agent_weights", None) if selection_mode == "multi_agent" else None,
        "weights_used": weights_used if selection_mode == "multi_agent" else None,
        "picks": picks_payload,
    }
    if selection_mode == "multi_agent":
        manifest["agent_dump_summary"] = _build_agent_dump_summary(agent_dump or {})

    manifest_path = ARTIFACTS_DIR / f"run_manifest_daily_pick_{job_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


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

        selection_mode = str(getattr(config, "selection_mode", "model")).lower()
        if selection_mode not in {"model", "multi_agent"}:
            selection_mode = "model"

        model_version = _load_latest_model(db_session)
        model = None
        feature_names = list(FEATURE_COLUMNS)
        if selection_mode == "model":
            if model_version is None:
                raise ValueError("No trained model found")
            artifact = joblib.load(model_version.artifact_path)
            model = artifact["model"]
            feature_names = artifact["feature_names"]
        
        # 取得有效股票 universe（排除 ETF、權證等）
        valid_universe_df = risk.get_universe(db_session, max(candidate_dates), config)
        tradability_logs: Dict[str, object] = {}
        if getattr(config, "enable_tradability_filter", True):
            valid_universe_df, tradability_logs = tradability_filter.filter_universe(
                db_session,
                valid_universe_df,
                max(candidate_dates),
                return_stats=True,
            )
            if tradability_logs.get("missing_status_count", 0) > 0:
                tradability_logs["warning"] = (
                    f"tradability status missing for {tradability_logs['missing_status_count']} stocks; "
                    "kept as tradable by policy"
                )
        else:
            tradability_logs = {"tradability_filter": "disabled"}
        valid_stocks = set(valid_universe_df["stock_id"].astype(str).tolist())
        coverage_stats["valid_stock_universe_count"] = len(valid_stocks)
        coverage_stats["tradability"] = tradability_logs
        
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
        )
        
        coverage_stats["price_universe_rows"] = len(price_df)
        coverage_stats["price_universe_stocks"] = price_df["stock_id"].nunique() if not price_df.empty else 0

        # ── 大盤過濾器：空頭市場減碼 ──
        effective_topn = config.topn
        bear_market = False
        if getattr(config, "market_filter_enabled", False):
            ma_days = getattr(config, "market_filter_ma_days", 60)
            bear_topn = getattr(config, "market_filter_bear_topn", config.topn // 2)
            detector = regime.get_regime_detector(config)
            market_df = _load_market_price_df(db_session, max(candidate_dates), ma_days)
            regime_result = detector.detect(market_df, config)
            bear_market = regime_result.get("regime") == "BEAR"
            coverage_stats["regime_detector"] = getattr(config, "regime_detector", "ma")
            coverage_stats["regime_meta"] = regime_result.get("meta", {})
            if bear_market:
                effective_topn = bear_topn
                coverage_stats["market_filter"] = "BEAR"
                coverage_stats["effective_topn"] = effective_topn
            else:
                coverage_stats["market_filter"] = "BULL"
        else:
            coverage_stats["market_filter"] = "disabled"
            coverage_stats["effective_topn"] = effective_topn

        chosen_date, chosen_df, fallback_logs = _choose_pick_date(
            candidate_dates,
            df,
            price_df,
            effective_topn,
            config,
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

        dq_ctx = _load_data_quality_degraded_context(db_session)
        degraded_datasets = set(str(x) for x in dq_ctx.get("degraded_datasets", []))
        use_research_fallback = _should_use_research_fallback(config, dq_ctx)
        coverage_stats["degraded_mode"] = bool(dq_ctx.get("degraded_mode", False))
        coverage_stats["degraded_datasets"] = sorted(degraded_datasets)

        records: List[Dict] = []
        selection_meta = {
            "tradability": tradability_logs,
            "liquidity": {
                "excluded_ratio": fallback_logs.get("liquidity_excluded_ratio", 0.0),
                "excluded_count": fallback_logs.get("liquidity_excluded_count", 0),
                "before_count": fallback_logs.get("candidate_count_before_liquidity", 0),
                "after_count": fallback_logs.get("candidate_count_after_liquidity", 0),
            },
        }
        weights_used_manifest: Dict[str, float] | None = None
        agent_dump: Dict[str, object] | None = None
        if selection_mode == "multi_agent":
            ma_topn = int(getattr(config, "multi_agent_topn", effective_topn) or effective_topn)
            picks_df, agent_dump = multi_agent_selector.run_multi_agent_selection(
                feature_df=feature_df,
                stock_ids=df["stock_id"],
                pick_date=chosen_date,
                topn=min(ma_topn, effective_topn),
                config=config,
                dq_ctx=dq_ctx,
                selection_meta=selection_meta,
            )
            coverage_stats["score_mode"] = (
                "multi_agent_degraded" if bool(dq_ctx.get("degraded_mode", False)) else "multi_agent"
            )
            df = picks_df.copy()
            if not df.empty:
                first_meta = df.iloc[0]["reason_json"].get("_selection_meta", {})
                weights_used_manifest = dict(first_meta.get("weights_used", {}))
        else:
            if use_research_fallback:
                scores = _research_score_candidates(feature_df).values
                coverage_stats["score_mode"] = "research_tech_liquidity_fallback"
            else:
                scores = model.predict(feature_df.values)
                coverage_stats["score_mode"] = "model"
            df["score"] = scores

            percentile_map = {}
            for feat in REASON_FEATURES:
                if feat not in feature_df.columns:
                    continue
                vals = pd.to_numeric(feature_df[feat], errors="coerce")
                percentile_map[feat] = vals.rank(pct=True)

            df = risk.pick_topn(df, effective_topn)
            for _, row in df.iterrows():
                reasons = {}
                for feat in REASON_FEATURES:
                    if feat not in feature_df.columns:
                        continue
                    value = float(feature_df.loc[row.name, feat])
                    pct = float(percentile_map[feat].loc[row.name]) if feat in percentile_map else None
                    reasons[feat] = {"value": value, "percentile": pct}
                reasons["_selection_meta"] = dict(selection_meta, dq_ctx=dq_ctx, selection_mode="model")

                records.append(
                    {
                        "pick_date": chosen_date,
                        "stock_id": row["stock_id"],
                        "score": float(row["score"]),
                        "model_id": model_version.model_id if model_version else "n/a",
                        "reason_json": reasons,
                    }
                )

        if selection_mode == "multi_agent":
            for _, row in df.iterrows():
                records.append(
                    {
                        "pick_date": chosen_date,
                        "stock_id": row["stock_id"],
                        "score": float(row["score"]),
                        "model_id": model_version.model_id if model_version else "multi_agent",
                        "reason_json": row["reason_json"],
                    }
                )

        if records:
            # 先清除同一天的舊 picks，避免殘留不同 stock_id 的過期資料
            db_session.query(Pick).filter(Pick.pick_date == chosen_date).delete()

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
            "model_id": model_version.model_id if model_version else "multi_agent",
            **fallback_logs,
            **impute_stats,
            **coverage_stats,
            "min_avg_turnover": config.min_avg_turnover,
            "min_amt_20": getattr(config, "min_amt_20", None),
            "selection_mode": selection_mode,
        }
        _write_run_manifest(
            job_id=job_id,
            chosen_date=chosen_date,
            rows_df=df,
            logs=logs,
            selection_mode=selection_mode,
            score_mode=str(coverage_stats.get("score_mode", "model")),
            config=config,
            selection_meta=selection_meta,
            dq_ctx=dq_ctx,
            weights_used=weights_used_manifest,
            agent_dump=agent_dump,
        )
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
