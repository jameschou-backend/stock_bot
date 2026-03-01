#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import func, select

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from app.models import Feature, ModelVersion, PriceAdjustFactor, RawInstitutional, RawMarginShort, RawPrice, RawFundamental, RawThemeFlow
from skills import multi_agent_selector, risk, tradability_filter
from skills.daily_pick import _impute_features, _parse_features, _research_score_candidates
from skills.build_features import FEATURE_COLUMNS


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MANIFEST_DIR = ARTIFACTS_DIR / "experiment_manifests"


def _to_config(base_config, spec: Dict[str, object], topn: int):
    return replace(
        base_config,
        selection_mode=str(spec.get("selection_mode", base_config.selection_mode)).lower(),
        data_quality_mode=str(spec.get("data_quality_mode", base_config.data_quality_mode)).lower(),
        multi_agent_weights=dict(spec.get("multi_agent_weights", base_config.multi_agent_weights or {})),
        topn=int(spec.get("topn", topn)),
    )


def _monthly_rebalance_dates(trading_dates: List[date]) -> List[date]:
    out: List[date] = []
    prev = None
    for d in sorted(trading_dates):
        ym = (d.year, d.month)
        if ym != prev:
            out.append(d)
            prev = ym
    return out


def _load_feature_rows(session, target_date: date, stock_ids: List[str]) -> pd.DataFrame:
    stmt = (
        select(Feature.stock_id, Feature.trading_date, Feature.features_json)
        .where(Feature.trading_date == target_date)
        .where(Feature.stock_id.in_(stock_ids))
        .order_by(Feature.stock_id)
    )
    return pd.read_sql(stmt, session.get_bind())


def _load_price_frame(session, start_date: date, end_date: date) -> pd.DataFrame:
    stmt = (
        select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close, RawPrice.volume)
        .where(RawPrice.trading_date.between(start_date, end_date))
        .order_by(RawPrice.trading_date, RawPrice.stock_id)
    )
    df = pd.read_sql(stmt, session.get_bind())
    if df.empty:
        return df
    df["stock_id"] = df["stock_id"].astype(str)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    return df


def _load_adjust_factors(session, start_date: date, end_date: date) -> pd.DataFrame:
    stmt = (
        select(PriceAdjustFactor.stock_id, PriceAdjustFactor.trading_date, PriceAdjustFactor.adj_factor)
        .where(PriceAdjustFactor.trading_date.between(start_date, end_date))
        .order_by(PriceAdjustFactor.trading_date, PriceAdjustFactor.stock_id)
    )
    df = pd.read_sql(stmt, session.get_bind())
    if df.empty:
        return df
    df["stock_id"] = df["stock_id"].astype(str)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0)
    return df


def _merge_adjusted_close(price_df: pd.DataFrame, factor_df: pd.DataFrame, use_adjusted_price: bool) -> pd.DataFrame:
    out = price_df.copy()
    if out.empty:
        return out
    if use_adjusted_price and not factor_df.empty:
        out = out.merge(factor_df, on=["stock_id", "trading_date"], how="left")
        out["adj_factor"] = out["adj_factor"].fillna(1.0)
        out["px"] = out["close"] * out["adj_factor"]
    else:
        out["px"] = out["close"]
    return out


def _resolve_degraded_for_date(session, cfg, asof_date: date, universe_count: int) -> Dict[str, object]:
    datasets = []
    inst_count = int(
        session.query(func.count(func.distinct(RawInstitutional.stock_id)))
        .filter(RawInstitutional.trading_date == asof_date)
        .scalar()
        or 0
    )
    margin_count = int(
        session.query(func.count(func.distinct(RawMarginShort.stock_id)))
        .filter(RawMarginShort.trading_date == asof_date)
        .scalar()
        or 0
    )
    # 基本面為月頻：以 as-of（當日以前最近可得）計算覆蓋率，避免精確日期比對造成誤判 degraded
    fund_subq = (
        session.query(
            RawFundamental.stock_id.label("stock_id"),
            func.max(RawFundamental.trading_date).label("latest_fund_date"),
        )
        .filter(RawFundamental.trading_date <= asof_date)
        .group_by(RawFundamental.stock_id)
        .subquery()
    )
    fund_count = int(session.query(func.count()).select_from(fund_subq).scalar() or 0)
    theme_count = int(
        session.query(func.count(func.distinct(RawThemeFlow.theme_id)))
        .filter(RawThemeFlow.trading_date == asof_date)
        .scalar()
        or 0
    )
    if universe_count > 0 and inst_count < int(universe_count * cfg.dq_coverage_ratio_institutional):
        datasets.append("raw_institutional")
    if universe_count > 0 and margin_count < int(universe_count * cfg.dq_coverage_ratio_margin):
        datasets.append("raw_margin_short")
    if fund_count == 0:
        datasets.append("raw_fundamentals")
    if theme_count == 0:
        datasets.append("raw_theme_flow")
    return {"degraded_mode": len(datasets) > 0, "degraded_datasets": sorted(set(datasets)), "dq_mode": cfg.data_quality_mode}


def _fund_snapshot_asof(session, asof_date: date, stock_ids: List[str]) -> pd.DataFrame:
    if not stock_ids:
        return pd.DataFrame(columns=["stock_id", "fund_revenue_mom", "fund_revenue_yoy", "fund_revenue_trend_3m"])
    stmt = (
        select(
            RawFundamental.stock_id,
            RawFundamental.trading_date,
            RawFundamental.revenue_current_month,
            RawFundamental.revenue_last_month,
            RawFundamental.revenue_last_year,
            RawFundamental.revenue_mom,
            RawFundamental.revenue_yoy,
        )
        .where(RawFundamental.stock_id.in_(stock_ids))
        .where(RawFundamental.trading_date <= asof_date)
        .where(RawFundamental.trading_date >= asof_date - timedelta(days=760))
        .order_by(RawFundamental.stock_id, RawFundamental.trading_date)
    )
    raw = pd.read_sql(stmt, session.get_bind())
    if raw.empty:
        return pd.DataFrame(columns=["stock_id", "fund_revenue_mom", "fund_revenue_yoy", "fund_revenue_trend_3m"])
    raw["stock_id"] = raw["stock_id"].astype(str)
    raw["trading_date"] = pd.to_datetime(raw["trading_date"], errors="coerce")
    raw["revenue_current_month"] = pd.to_numeric(raw["revenue_current_month"], errors="coerce")
    raw["revenue_last_month"] = pd.to_numeric(raw["revenue_last_month"], errors="coerce")
    raw["revenue_last_year"] = pd.to_numeric(raw["revenue_last_year"], errors="coerce")
    raw["revenue_mom"] = pd.to_numeric(raw["revenue_mom"], errors="coerce")
    raw["revenue_yoy"] = pd.to_numeric(raw["revenue_yoy"], errors="coerce")
    raw = raw.sort_values(["stock_id", "trading_date"])
    raw["rev_prev_1m"] = raw.groupby("stock_id")["revenue_current_month"].shift(1)
    raw["rev_prev_12m"] = raw.groupby("stock_id")["revenue_current_month"].shift(12)
    # 舊資料若未填 yoy/mom，使用營收欄位回補（避免 fund agent 永久無效）
    prev_month = raw["revenue_last_month"].where(raw["revenue_last_month"].notna(), raw["rev_prev_1m"])
    prev_year = raw["revenue_last_year"].where(raw["revenue_last_year"].notna(), raw["rev_prev_12m"])
    mom_fallback = raw["revenue_current_month"] / prev_month.replace(0, np.nan) - 1.0
    yoy_fallback = raw["revenue_current_month"] / prev_year.replace(0, np.nan) - 1.0
    raw["revenue_mom"] = raw["revenue_mom"].fillna(mom_fallback)
    raw["revenue_yoy"] = raw["revenue_yoy"].fillna(yoy_fallback)
    out_rows = []
    for sid, g in raw.groupby("stock_id", sort=False):
        g = g.sort_values("trading_date")
        latest = g.iloc[-1]
        trend = pd.to_numeric(g["revenue_yoy"], errors="coerce").rolling(3, min_periods=2).mean().iloc[-1]
        out_rows.append(
            {
                "stock_id": sid,
                "fund_revenue_mom": float(latest["revenue_mom"]) if pd.notna(latest["revenue_mom"]) else np.nan,
                "fund_revenue_yoy": float(latest["revenue_yoy"]) if pd.notna(latest["revenue_yoy"]) else np.nan,
                "fund_revenue_trend_3m": float(trend) if pd.notna(trend) else np.nan,
            }
        )
    return pd.DataFrame(out_rows)


def _period_return(picks: List[str], date_a: date, date_b: date, px_df: pd.DataFrame, tx_cost: float) -> tuple[float | None, Dict[str, object]]:
    if not picks:
        return None, {"reason": "empty_picks", "rows": []}
    period = px_df[(px_df["trading_date"].isin([date_a, date_b])) & (px_df["stock_id"].isin(picks))]
    if period.empty:
        return None, {"reason": "no_price_rows", "rows": []}
    ent = period[period["trading_date"] == date_a][["stock_id", "px"]].rename(columns={"px": "entry_px"})
    ext = period[period["trading_date"] == date_b][["stock_id", "px"]].rename(columns={"px": "exit_px"})
    merged = ent.merge(ext, on="stock_id", how="inner")
    if merged.empty:
        return None, {"reason": "no_matched_entry_exit", "rows": []}

    debug_rows = []
    for _, row in merged.iterrows():
        ep = float(row["entry_px"]) if pd.notna(row["entry_px"]) else np.nan
        xp = float(row["exit_px"]) if pd.notna(row["exit_px"]) else np.nan
        zero_price = bool((np.isfinite(ep) and ep <= 0) or (np.isfinite(xp) and xp <= 0))
        has_nan = bool(not np.isfinite(ep) or not np.isfinite(xp))
        extreme_price = bool((np.isfinite(ep) and abs(ep) > 1e6) or (np.isfinite(xp) and abs(xp) > 1e6))
        debug_rows.append(
            {
                "stock_id": str(row["stock_id"]),
                "entry_price": ep,
                "exit_price": xp,
                "zero_price": zero_price,
                "nan_price": has_nan,
                "extreme_price": extreme_price,
            }
        )

    merged = merged[(merged["entry_px"] > 0) & (merged["exit_px"] > 0)]
    if merged.empty:
        return None, {"reason": "all_invalid_entry_exit_prices", "rows": debug_rows}
    r = (merged["exit_px"] / merged["entry_px"] - 1.0) - tx_cost
    r = r[np.isfinite(r)]
    # 避免極端資料異常（如接近 0 的價格）造成非現實報酬污染評估
    r = r[(r > -0.99) & (r < 5.0)]
    if r.empty:
        return None, {"reason": "all_returns_filtered_extreme_or_nonfinite", "rows": debug_rows}
    return float(r.mean()), {"reason": "ok", "rows": debug_rows, "avg_period_return": float(r.mean())}


def _calc_metrics(returns: List[float], overlaps: List[float], turnover_list: List[float], degraded_flags: List[bool], start: date, end: date) -> Dict[str, object]:
    clean_returns = [float(x) for x in returns if np.isfinite(x)]
    if not clean_returns:
        return {
            "total_return": 0.0,
            "cagr": "N/A: no valid period returns",
            "max_drawdown": 0.0,
            "volatility": "N/A: no valid period returns",
            "sharpe": "N/A: no valid period returns",
            "turnover": float(np.mean(turnover_list)) if turnover_list else 0.0,
            "win_rate": "N/A: no valid period returns",
            "average_holding_return": "N/A: no valid period returns",
            "picks_stability": float(np.mean(overlaps)) if overlaps else 0.0,
            "degraded_period_ratio": float(np.mean([1.0 if x else 0.0 for x in degraded_flags])) if degraded_flags else 0.0,
        }

    ret_arr = np.array(clean_returns, dtype=float)
    equity = np.cumprod(1 + ret_arr)
    total_return = float(equity[-1] - 1.0)
    years = max((end - start).days / 365.25, 1e-9)
    cagr = float((equity[-1] ** (1 / years)) - 1.0) if years > 0 and equity[-1] > 0 else "N/A: non-positive ending equity"
    rolling_peak = np.maximum.accumulate(equity)
    dd = (equity / rolling_peak) - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    vol = float(ret_arr.std() * np.sqrt(12)) if len(ret_arr) > 1 else "N/A: <2 periods"
    sharpe = float((ret_arr.mean() / ret_arr.std()) * np.sqrt(12)) if len(ret_arr) > 1 and ret_arr.std() > 0 else "N/A: zero std or <2 periods"
    win_rate = float((ret_arr > 0).mean())
    avg_ret = float(ret_arr.mean())
    return {
        "total_return": total_return if np.isfinite(total_return) else "N/A: non-finite",
        "cagr": cagr if (not isinstance(cagr, float) or np.isfinite(cagr)) else "N/A: non-finite",
        "max_drawdown": max_dd if np.isfinite(max_dd) else "N/A: non-finite",
        "volatility": vol if (not isinstance(vol, float) or np.isfinite(vol)) else "N/A: non-finite",
        "sharpe": sharpe if (not isinstance(sharpe, float) or np.isfinite(sharpe)) else "N/A: non-finite",
        "turnover": float(np.mean(turnover_list)) if turnover_list else 0.0,
        "win_rate": win_rate,
        "average_holding_return": avg_ret,
        "picks_stability": float(np.mean(overlaps)) if overlaps else 0.0,
        "degraded_period_ratio": float(np.mean([1.0 if x else 0.0 for x in degraded_flags])) if degraded_flags else 0.0,
    }


def _write_json_md(prefix: str, payload: Dict[str, object], md_lines: List[str]) -> Tuple[Path, Path]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    jp = ARTIFACTS_DIR / f"{prefix}.json"
    mp = ARTIFACTS_DIR / f"{prefix}.md"
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    mp.write_text("\n".join(md_lines), encoding="utf-8")
    return jp, mp


def run_experiment(
    experiment_id: str,
    start_date: date,
    end_date: date,
    cfg,
    resume: bool = False,
) -> Dict[str, object]:
    eval_json = ARTIFACTS_DIR / f"evaluation_{experiment_id}.json"
    picks_json = ARTIFACTS_DIR / f"experiment_picks_{experiment_id}.json"
    if resume and eval_json.exists() and picks_json.exists():
        return json.loads(eval_json.read_text(encoding="utf-8"))

    with get_session() as session:
        model = None
        feature_names = None
        ma_feature_names = sorted(
            set(FEATURE_COLUMNS)
            | set(sum(multi_agent_selector.AGENT_REQUIRED_COLUMNS.values(), []))
        )
        model_id = None
        if cfg.selection_mode == "model":
            mv = session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).first()
            if mv is None:
                raise ValueError("No model_versions available for model selection_mode")
            import joblib

            artifact = joblib.load(mv.artifact_path)
            model = artifact["model"]
            feature_names = artifact["feature_names"]
            model_id = mv.model_id

        price_df = _load_price_frame(session, start_date, end_date)
        factor_df = _load_adjust_factors(session, start_date, end_date)
        px_df = _merge_adjusted_close(price_df, factor_df, bool(getattr(cfg, "use_adjusted_price", True)))

        rows = (
            session.query(Feature.trading_date)
            .filter(Feature.trading_date.between(start_date, end_date))
            .distinct()
            .order_by(Feature.trading_date)
            .all()
        )
        trading_dates = [row[0] for row in rows]
        rb_dates = _monthly_rebalance_dates(trading_dates)

        universe_df = risk.get_universe(session, end_date, cfg)
        if getattr(cfg, "enable_tradability_filter", True):
            universe_df, tradability_stats = tradability_filter.filter_universe(session, universe_df, end_date, return_stats=True)
        else:
            tradability_stats = {"tradability_filter": "disabled"}
        stock_ids = universe_df["stock_id"].astype(str).tolist()
        universe_count = len(stock_ids)

        picks_rows: List[Dict[str, object]] = []
        period_returns: List[float] = []
        period_debug_rows: List[Dict[str, object]] = []
        overlaps: List[float] = []
        turnover_list: List[float] = []
        degraded_flags: List[bool] = []
        skipped_periods: List[Dict[str, object]] = []
        feature_columns_observed: set[str] = set()
        prev_pick_set: set[str] | None = None

        manifest_exp_dir = MANIFEST_DIR / experiment_id
        manifest_exp_dir.mkdir(parents=True, exist_ok=True)

        for idx, rb in enumerate(rb_dates):
            dq_ctx = _resolve_degraded_for_date(session, cfg, rb, universe_count)
            degraded_flags.append(bool(dq_ctx["degraded_mode"]))
            if cfg.data_quality_mode == "strict" and dq_ctx["degraded_mode"]:
                skipped_periods.append({"date": rb.isoformat(), "reason": f"strict mode degraded: {dq_ctx['degraded_datasets']}"})
                continue

            feat_rows = _load_feature_rows(session, rb, stock_ids)
            if feat_rows.empty:
                skipped_periods.append({"date": rb.isoformat(), "reason": "no features on rebalance date"})
                continue
            fdf = _parse_features(feat_rows["features_json"])
            feature_columns_observed.update([str(c) for c in fdf.columns.tolist()])
            if feature_names is None:
                if cfg.selection_mode == "multi_agent":
                    feature_names = list(ma_feature_names)
                else:
                    feature_names = list(fdf.columns)
            for c in feature_names:
                if c not in fdf.columns:
                    fdf[c] = np.nan
            work = feat_rows.reset_index(drop=True).copy()
            # Round4: fundamentals 以 as-of snapshot 覆寫，避免月頻資料和 rebalance date 精確對齊造成缺失
            fund_asof = _fund_snapshot_asof(session, rb, work["stock_id"].astype(str).tolist())
            if not fund_asof.empty:
                fm = fund_asof.set_index("stock_id")
                sid_series = work["stock_id"].astype(str)
                for c in ["fund_revenue_mom", "fund_revenue_yoy", "fund_revenue_trend_3m"]:
                    if c in fdf.columns and c in fm.columns:
                        fdf[c] = sid_series.map(fm[c]).values
            fdf = fdf[feature_names]
            fdf, _ = _impute_features(fdf)
            selection_meta = {"tradability": tradability_stats, "liquidity": {}}

            if cfg.selection_mode == "multi_agent":
                picks_df, agent_dump = multi_agent_selector.run_multi_agent_selection(
                    feature_df=fdf,
                    stock_ids=work["stock_id"].astype(str),
                    pick_date=rb,
                    topn=int(cfg.topn),
                    config=cfg,
                    dq_ctx=dq_ctx,
                    selection_meta=selection_meta,
                )
                score_mode = "multi_agent_degraded" if dq_ctx["degraded_mode"] else "multi_agent"
                weights_used = (
                    picks_df.iloc[0]["reason_json"].get("_selection_meta", {}).get("weights_used", {})
                    if not picks_df.empty
                    else {}
                )
                agent_summary = agent_dump.get("summary", {})
            else:
                if cfg.data_quality_mode == "research" and "raw_institutional" in set(dq_ctx["degraded_datasets"]):
                    scores = _research_score_candidates(fdf).values
                    score_mode = "research_tech_liquidity_fallback"
                else:
                    scores = model.predict(fdf.values)
                    score_mode = "model"
                work["score"] = scores
                picks_df = risk.pick_topn(work[["stock_id", "score"]].copy(), int(cfg.topn)).reset_index(drop=True)
                picks_df["reason_json"] = picks_df.apply(
                    lambda r: {"_selection_meta": {"selection_mode": "model", "dq_ctx": dq_ctx, **selection_meta}}, axis=1
                )
                weights_used = {}
                agent_summary = {}

            pick_set = set(picks_df["stock_id"].astype(str).tolist())
            if prev_pick_set is not None:
                ov = len(prev_pick_set & pick_set) / float(max(int(cfg.topn), 1))
                overlaps.append(ov)
                turnover_list.append(1.0 - ov)
            prev_pick_set = pick_set

            run_id = f"{experiment_id}_{rb.isoformat()}"
            manifest = {
                "job_id": run_id,
                "pick_date": rb.isoformat(),
                "selection_mode": cfg.selection_mode,
                "score_mode": score_mode,
                "data_quality_mode": cfg.data_quality_mode,
                "degraded_mode": dq_ctx["degraded_mode"],
                "degraded_datasets": dq_ctx["degraded_datasets"],
                "effective_topn": int(cfg.topn),
                "weights_requested": cfg.multi_agent_weights if cfg.selection_mode == "multi_agent" else None,
                "weights_used": weights_used if cfg.selection_mode == "multi_agent" else None,
                "agent_dump_summary": agent_summary if cfg.selection_mode == "multi_agent" else None,
                "picks": [
                    {"stock_id": str(sid), "rank": int(i + 1), "score": float(sc)}
                    for i, (sid, sc) in enumerate(zip(picks_df["stock_id"], picks_df["score"]))
                ],
            }
            (manifest_exp_dir / f"run_manifest_{run_id}.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            picks_rows.extend(
                [
                    {
                        "experiment_id": experiment_id,
                        "date": rb.isoformat(),
                        "stock_id": str(row["stock_id"]),
                        "rank": int(i + 1),
                        "score": float(row["score"]),
                        "reason_json": row["reason_json"],
                        "degraded_mode": bool(dq_ctx["degraded_mode"]),
                        "degraded_datasets": list(dq_ctx["degraded_datasets"]),
                        "selection_mode": cfg.selection_mode,
                        "score_mode": score_mode,
                    }
                    for i, (_, row) in enumerate(picks_df.iterrows())
                ]
            )

            if idx + 1 < len(rb_dates):
                nxt = rb_dates[idx + 1]
                ret, ret_dbg = _period_return(list(pick_set), rb, nxt, px_df, float(getattr(cfg, "transaction_cost_pct", 0.001425)))
                period_debug_rows.append(
                    {
                        "entry_date": rb.isoformat(),
                        "exit_date": nxt.isoformat(),
                        "period_return": ret,
                        "calc_status": ret_dbg.get("reason"),
                        "rows": ret_dbg.get("rows", []),
                    }
                )
                if ret is not None:
                    period_returns.append(ret)

        metrics = _calc_metrics(period_returns, overlaps, turnover_list, degraded_flags, start_date, end_date)
        invalid_result = any(
            isinstance(v, float) and (not np.isfinite(v))
            for v in metrics.values()
        )
        payload = {
            "experiment_id": experiment_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "selection_mode": cfg.selection_mode,
            "data_quality_mode": cfg.data_quality_mode,
            "weights_requested": cfg.multi_agent_weights if cfg.selection_mode == "multi_agent" else None,
            "rebalance_dates": [d.isoformat() for d in rb_dates],
            "number_of_rebalance_dates": len(rb_dates),
            "degraded_ratio": float(np.mean([1.0 if x else 0.0 for x in degraded_flags])) if degraded_flags else 0.0,
            "skipped_periods": skipped_periods,
            "metrics": metrics,
            "invalid_result": bool(invalid_result),
            "manifest_dir": str(manifest_exp_dir),
            "model_id": model_id,
            "feature_columns_observed": sorted(set(feature_columns_observed)),
            "feature_columns_input_selector": sorted(set(feature_names or [])),
        }
        picks_json.write_text(json.dumps(picks_rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        md_lines = [
            f"# Evaluation {experiment_id}",
            "",
            f"- selection_mode: `{cfg.selection_mode}`",
            f"- data_quality_mode: `{cfg.data_quality_mode}`",
            f"- rebalance_dates: `{len(rb_dates)}`",
            f"- degraded_ratio: `{payload['degraded_ratio']:.2%}`",
            "",
            "## Metrics",
            f"- total_return: `{metrics['total_return']}`",
            f"- cagr: `{metrics['cagr']}`",
            f"- max_drawdown: `{metrics['max_drawdown']}`",
            f"- volatility: `{metrics['volatility']}`",
            f"- sharpe: `{metrics['sharpe']}`",
            f"- turnover: `{metrics['turnover']}`",
            f"- win_rate: `{metrics['win_rate']}`",
            f"- average_holding_return: `{metrics['average_holding_return']}`",
            f"- picks_stability: `{metrics['picks_stability']}`",
            "",
        ]
        _write_json_md(f"evaluation_{experiment_id}", payload, md_lines)
        if cfg.selection_mode == "model":
            debug_md = ARTIFACTS_DIR / f"debug_evaluation_model_{experiment_id}.md"
            anomaly_rows = [
                (p["entry_date"], p["exit_date"], rr)
                for p in period_debug_rows
                for rr in p["rows"]
                if rr.get("zero_price") or rr.get("nan_price") or rr.get("extreme_price")
            ]
            dbg_lines = [
                f"# Debug Evaluation Model {experiment_id}",
                "",
                f"- invalid_result: `{payload['invalid_result']}`",
                f"- periods_checked: `{len(period_debug_rows)}`",
                f"- anomaly_rows: `{len(anomaly_rows)}`",
                "- inf/nan root cause path: `period_return = (exit_px / entry_px - 1) - tx_cost`；若 `entry_px=0` 會導致除零，進而污染 total_return/cagr/average_holding_return 與波動指標。",
                "",
                "## Period Diagnostics",
            ]
            for p in period_debug_rows:
                dbg_lines.extend(
                    [
                        f"### {p['entry_date']} -> {p['exit_date']}",
                        f"- period_return: `{p['period_return']}`",
                        f"- calc_status: `{p['calc_status']}`",
                        "",
                        "| stock_id | entry_price | exit_price | zero_price | nan_price | extreme_price |",
                        "|---|---:|---:|---|---|---|",
                    ]
                )
                for rr in p["rows"][:50]:
                    dbg_lines.append(
                        f"| {rr['stock_id']} | {rr['entry_price']} | {rr['exit_price']} | "
                        f"{rr['zero_price']} | {rr['nan_price']} | {rr['extreme_price']} |"
                    )
                dbg_lines.append("")
            debug_md.write_text("\n".join(dbg_lines), encoding="utf-8")
        return payload


def _load_matrix(path: Path) -> Dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one experiment or matrix entry")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--selection-mode", type=str, required=True)
    parser.add_argument("--dq-mode", type=str, required=True)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--weights-json", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base = load_config()
    spec = {
        "selection_mode": args.selection_mode,
        "data_quality_mode": args.dq_mode,
        "topn": args.topn,
    }
    if args.weights_json:
        spec["multi_agent_weights"] = json.loads(args.weights_json)
    cfg = _to_config(base, spec, topn=args.topn)
    out = run_experiment(
        experiment_id=args.experiment_id,
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        cfg=cfg,
        resume=args.resume,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
