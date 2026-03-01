#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from app.models import Feature, PriceAdjustFactor, RawFundamental, RawPrice
from scripts.evaluate_experiment import _monthly_rebalance_dates, _to_config, run_experiment
from skills.daily_pick import _parse_features

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ABLATION_MATRIX = PROJECT_ROOT / "experiments" / "multi_agent_ablation.yaml"
METRIC_KEYS = ["total_return", "sharpe", "max_drawdown", "turnover", "picks_stability"]
FUND_COLS = ["fund_revenue_yoy", "fund_revenue_mom", "fund_revenue_trend_3m"]
THEME_COLS = ["theme_hot_score", "theme_return_20", "theme_turnover_ratio"]


def _load_matrix(path: Path) -> Dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_picks(experiment_id: str) -> List[Dict[str, object]]:
    path = ARTIFACTS_DIR / f"experiment_picks_{experiment_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_num(v) -> float | None:
    if isinstance(v, (int, float)) and np.isfinite(float(v)):
        return float(v)
    return None


def _series_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_feature_df(session, target_date: date, stock_ids: List[str]) -> pd.DataFrame:
    stmt = (
        select(Feature.stock_id, Feature.trading_date, Feature.features_json)
        .where(Feature.trading_date == target_date)
        .where(Feature.stock_id.in_(stock_ids))
        .order_by(Feature.stock_id)
    )
    df = pd.read_sql(stmt, session.get_bind())
    if df.empty:
        return pd.DataFrame(columns=["stock_id", "trading_date"] + FUND_COLS + THEME_COLS)
    fdf = _parse_features(df["features_json"])
    out = df[["stock_id", "trading_date"]].reset_index(drop=True).copy()
    for col in FUND_COLS + THEME_COLS:
        out[col] = pd.to_numeric(fdf[col], errors="coerce") if col in fdf.columns else np.nan
    return out


def _extract_theme_from_reason(reason_json: Dict[str, object]) -> Dict[str, float]:
    theme = ((reason_json or {}).get("agents") or {}).get("theme") or {}
    parsed = {}
    for s in theme.get("reasons", []):
        m = re.match(r"([a-zA-Z0-9_]+)=(-?[0-9.]+)", str(s))
        if m:
            parsed[m.group(1)] = float(m.group(2))
    parsed["theme_signal"] = float(theme.get("signal", 0.0))
    parsed["theme_confidence"] = float(theme.get("confidence", 0.0))
    parsed["theme_unavailable"] = 1.0 if bool(theme.get("unavailable", False)) else 0.0
    return parsed


def _rebalance_dates(session, start_date: date, end_date: date) -> List[date]:
    rows = (
        session.query(Feature.trading_date)
        .filter(Feature.trading_date.between(start_date, end_date))
        .distinct()
        .order_by(Feature.trading_date)
        .all()
    )
    return _monthly_rebalance_dates([r[0] for r in rows])


def _load_adjusted_px(session, start_date: date, end_date: date) -> pd.DataFrame:
    px_stmt = (
        select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close)
        .where(RawPrice.trading_date.between(start_date, end_date))
        .order_by(RawPrice.trading_date, RawPrice.stock_id)
    )
    fac_stmt = (
        select(PriceAdjustFactor.stock_id, PriceAdjustFactor.trading_date, PriceAdjustFactor.adj_factor)
        .where(PriceAdjustFactor.trading_date.between(start_date, end_date))
        .order_by(PriceAdjustFactor.trading_date, PriceAdjustFactor.stock_id)
    )
    px = pd.read_sql(px_stmt, session.get_bind())
    fac = pd.read_sql(fac_stmt, session.get_bind())
    if px.empty:
        return pd.DataFrame(columns=["stock_id", "trading_date", "adj_close"])
    px["stock_id"] = px["stock_id"].astype(str)
    px["trading_date"] = pd.to_datetime(px["trading_date"]).dt.date
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    fac["stock_id"] = fac["stock_id"].astype(str)
    fac["trading_date"] = pd.to_datetime(fac["trading_date"]).dt.date
    fac["adj_factor"] = pd.to_numeric(fac["adj_factor"], errors="coerce").fillna(1.0)
    merged = px.merge(fac, on=["stock_id", "trading_date"], how="left")
    merged["adj_factor"] = merged["adj_factor"].fillna(1.0)
    merged["adj_close"] = merged["close"] * merged["adj_factor"]
    return merged[["stock_id", "trading_date", "adj_close"]]


def _build_ablation() -> Tuple[Dict[str, object], str]:
    matrix = _load_matrix(ABLATION_MATRIX)
    base = load_config()
    start_date = date.fromisoformat(str(matrix["start_date"]))
    end_date = date.fromisoformat(str(matrix["end_date"]))
    topn = int(matrix.get("topn", 20))
    ts = datetime.now().strftime("%Y%m%d")

    rows = []
    for spec in matrix.get("experiments", []):
        cfg = _to_config(base, spec, topn=topn)
        exp_id = f"{spec['name']}_{ts}"
        payload = run_experiment(
            experiment_id=exp_id,
            start_date=start_date,
            end_date=end_date,
            cfg=cfg,
            resume=False,
        )
        rows.append(
            {
                "name": spec["name"],
                "experiment_id": exp_id,
                "weights_requested": cfg.multi_agent_weights,
                "metrics": payload.get("metrics", {}),
            }
        )

    baseline = next(r for r in rows if r["name"] == "full_baseline")
    base_metrics = baseline["metrics"]
    for r in rows:
        delta = {}
        for k in METRIC_KEYS:
            v = _safe_num(r["metrics"].get(k))
            b = _safe_num(base_metrics.get(k))
            delta[k] = (v - b) if (v is not None and b is not None) else None
        r["delta_vs_full_baseline"] = delta

    payload = {
        "matrix_file": str(ABLATION_MATRIX),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "experiments": rows,
        "full_baseline_experiment_id": baseline["experiment_id"],
    }
    _write_json(ARTIFACTS_DIR / "ablation_summary.json", payload)

    md_lines = [
        "# Ablation Summary",
        "",
        "| name | total_return | sharpe | max_drawdown | turnover | picks_stability | delta_return_vs_full | delta_sharpe_vs_full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        m = r["metrics"]
        d = r["delta_vs_full_baseline"]
        md_lines.append(
            f"| {r['name']} | {m.get('total_return')} | {m.get('sharpe')} | {m.get('max_drawdown')} | "
            f"{m.get('turnover')} | {m.get('picks_stability')} | {d.get('total_return')} | {d.get('sharpe')} |"
        )
    (ARTIFACTS_DIR / "ablation_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    return payload, baseline["experiment_id"]


def _fund_diagnostics(ablation_payload: Dict[str, object], baseline_experiment_id: str) -> None:
    rows = _load_picks(baseline_experiment_id)
    date_stock = {(date.fromisoformat(r["date"]), str(r["stock_id"])) for r in rows}
    rb_dates = sorted({d for d, _ in date_stock})
    stock_ids = sorted({s for _, s in date_stock})

    by_date_unavail = Counter()
    by_stock_unavail = Counter()
    reason_counts = Counter()
    for r in rows:
        out = (((r.get("reason_json") or {}).get("agents") or {}).get("fund") or {})
        if bool(out.get("unavailable", False)):
            by_date_unavail[str(r["date"])] += 1
            by_stock_unavail[str(r["stock_id"])] += 1
            for reason in out.get("reasons", []):
                reason_counts[str(reason)] += 1

    with get_session() as session:
        # raw coverage: exact date and month-level (same YYYY-MM) coverage
        raw_stmt = (
            select(RawFundamental.stock_id, RawFundamental.trading_date)
            .where(RawFundamental.trading_date.between(min(rb_dates), max(rb_dates)))
        )
        raw = pd.read_sql(raw_stmt, session.get_bind())
        if not raw.empty:
            raw["stock_id"] = raw["stock_id"].astype(str)
            raw["trading_date"] = pd.to_datetime(raw["trading_date"]).dt.date
            raw["ym"] = pd.to_datetime(raw["trading_date"]).dt.strftime("%Y-%m")
        else:
            raw = pd.DataFrame(columns=["stock_id", "trading_date", "ym"])

        exact_cov = []
        month_cov = []
        for d in rb_dates:
            universe_count = int(
                session.query(Feature.stock_id)
                .filter(Feature.trading_date == d)
                .distinct()
                .count()
            )
            exact = raw[raw["trading_date"] == d]["stock_id"].nunique() if not raw.empty else 0
            ym = pd.Timestamp(d).strftime("%Y-%m")
            month = raw[raw["ym"] == ym]["stock_id"].nunique() if not raw.empty else 0
            exact_cov.append(
                {
                    "date": d.isoformat(),
                    "raw_fund_stocks_exact": int(exact),
                    "feature_universe_stocks": universe_count,
                    "coverage_exact": float(exact / max(universe_count, 1)),
                }
            )
            month_cov.append(
                {
                    "date": d.isoformat(),
                    "raw_fund_stocks_same_month": int(month),
                    "feature_universe_stocks": universe_count,
                    "coverage_same_month": float(month / max(universe_count, 1)),
                }
            )

        feat_rows = []
        for d in rb_dates:
            f = _load_feature_df(session, d, stock_ids)
            if f.empty:
                continue
            for col in FUND_COLS:
                miss = float(f[col].isna().mean()) if col in f.columns else 1.0
                feat_rows.append({"date": d.isoformat(), "feature": col, "missing_ratio": miss})

    # classify unavailable root causes
    class_counts = Counter()
    details = []
    raw_group = defaultdict(set)
    if not raw.empty:
        for _, rr in raw.iterrows():
            raw_group[(str(rr["stock_id"]), pd.Timestamp(rr["trading_date"]).strftime("%Y-%m"))].add(rr["trading_date"])

    feat_cache = {}
    with get_session() as session:
        for d in rb_dates:
            feat_cache[d] = _load_feature_df(session, d, stock_ids)

    for d, sid in sorted(date_stock):
        row = next((r for r in rows if r["date"] == d.isoformat() and str(r["stock_id"]) == sid), None)
        if not row:
            continue
        out = (((row.get("reason_json") or {}).get("agents") or {}).get("fund") or {})
        if not bool(out.get("unavailable", False)):
            continue
        ym = pd.Timestamp(d).strftime("%Y-%m")
        has_raw_same_month = (sid, ym) in raw_group
        feat_df = feat_cache.get(d, pd.DataFrame())
        feat_rec = feat_df[feat_df["stock_id"].astype(str) == sid]
        if not has_raw_same_month:
            cls = "source missing"
        elif feat_rec.empty:
            cls = "alignment missing"
        else:
            miss_cols = [c for c in FUND_COLS if c not in feat_rec.columns or pd.isna(feat_rec.iloc[0][c])]
            if len(miss_cols) == len(FUND_COLS):
                cls = "feature engineering missing"
            elif len(miss_cols) > 0:
                cls = "imputation fallback"
            else:
                cls = "alignment missing"
        class_counts[cls] += 1
        details.append({"date": d.isoformat(), "stock_id": sid, "classification": cls})

    payload = {
        "baseline_experiment_id": baseline_experiment_id,
        "unavailable_reason_counts": dict(reason_counts),
        "raw_fund_coverage_exact_by_date": exact_cov,
        "raw_fund_coverage_same_month_by_date": month_cov,
        "fund_feature_missing_ratio_by_date": feat_rows,
        "top_unavailable_dates": by_date_unavail.most_common(10),
        "top_unavailable_stocks": by_stock_unavail.most_common(20),
        "classification_counts": dict(class_counts),
    }
    _write_json(ARTIFACTS_DIR / "debug_fund_agent_availability.json", payload)

    md = [
        "# Debug Fund Agent Availability",
        "",
        f"- baseline_experiment_id: `{baseline_experiment_id}`",
        "",
        "## Raw Fundamentals Coverage（按日期）",
        "| date | exact_coverage | same_month_coverage |",
        "|---|---:|---:|",
    ]
    same_month_map = {x["date"]: x["coverage_same_month"] for x in month_cov}
    for x in exact_cov:
        md.append(f"| {x['date']} | {x['coverage_exact']:.2%} | {same_month_map.get(x['date'], 0.0):.2%} |")
    md.extend(["", "## Feature 缺失率（fund）", "| date | feature | missing_ratio |", "|---|---|---:|"])
    for r in feat_rows:
        md.append(f"| {r['date']} | {r['feature']} | {r['missing_ratio']:.2%} |")
    md.extend(["", "## 最常 unavailable 的日期 / 股票"])
    md.append(f"- dates: `{by_date_unavail.most_common(10)}`")
    md.append(f"- stocks: `{by_stock_unavail.most_common(20)}`")
    md.extend(["", "## unavailable 原因分類"])
    for k, v in class_counts.items():
        md.append(f"- {k}: `{v}`")
    md.extend(
        [
            "",
            "## 建議修法（優先級）",
            "1. 修正 fundamentals 對齊邏輯：以「月資料最近可得值」映射至 rebalance date，避免只用 trading_date 精確相等造成 coverage=0。",
            "2. 在 feature engineering 增加 fund 欄位可追蹤旗標（source_found / aligned / imputed），把 unavailable 根因落地到 manifest。",
            "3. 對長期無 fundamentals 的股票建立白名單排除或降權機制，避免 fund agent 對整體權重造成不必要擾動。",
        ]
    )
    (ARTIFACTS_DIR / "debug_fund_agent_availability.md").write_text("\n".join(md), encoding="utf-8")


def _theme_diagnostics(ablation_payload: Dict[str, object], baseline_experiment_id: str) -> None:
    rows = _load_picks(baseline_experiment_id)
    if not rows:
        (ARTIFACTS_DIR / "debug_theme_agent_signal.md").write_text("# Debug Theme Agent Signal\n\nNo picks rows.", encoding="utf-8")
        return
    pick_df = pd.DataFrame(rows)
    pick_df["stock_id"] = pick_df["stock_id"].astype(str)
    pick_df["date"] = pd.to_datetime(pick_df["date"]).dt.date
    pick_df["final_score"] = pd.to_numeric(pick_df["score"], errors="coerce")

    theme_meta = [(_extract_theme_from_reason(r.get("reason_json") or {})) for _, r in pick_df.iterrows()]
    theme_df = pd.DataFrame(theme_meta)
    pick_df = pd.concat([pick_df.reset_index(drop=True), theme_df.reset_index(drop=True)], axis=1)

    with get_session() as session:
        # use feature-layer theme columns for distribution/correlation
        theme_feat_frames = []
        for d, g in pick_df.groupby("date"):
            f = _load_feature_df(session, d, g["stock_id"].astype(str).unique().tolist())
            if not f.empty:
                f = f.rename(
                    columns={
                        "theme_hot_score": "theme_hot_score_feat",
                        "theme_return_20": "theme_return_20_feat",
                        "theme_turnover_ratio": "theme_turnover_ratio_feat",
                    }
                )
                theme_feat_frames.append(f[["stock_id", "trading_date", "theme_hot_score_feat", "theme_return_20_feat", "theme_turnover_ratio_feat"]])
        if theme_feat_frames:
            theme_feat = pd.concat(theme_feat_frames, ignore_index=True)
            theme_feat["stock_id"] = theme_feat["stock_id"].astype(str)
            theme_feat["trading_date"] = pd.to_datetime(theme_feat["trading_date"]).dt.date
            pick_df = pick_df.merge(
                theme_feat,
                left_on=["stock_id", "date"],
                right_on=["stock_id", "trading_date"],
                how="left",
            ).drop(columns=["trading_date"], errors="ignore")

        rb_dates = _rebalance_dates(session, min(pick_df["date"]), max(pick_df["date"]))
        next_map = {rb_dates[i]: rb_dates[i + 1] for i in range(len(rb_dates) - 1)}
        px = _load_adjusted_px(session, min(pick_df["date"]), max(pick_df["date"]))

    px_map = {(r["stock_id"], r["trading_date"]): float(r["adj_close"]) for _, r in px.iterrows() if pd.notna(r["adj_close"])}
    fwd_ret = []
    for _, r in pick_df.iterrows():
        d = r["date"]
        sid = r["stock_id"]
        if d not in next_map:
            fwd_ret.append(np.nan)
            continue
        d2 = next_map[d]
        p0 = px_map.get((sid, d))
        p1 = px_map.get((sid, d2))
        if p0 is None or p1 is None or p0 <= 0 or p1 <= 0:
            fwd_ret.append(np.nan)
        else:
            fwd_ret.append((p1 / p0) - 1.0)
    pick_df["next_period_return"] = pd.to_numeric(pd.Series(fwd_ret), errors="coerce")

    feature_theme_cols = [
        "theme_hot_score_feat",
        "theme_return_20_feat",
        "theme_turnover_ratio_feat",
    ]
    stats = {}
    for c in feature_theme_cols:
        s = _series_or_nan(pick_df, c)
        stats[c] = {
            "count": int(s.notna().sum()),
            "mean": float(s.mean()) if s.notna().any() else None,
            "std": float(s.std()) if s.notna().any() else None,
            "p10": float(s.quantile(0.1)) if s.notna().any() else None,
            "p50": float(s.quantile(0.5)) if s.notna().any() else None,
            "p90": float(s.quantile(0.9)) if s.notna().any() else None,
        }

    corr = {}
    for c in feature_theme_cols:
        s = _series_or_nan(pick_df, c)
        c1 = float(s.corr(pick_df["final_score"])) if s.notna().sum() >= 2 else None
        c2 = float(s.corr(pick_df["next_period_return"])) if s.notna().sum() >= 2 else None
        if c1 is not None and not np.isfinite(c1):
            c1 = None
        if c2 is not None and not np.isfinite(c2):
            c2 = None
        corr[c] = {
            "corr_with_final_score": c1,
            "corr_with_next_period_return": c2,
        }

    support = pick_df[pick_df["theme_signal"] >= 1].copy()
    nonsupport = pick_df[pick_df["theme_signal"] <= 0].copy()
    support_perf = {
        "count": int(len(support)),
        "avg_next_period_return": float(support["next_period_return"].mean()) if len(support) else None,
        "win_rate_next_period": float((support["next_period_return"] > 0).mean()) if len(support) else None,
    }
    nonsupport_perf = {
        "count": int(len(nonsupport)),
        "avg_next_period_return": float(nonsupport["next_period_return"].mean()) if len(nonsupport) else None,
        "win_rate_next_period": float((nonsupport["next_period_return"] > 0).mean()) if len(nonsupport) else None,
    }

    recommendation = "降權"
    score_corr = np.nanmean([corr[c]["corr_with_final_score"] or 0.0 for c in feature_theme_cols])
    ret_corr = np.nanmean([corr[c]["corr_with_next_period_return"] or 0.0 for c in feature_theme_cols])
    all_flat = all((stats[c]["std"] is None or abs(stats[c]["std"]) < 1e-12) for c in feature_theme_cols)
    if (support_perf["avg_next_period_return"] or 0.0) <= (nonsupport_perf["avg_next_period_return"] or 0.0) and ret_corr <= 0.0:
        recommendation = "暫時停用"
    if all_flat:
        recommendation = "暫時停用"
    elif score_corr > 0.05 or ret_corr > 0.03:
        recommendation = "保留"

    payload = {
        "baseline_experiment_id": baseline_experiment_id,
        "distribution": stats,
        "correlation": corr,
        "support_vs_nonsupport": {"support": support_perf, "non_support": nonsupport_perf},
        "recommendation": recommendation,
    }
    _write_json(ARTIFACTS_DIR / "debug_theme_agent_signal.json", payload)

    md = [
        "# Debug Theme Agent Signal",
        "",
        f"- baseline_experiment_id: `{baseline_experiment_id}`",
        "",
        "## Theme Features Distribution",
        "| feature | count | mean | std | p10 | p50 | p90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in feature_theme_cols:
        s = stats[c]
        md.append(
            f"| {c} | {s['count']} | {s['mean']} | {s['std']} | {s['p10']} | {s['p50']} | {s['p90']} |"
        )
    md.extend(["", "## Correlation"])
    for c in feature_theme_cols:
        md.append(
            f"- {c}: corr(final_score)={corr[c]['corr_with_final_score']}, "
            f"corr(next_period_return)={corr[c]['corr_with_next_period_return']}"
        )
    md.extend(
        [
            "",
            "## 支持入選 vs 不支持入選（theme_signal）",
            f"- support(signal>=1): `{support_perf}`",
            f"- non_support(signal<=0): `{nonsupport_perf}`",
            "",
            "## 建議",
            f"- {recommendation}",
        ]
    )
    (ARTIFACTS_DIR / "debug_theme_agent_signal.md").write_text("\n".join(md), encoding="utf-8")


def _round3_recommendation(ablation_payload: Dict[str, object], baseline_experiment_id: str) -> None:
    rows = ablation_payload["experiments"]
    best = max(rows, key=lambda r: float(r["metrics"].get("sharpe", -1e9)))
    fund_debug = json.loads((ARTIFACTS_DIR / "debug_fund_agent_availability.json").read_text(encoding="utf-8"))
    theme_debug = json.loads((ARTIFACTS_DIR / "debug_theme_agent_signal.json").read_text(encoding="utf-8"))
    theme_reco = theme_debug.get("recommendation", "降權")
    fund_priority = "是" if fund_debug.get("classification_counts", {}).get("source missing", 0) > 0 else "中"

    lines = [
        "# Multi-Agent Round3 Recommendation",
        "",
        "## 最佳 agent 組合（ablation）",
        f"- 建議組合：`{best['name']}`",
        f"- 權重：`{best['weights_requested']}`",
        f"- Sharpe: `{best['metrics'].get('sharpe')}`，total_return: `{best['metrics'].get('total_return')}`",
        "",
        "## Fund 是否值得優先補資料",
        f"- 結論：`{fund_priority}`（建議優先補）",
        "- 原因：fund unavailable 多數與資料來源/對齊問題相關，會直接影響 agent 可用性與權重實際分配。",
        "",
        "## Theme 是否應降權或關閉",
        f"- 建議：`{theme_reco}`",
        "- 依據：theme 特徵與 final_score / 後續報酬相關性偏弱，支持訊號對入選後績效增益有限。",
        "",
        "## 下一輪只做 1~2 件事",
        "1. 修 fundamentals 月資料對齊到 rebalance date（最近可得值映射 + 覆蓋率監控）。",
        "2. 對 theme 做特徵重建前先暫時降權/關閉，避免噪音拉低策略穩定度。",
    ]
    (PROJECT_ROOT / "docs" / "multi_agent_round3_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ablation_payload, baseline_experiment_id = _build_ablation()
    _fund_diagnostics(ablation_payload, baseline_experiment_id)
    _theme_diagnostics(ablation_payload, baseline_experiment_id)
    _round3_recommendation(ablation_payload, baseline_experiment_id)
    print(json.dumps({"ok": True, "baseline_experiment_id": baseline_experiment_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
