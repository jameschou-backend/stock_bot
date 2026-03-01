#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import get_session
from app.models import Feature, RawPrice

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
WINDOW_KEYS = ["1m", "3m", "6m"]
LIQ_FIELDS = ["amt", "amt_20", "amt_ratio_20"]


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_picks_map(experiment_id: str) -> Dict[str, List[str]]:
    rows = _load_json(ARTIFACTS_DIR / f"experiment_picks_{experiment_id}.json")
    out: Dict[str, List[str]] = {}
    for r in rows:
        out.setdefault(str(r["date"]), []).append(str(r["stock_id"]))
    return {k: sorted(set(v)) for k, v in out.items()}


def _as_finite(v) -> float | None:
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _build_raw_feature_map(stock_ids: List[str], target_dates: List[datetime.date]) -> Dict[Tuple[str, str], Dict[str, float]]:
    if not stock_ids or not target_dates:
        return {}
    min_date = min(target_dates) - timedelta(days=60)
    max_date = max(target_dates)
    with get_session() as session:
        stmt = (
            select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close, RawPrice.volume)
            .where(RawPrice.stock_id.in_(stock_ids))
            .where(RawPrice.trading_date.between(min_date, max_date))
            .order_by(RawPrice.stock_id, RawPrice.trading_date)
        )
        df = pd.read_sql(stmt, session.get_bind())
    if df.empty:
        return {}
    df["stock_id"] = df["stock_id"].astype(str)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["amt"] = df["close"] * df["volume"]
    df = df.sort_values(["stock_id", "trading_date"])
    df["amt_20"] = df.groupby("stock_id")["amt"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    df["amt_ratio_20"] = df["amt"] / df["amt_20"].replace(0, np.nan)
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for _, row in df.iterrows():
        vals = {}
        for f in LIQ_FIELDS:
            x = row.get(f)
            if pd.notna(x):
                vals[f] = float(x)
        out[(str(row["stock_id"]), str(row["trading_date"]))] = vals
    return out


def analyze_liquidity_observability() -> Dict[str, object]:
    shadow = _load_json(ARTIFACTS_DIR / "shadow_monitor_latest.json")
    windows = shadow.get("windows", {})

    points: List[Tuple[str, str, str]] = []
    for wk in WINDOW_KEYS:
        w = windows.get(wk, {})
        if not w.get("available"):
            continue
        picks_map = _load_picks_map(str(w.get("multi_agent_experiment_id")))
        for d, sids in picks_map.items():
            for sid in sids:
                points.append((wk, d, sid))

    uniq_dates = sorted({datetime.fromisoformat(d).date() for _, d, _ in points})
    uniq_stocks = sorted({sid for _, _, sid in points})
    date_str_set = {str(d) for d in uniq_dates}

    feat_map: Dict[Tuple[str, str], Dict[str, float]] = {}
    if uniq_dates and uniq_stocks:
        with get_session() as session:
            stmt = (
                select(Feature.trading_date, Feature.stock_id, Feature.features_json)
                .where(Feature.trading_date.in_(uniq_dates))
                .where(Feature.stock_id.in_(uniq_stocks))
            )
            feat_df = pd.read_sql(stmt, session.get_bind())
        for _, row in feat_df.iterrows():
            d = str(pd.to_datetime(row["trading_date"]).date())
            sid = str(row["stock_id"])
            feat = row.get("features_json") or {}
            vals = {}
            for f in LIQ_FIELDS:
                x = _as_finite(feat.get(f))
                if x is not None:
                    vals[f] = x
            feat_map[(sid, d)] = vals

    raw_map = _build_raw_feature_map(uniq_stocks, uniq_dates)

    has_raw_by_sid: Dict[str, bool] = {sid: False for sid in uniq_stocks}
    has_raw_by_sid_near: Dict[Tuple[str, str], bool] = {}
    for sid in uniq_stocks:
        sid_dates = sorted([datetime.fromisoformat(d).date() for s, d in raw_map.keys() if s == sid])
        has_raw_by_sid[sid] = bool(sid_dates)
        for d in date_str_set:
            target = datetime.fromisoformat(d).date()
            near = any(abs((target - x).days) <= 3 for x in sid_dates)
            has_raw_by_sid_near[(sid, d)] = near

    per_window = {}
    reason_counts = {"source": 0, "pipeline": 0, "alignment": 0}
    blind_spot_counts = {"pipeline": 0}
    field_missing = {f: 0 for f in LIQ_FIELDS}
    field_total = {f: 0 for f in LIQ_FIELDS}
    feature_field_present = {f: 0 for f in LIQ_FIELDS}
    for wk in WINDOW_KEYS:
        wk_points = [(d, sid) for w, d, sid in points if w == wk]
        if not wk_points:
            per_window[wk] = {"available": False}
            continue
        wk_total = len(wk_points)
        wk_field_ok = {f: 0 for f in LIQ_FIELDS}
        for d, sid in wk_points:
            fvals = feat_map.get((sid, d), {})
            rvals = raw_map.get((sid, d), {})
            for f in LIQ_FIELDS:
                field_total[f] += 1
                if f in fvals:
                    feature_field_present[f] += 1
                if f in fvals or f in rvals:
                    if f not in fvals and f in rvals:
                        blind_spot_counts["pipeline"] += 1
                    wk_field_ok[f] += 1
                else:
                    field_missing[f] += 1
                    has_near = has_raw_by_sid_near.get((sid, d), False)
                    has_any = has_raw_by_sid.get(sid, False)
                    if has_near and (sid, d) not in raw_map:
                        reason_counts["alignment"] += 1
                    elif has_any:
                        reason_counts["pipeline"] += 1
                    else:
                        reason_counts["source"] += 1
        per_window[wk] = {
            "available": True,
            "points": wk_total,
            "availability_rate": {f: (wk_field_ok[f] / wk_total if wk_total else 0.0) for f in LIQ_FIELDS},
        }

    overall_availability = {
        f: (1.0 - (field_missing[f] / field_total[f])) if field_total[f] else 0.0 for f in LIQ_FIELDS
    }
    feature_layer_coverage = {f: (feature_field_present[f] / field_total[f] if field_total[f] else 0.0) for f in LIQ_FIELDS}
    support_d3 = bool(
        per_window.get("6m", {}).get("available")
        and per_window["6m"]["availability_rate"]["amt_20"] >= 0.90
        and per_window["6m"]["availability_rate"]["amt_ratio_20"] >= 0.90
    )
    classification = []
    for k in ["source", "pipeline", "alignment"]:
        if reason_counts[k] > 0:
            classification.append({"category": k, "count": int(reason_counts[k])})

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "liquidity_fields": LIQ_FIELDS,
        "overall_availability": overall_availability,
        "feature_layer_coverage": feature_layer_coverage,
        "window_availability": per_window,
        "missing_reason_counts": reason_counts,
        "blind_spot_reason_counts": blind_spot_counts,
        "missing_reason_classification": classification,
        "sufficient_for_d3_liquidity": support_d3,
        "conclusion": (
            "D3_liquidity 可有效評估" if support_d3 else "D3_liquidity 仍有觀測盲區，需先補齊資料可用率"
        ),
    }
    (ARTIFACTS_DIR / "liquidity_observability_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Liquidity Observability Report",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- sufficient_for_d3_liquidity: `{support_d3}`",
        "",
        "## 缺失欄位與可用率",
    ]
    for f in LIQ_FIELDS:
        lines.append(
            f"- {f}: overall_availability={overall_availability[f]:.4f}, feature_layer_coverage={feature_layer_coverage[f]:.4f}"
        )
    lines.extend(["", "## 缺失原因分類（source / pipeline / alignment）"])
    for c in classification:
        lines.append(f"- {c['category']}: {c['count']}")
    if not classification:
        lines.append("- 無缺失")
    lines.append(f"- pipeline blind spot (feature 缺欄但 raw 可回補): {blind_spot_counts['pipeline']}")
    lines.extend(["", "## 1m / 3m / 6m 可用率"])
    for wk in WINDOW_KEYS:
        w = per_window.get(wk, {})
        if not w.get("available"):
            lines.append(f"- {wk}: unavailable")
            continue
        rates = w["availability_rate"]
        lines.append(
            f"- {wk}: amt={rates['amt']:.4f}, amt_20={rates['amt_20']:.4f}, amt_ratio_20={rates['amt_ratio_20']:.4f}"
        )
    lines.extend(["", "## D3 判斷支撐性", f"- 結論：{payload['conclusion']}"])
    (ARTIFACTS_DIR / "liquidity_observability_report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    payload = analyze_liquidity_observability()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
