#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

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


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_picks_map(experiment_id: str) -> Dict[str, List[str]]:
    path = ARTIFACTS_DIR / f"experiment_picks_{experiment_id}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {}
    for r in rows:
        out.setdefault(str(r["date"]), []).append(str(r["stock_id"]))
    return {k: sorted(v) for k, v in out.items()}


def _load_amt20_for_date(date_str: str, stock_ids: List[str]) -> Dict[str, float]:
    if not stock_ids:
        return {}
    d = datetime.fromisoformat(date_str).date()
    with get_session() as session:
        stmt = (
            select(Feature.stock_id, Feature.features_json)
            .where(Feature.trading_date == d)
            .where(Feature.stock_id.in_(stock_ids))
        )
        df = pd.read_sql(stmt, session.get_bind())
    if df.empty:
        return {}
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        sid = str(r["stock_id"])
        feat = r.get("features_json") or {}
        amt20 = feat.get("amt_20")
        try:
            v = float(amt20)
            if np.isfinite(v):
                out[sid] = v
        except Exception:
            continue
    return out


def _load_liquidity_with_fallback(date_str: str, stock_ids: List[str]) -> tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    if not stock_ids:
        return {}, {"feature": 0, "raw_fallback": 0, "missing": 0}
    d = datetime.fromisoformat(date_str).date()
    with get_session() as session:
        feat_stmt = (
            select(Feature.stock_id, Feature.features_json)
            .where(Feature.trading_date == d)
            .where(Feature.stock_id.in_(stock_ids))
        )
        feat_df = pd.read_sql(feat_stmt, session.get_bind())
        raw_stmt = (
            select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close, RawPrice.volume)
            .where(RawPrice.stock_id.in_(stock_ids))
            .where(RawPrice.trading_date.between(d - timedelta(days=60), d))
            .order_by(RawPrice.stock_id, RawPrice.trading_date)
        )
        raw_df = pd.read_sql(raw_stmt, session.get_bind())

    feat_map = {}
    if not feat_df.empty:
        for _, r in feat_df.iterrows():
            sid = str(r["stock_id"])
            f = r.get("features_json") or {}
            vals = {}
            for k in ["amt", "amt_20", "amt_ratio_20"]:
                try:
                    v = float(f.get(k))
                    if np.isfinite(v):
                        vals[k] = v
                except Exception:
                    pass
            feat_map[sid] = vals

    raw_map: Dict[tuple[str, str], Dict[str, float]] = {}
    if not raw_df.empty:
        raw_df["stock_id"] = raw_df["stock_id"].astype(str)
        raw_df["trading_date"] = pd.to_datetime(raw_df["trading_date"]).dt.date
        raw_df["close"] = pd.to_numeric(raw_df["close"], errors="coerce")
        raw_df["volume"] = pd.to_numeric(raw_df["volume"], errors="coerce")
        raw_df["amt"] = raw_df["close"] * raw_df["volume"]
        raw_df = raw_df.sort_values(["stock_id", "trading_date"])
        raw_df["amt_20"] = raw_df.groupby("stock_id")["amt"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        raw_df["amt_ratio_20"] = raw_df["amt"] / raw_df["amt_20"].replace(0, np.nan)
        for _, r in raw_df.iterrows():
            key = (str(r["stock_id"]), str(r["trading_date"]))
            vals = {}
            for k in ["amt", "amt_20", "amt_ratio_20"]:
                v = r.get(k)
                if pd.notna(v):
                    vals[k] = float(v)
            raw_map[key] = vals

    out: Dict[str, Dict[str, float]] = {}
    source_count = {"feature": 0, "raw_fallback": 0, "missing": 0}
    d_key = str(d)
    for sid in stock_ids:
        fvals = feat_map.get(sid, {})
        if all(k in fvals for k in ["amt", "amt_20", "amt_ratio_20"]):
            out[sid] = fvals
            source_count["feature"] += 1
            continue
        rvals = raw_map.get((sid, d_key), {})
        if all(k in rvals for k in ["amt", "amt_20", "amt_ratio_20"]):
            out[sid] = rvals
            source_count["raw_fallback"] += 1
            continue
        source_count["missing"] += 1
    return out, source_count


def _window_breakdown(window_key: str, w: Dict[str, object]) -> Dict[str, object]:
    def _safe_mean(values: List[float | None]) -> float | None:
        arr = [float(v) for v in values if v is not None and np.isfinite(float(v))]
        return float(np.mean(arr)) if arr else None

    if not w.get("available"):
        return {"window": window_key, "available": False}
    model_metrics = w.get("model_metrics", {})
    ma_metrics = w.get("multi_agent_metrics", {})
    per = w.get("per_period", [])
    model_id = w.get("model_experiment_id")
    ma_id = w.get("multi_agent_experiment_id")
    model_map = _load_picks_map(model_id)
    ma_map = _load_picks_map(ma_id)

    turnover_rows = []
    overlap_rows = []
    liquidity_rows = []
    switching_rows = []
    for p in per:
        d = p["entry_date"]
        mset = set(model_map.get(d, []))
        aset = set(ma_map.get(d, []))
        inter = len(mset & aset)
        union = len(mset | aset)
        topn = max(len(mset), len(aset), 1)
        overlap_j = inter / float(max(union, 1))
        replace_count = topn - inter
        replace_ratio = replace_count / float(topn)

        liq_map, liq_source = _load_liquidity_with_fallback(d, sorted(aset))
        amt_vals = [liq_map[s]["amt_20"] for s in aset if s in liq_map and "amt_20" in liq_map[s]]
        avg_amt20 = float(np.mean(amt_vals)) if amt_vals else None
        p20_amt20 = float(np.percentile(amt_vals, 20)) if amt_vals else None
        low_liq_ratio = float(np.mean([1.0 if v < 5e7 else 0.0 for v in amt_vals])) if amt_vals else None

        turnover_rows.append(
            {
                "date": d,
                "model_turnover_window": model_metrics.get("turnover"),
                "multi_agent_turnover_window": ma_metrics.get("turnover"),
            }
        )
        overlap_rows.append({"date": d, "overlap_jaccard": overlap_j})
        liquidity_rows.append(
            {
                "date": d,
                "avg_amt_20": avg_amt20,
                "p20_amt_20": p20_amt20,
                "low_liquidity_ratio": low_liq_ratio,
                "liquidity_source_count": liq_source,
            }
        )
        switching_rows.append(
            {
                "date": d,
                "replace_count": int(replace_count),
                "replace_ratio": replace_ratio,
            }
        )

    low_overlap_dates = [r["date"] for r in overlap_rows if r["overlap_jaccard"] < 0.1]
    return {
        "window": window_key,
        "available": True,
        "period": w.get("period", {}),
        "summary": {
            "model_turnover": model_metrics.get("turnover"),
            "multi_agent_turnover": ma_metrics.get("turnover"),
            "avg_overlap_jaccard": float(np.mean([r["overlap_jaccard"] for r in overlap_rows])) if overlap_rows else 0.0,
            "avg_replace_count": float(np.mean([r["replace_count"] for r in switching_rows])) if switching_rows else 0.0,
            "avg_replace_ratio": float(np.mean([r["replace_ratio"] for r in switching_rows])) if switching_rows else 0.0,
            "avg_amt_20": _safe_mean([r["avg_amt_20"] for r in liquidity_rows]),
            "avg_low_liquidity_ratio": _safe_mean([r["low_liquidity_ratio"] for r in liquidity_rows]),
        },
        "turnover_rows": turnover_rows,
        "overlap_rows": overlap_rows,
        "low_overlap_dates": low_overlap_dates,
        "liquidity_rows": liquidity_rows,
        "switching_rows": switching_rows,
    }


def analyze_tradability_gap() -> Dict[str, object]:
    shadow = _load_json(ARTIFACTS_DIR / "shadow_monitor_latest.json")
    windows = shadow.get("windows", {})
    details = {k: _window_breakdown(k, windows.get(k, {})) for k in WINDOW_KEYS}

    # 主因排序（依 6m，若無則 3m）
    base = details["6m"] if details["6m"].get("available") else details["3m"]
    root_causes: List[Tuple[str, float]] = []
    if base.get("available"):
        s = base["summary"]
        overlap = s.get("avg_overlap_jaccard", 0.0) or 0.0
        turnover_gap = (s.get("multi_agent_turnover") or 0.0) - (s.get("model_turnover") or 0.0)
        low_liq = s.get("avg_low_liquidity_ratio") or 0.0
        switch = s.get("avg_replace_ratio") or 0.0
        root_causes = [
            ("overlap過低（策略分歧）", 1.0 - overlap),
            ("switching風險偏高（持股替換率）", switch),
            ("turnover偏高", max(turnover_gap, 0.0)),
            ("流動性偏弱", low_liq),
        ]
        root_causes.sort(key=lambda x: x[1], reverse=True)

    governance_items = {"governance": ["overlap過低（策略分歧）", "switching風險偏高（持股替換率）"], "strategy": ["turnover偏高", "流動性偏弱"]}
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "windows": details,
        "root_cause_ranking": [{"reason": k, "score": float(v)} for k, v in root_causes[:3]],
        "classification": governance_items,
    }
    (ARTIFACTS_DIR / "tradability_gap_analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Tradability Gap Analysis",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        "",
        "## 1) Turnover Breakdown",
    ]
    for k in WINDOW_KEYS:
        d = details[k]
        if not d.get("available"):
            lines.append(f"- {k}: unavailable")
            continue
        s = d["summary"]
        lines.append(f"- {k}: model={s['model_turnover']}, multi_agent={s['multi_agent_turnover']}")
    lines.extend(["", "## 2) Overlap Breakdown"])
    for k in WINDOW_KEYS:
        d = details[k]
        if not d.get("available"):
            continue
        lines.append(f"- {k}: avg_overlap={d['summary']['avg_overlap_jaccard']:.4f}, low_overlap_dates={d['low_overlap_dates']}")
    lines.extend(["", "## 3) Liquidity Breakdown"])
    for k in WINDOW_KEYS:
        d = details[k]
        if not d.get("available"):
            continue
        s = d["summary"]
        lines.append(f"- {k}: avg_amt_20={s['avg_amt_20']}, avg_low_liquidity_ratio={s['avg_low_liquidity_ratio']}")
    lines.extend(["", "## 4) Switching Risk"])
    for k in WINDOW_KEYS:
        d = details[k]
        if not d.get("available"):
            continue
        s = d["summary"]
        lines.append(f"- {k}: avg_replace_count={s['avg_replace_count']}, avg_replace_ratio={s['avg_replace_ratio']:.4f}")
    lines.extend(["", "## 5) 結論（主因排序）"])
    for i, rc in enumerate(payload["root_cause_ranking"], start=1):
        lines.append(f"{i}. {rc['reason']} (score={rc['score']:.4f})")
    lines.extend(
        [
            "",
            "- 策略問題：`turnover偏高`、`流動性偏弱`",
            "- 治理問題：`overlap過低`、`switching風險偏高`",
        ]
    )
    (ARTIFACTS_DIR / "tradability_gap_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    payload = analyze_tradability_gap()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
