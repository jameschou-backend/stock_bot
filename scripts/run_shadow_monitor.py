#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from app.models import Feature
from scripts.analyze_market_regime import _build_market_regime
from scripts.evaluate_experiment import (
    ARTIFACTS_DIR,
    _load_adjust_factors,
    _load_price_frame,
    _merge_adjusted_close,
    _monthly_rebalance_dates,
    _period_return,
    _to_config,
    run_experiment,
)


def _load_best_weights() -> Dict[str, float]:
    data = json.loads((ARTIFACTS_DIR / "round4_summary.json").read_text(encoding="utf-8"))
    return dict(data["validation_model_vs_best_multi_agent"]["best_multi_agent_by_sharpe"]["weights_requested"])


def _all_rebalance_dates() -> List[date]:
    with get_session() as session:
        rows = (
            session.query(Feature.trading_date)
            .order_by(Feature.trading_date)
            .distinct()
            .all()
        )
    return _monthly_rebalance_dates([r[0] for r in rows])


def _load_picks_map(experiment_id: str) -> Dict[str, List[Dict[str, object]]]:
    rows = json.loads((ARTIFACTS_DIR / f"experiment_picks_{experiment_id}.json").read_text(encoding="utf-8"))
    out: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        out.setdefault(str(r["date"]), []).append(r)
    return out


def _attribution_trend(rows_by_date: Dict[str, List[Dict[str, object]]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for d, rows in rows_by_date.items():
        agg: Dict[str, List[float]] = {}
        for r in rows:
            agents = ((r.get("reason_json") or {}).get("agents") or {})
            meta = ((r.get("reason_json") or {}).get("_selection_meta") or {})
            wu = meta.get("weights_used", {}) if isinstance(meta, dict) else {}
            for a in ["tech", "flow", "margin", "fund"]:
                ao = agents.get(a, {}) or {}
                sig = float(ao.get("signal", 0.0))
                conf = float(ao.get("confidence", 0.0))
                contrib = float(wu.get(a, 0.0)) * sig * conf
                agg.setdefault(f"{a}_contrib", []).append(contrib)
                agg.setdefault(f"{a}_signal", []).append(sig)
        out[d] = {k: float(np.mean(v)) for k, v in agg.items() if v}
    return out


def _window_report(months: int, rb_all: List[date], weights: Dict[str, float]) -> Dict[str, object]:
    if len(rb_all) < months + 1:
        return {"window_months": months, "available": False, "reason": "insufficient_rebalance_dates"}
    rb = rb_all[-(months + 1) :]
    start_date, end_date = rb[0], rb[-1]
    cfg_base = load_config()
    model_cfg = _to_config(cfg_base, {"selection_mode": "model", "data_quality_mode": "research", "topn": cfg_base.topn}, cfg_base.topn)
    ma_cfg = _to_config(
        cfg_base,
        {
            "selection_mode": "multi_agent",
            "data_quality_mode": "research",
            "topn": cfg_base.topn,
            "multi_agent_weights": weights,
        },
        cfg_base.topn,
    )
    tag = f"{months}m_{start_date.isoformat()}_{end_date.isoformat()}"
    model_id = f"shadow_monitor_model_{tag}"
    ma_id = f"shadow_monitor_ma_{tag}"
    model_eval = run_experiment(model_id, start_date, end_date, model_cfg, resume=False)
    ma_eval = run_experiment(ma_id, start_date, end_date, ma_cfg, resume=False)
    model_map = _load_picks_map(model_id)
    ma_map = _load_picks_map(ma_id)

    with get_session() as session:
        px_df = _merge_adjusted_close(
            _load_price_frame(session, start_date, end_date),
            _load_adjust_factors(session, start_date, end_date),
            bool(getattr(cfg_base, "use_adjusted_price", True)),
        )
    regime_df = _build_market_regime(start_date, end_date, rb)
    regime_map = {str(r["trading_date"]): {"trend": r["trend_regime"], "strength": r["strength_regime"], "vol": r["vol_regime"]} for _, r in regime_df.iterrows()}

    per_period = []
    for i in range(len(rb) - 1):
        d1, d2 = rb[i], rb[i + 1]
        k = d1.isoformat()
        mret, _ = _period_return([str(x["stock_id"]) for x in model_map.get(k, [])], d1, d2, px_df, float(cfg_base.transaction_cost_pct))
        aret, _ = _period_return([str(x["stock_id"]) for x in ma_map.get(k, [])], d1, d2, px_df, float(cfg_base.transaction_cost_pct))
        s1 = {str(x["stock_id"]) for x in model_map.get(k, [])}
        s2 = {str(x["stock_id"]) for x in ma_map.get(k, [])}
        overlap = len(s1 & s2) / float(max(len(s1 | s2), 1))
        per_period.append(
            {
                "entry_date": k,
                "exit_date": d2.isoformat(),
                "model_return": mret,
                "multi_agent_return": aret,
                "picks_overlap_jaccard": overlap,
                "regime_tags": regime_map.get(k, {}),
            }
        )

    return {
        "window_months": months,
        "available": True,
        "period": {"start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "rebalance_dates": [d.isoformat() for d in rb]},
        "model_experiment_id": model_id,
        "multi_agent_experiment_id": ma_id,
        "model_metrics": model_eval.get("metrics", {}),
        "multi_agent_metrics": ma_eval.get("metrics", {}),
        "model_invalid_result": bool(model_eval.get("invalid_result", False)),
        "multi_agent_invalid_result": bool(ma_eval.get("invalid_result", False)),
        "avg_picks_overlap_jaccard": float(np.mean([x["picks_overlap_jaccard"] for x in per_period])) if per_period else 0.0,
        "per_period": per_period,
        "attribution_trend": _attribution_trend(ma_map),
    }


def run_shadow_monitor() -> Dict[str, object]:
    rb_all = _all_rebalance_dates()
    weights = _load_best_weights()
    windows = {}
    for m in [1, 3, 6]:
        windows[f"{m}m"] = _window_report(m, rb_all, weights)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "multi_agent_weights": weights,
        "windows": windows,
    }
    (ARTIFACTS_DIR / "shadow_monitor_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# Shadow Monitor Latest",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- multi_agent_weights(theme=0): `{weights}`",
    ]
    for key in ["1m", "3m", "6m"]:
        w = windows.get(key, {})
        lines.extend(["", f"## Window {key}"])
        if not w.get("available"):
            lines.append(f"- unavailable: `{w.get('reason', 'unknown')}`")
            continue
        mm = w["model_metrics"]
        am = w["multi_agent_metrics"]
        lines.extend(
            [
                f"- period: `{w['period']['start_date']} ~ {w['period']['end_date']}`",
                "| strategy | total_return | sharpe | max_drawdown | turnover | overlap |",
                "|---|---:|---:|---:|---:|---:|",
                f"| model | {mm.get('total_return')} | {mm.get('sharpe')} | {mm.get('max_drawdown')} | {mm.get('turnover')} | {w['avg_picks_overlap_jaccard']} |",
                f"| multi_agent_4 | {am.get('total_return')} | {am.get('sharpe')} | {am.get('max_drawdown')} | {am.get('turnover')} | {w['avg_picks_overlap_jaccard']} |",
                "",
                "per-period regime tags:",
            ]
        )
        for p in w["per_period"]:
            lines.append(f"- {p['entry_date']} -> {p['exit_date']}: {p['regime_tags']}")
        lines.append("attribution trend (avg contrib):")
        for d, tr in sorted(w["attribution_trend"].items()):
            lines.append(
                f"- {d}: tech={tr.get('tech_contrib', 0):.4f}, flow={tr.get('flow_contrib', 0):.4f}, "
                f"margin={tr.get('margin_contrib', 0):.4f}, fund={tr.get('fund_contrib', 0):.4f}"
            )
    (ARTIFACTS_DIR / "shadow_monitor_latest.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    payload = run_shadow_monitor()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
