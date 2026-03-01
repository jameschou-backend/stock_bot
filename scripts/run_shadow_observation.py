#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from app.models import Feature
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


def _load_round4_best_weights() -> Dict[str, float]:
    path = ARTIFACTS_DIR / "round4_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data["validation_model_vs_best_multi_agent"]["best_multi_agent_by_sharpe"]["weights_requested"])


def _recent_rebalance_dates(n: int) -> List[date]:
    with get_session() as session:
        rows = (
            session.query(Feature.trading_date)
            .order_by(Feature.trading_date)
            .distinct()
            .all()
        )
    rb = _monthly_rebalance_dates([r[0] for r in rows])
    need = min(len(rb), max(n + 1, 2))
    return rb[-need:]


def _load_picks_map(experiment_id: str) -> Dict[str, List[str]]:
    path = ARTIFACTS_DIR / f"experiment_picks_{experiment_id}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {}
    for r in rows:
        out.setdefault(str(r["date"]), []).append(str(r["stock_id"]))
    return {k: sorted(v) for k, v in out.items()}


def _agent_trend(experiment_id: str) -> Dict[str, Dict[str, float]]:
    path = ARTIFACTS_DIR / f"experiment_picks_{experiment_id}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    bucket: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        d = str(r["date"])
        agents = ((r.get("reason_json") or {}).get("agents") or {})
        bucket.setdefault(d, {})
        for a in ["tech", "flow", "margin", "fund"]:
            out = agents.get(a, {}) or {}
            sig = float(out.get("signal", 0.0))
            conf = float(out.get("confidence", 0.0))
            contrib = sig * conf
            bucket[d].setdefault(f"{a}_signal", []).append(sig)
            bucket[d].setdefault(f"{a}_contrib", []).append(contrib)
    summary: Dict[str, Dict[str, float]] = {}
    for d, metrics in bucket.items():
        summary[d] = {k: float(np.mean(v)) for k, v in metrics.items() if v}
    return summary


def run_shadow_observation(recent_n: int = 6) -> Dict[str, object]:
    weights = _load_round4_best_weights()
    rb = _recent_rebalance_dates(recent_n)
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
    model_id = f"shadow_model_{start_date.isoformat()}_{end_date.isoformat()}"
    ma_id = f"shadow_ma_{start_date.isoformat()}_{end_date.isoformat()}"
    model_eval = run_experiment(model_id, start_date, end_date, model_cfg, resume=False)
    ma_eval = run_experiment(ma_id, start_date, end_date, ma_cfg, resume=False)

    model_picks = _load_picks_map(model_id)
    ma_picks = _load_picks_map(ma_id)
    with get_session() as session:
        px_df = _merge_adjusted_close(
            _load_price_frame(session, start_date, end_date),
            _load_adjust_factors(session, start_date, end_date),
            bool(getattr(cfg_base, "use_adjusted_price", True)),
        )

    per_period = []
    for i in range(len(rb) - 1):
        d1, d2 = rb[i], rb[i + 1]
        k1 = d1.isoformat()
        mret, _ = _period_return(model_picks.get(k1, []), d1, d2, px_df, float(cfg_base.transaction_cost_pct))
        aret, _ = _period_return(ma_picks.get(k1, []), d1, d2, px_df, float(cfg_base.transaction_cost_pct))
        s1, s2 = set(model_picks.get(k1, [])), set(ma_picks.get(k1, []))
        overlap = len(s1 & s2) / float(max(len(s1 | s2), 1))
        per_period.append(
            {
                "entry_date": d1.isoformat(),
                "exit_date": d2.isoformat(),
                "model_period_return": mret,
                "multi_agent_period_return": aret,
                "picks_overlap_jaccard": overlap,
            }
        )

    payload = {
        "period": {"start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "rebalance_dates": [d.isoformat() for d in rb]},
        "model_experiment_id": model_id,
        "multi_agent_experiment_id": ma_id,
        "multi_agent_weights": weights,
        "model_metrics": model_eval.get("metrics", {}),
        "multi_agent_metrics": ma_eval.get("metrics", {}),
        "avg_picks_overlap_jaccard": float(np.mean([p["picks_overlap_jaccard"] for p in per_period])) if per_period else 0.0,
        "per_period_comparison": per_period,
        "attribution_trend": _agent_trend(ma_id),
    }

    (ARTIFACTS_DIR / "shadow_observation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    lines = [
        "# Shadow Observation Summary",
        "",
        f"- period: `{start_date.isoformat()} ~ {end_date.isoformat()}`",
        f"- multi_agent_weights(theme=0): `{weights}`",
        "",
        "## Metrics (Model vs Multi-Agent)",
        "| strategy | total_return | sharpe | max_drawdown | turnover | picks_stability |",
        "|---|---:|---:|---:|---:|---:|",
        f"| model | {payload['model_metrics'].get('total_return')} | {payload['model_metrics'].get('sharpe')} | {payload['model_metrics'].get('max_drawdown')} | {payload['model_metrics'].get('turnover')} | {payload['model_metrics'].get('picks_stability')} |",
        f"| multi_agent_4 | {payload['multi_agent_metrics'].get('total_return')} | {payload['multi_agent_metrics'].get('sharpe')} | {payload['multi_agent_metrics'].get('max_drawdown')} | {payload['multi_agent_metrics'].get('turnover')} | {payload['multi_agent_metrics'].get('picks_stability')} |",
        "",
        f"- average_picks_overlap_jaccard: `{payload['avg_picks_overlap_jaccard']:.4f}`",
        "",
        "## Per-Period",
        "| entry_date | exit_date | model_ret | multi_agent_ret | picks_overlap_jaccard |",
        "|---|---|---:|---:|---:|",
    ]
    for p in per_period:
        lines.append(
            f"| {p['entry_date']} | {p['exit_date']} | {p['model_period_return']} | {p['multi_agent_period_return']} | {p['picks_overlap_jaccard']:.4f} |"
        )
    lines.extend(["", "## Attribution Trend (tech/flow/margin/fund)"])
    for d, tr in sorted(payload["attribution_trend"].items()):
        lines.append(
            f"- {d}: tech={tr.get('tech_contrib', 0):.4f}, flow={tr.get('flow_contrib', 0):.4f}, "
            f"margin={tr.get('margin_contrib', 0):.4f}, fund={tr.get('fund_contrib', 0):.4f}"
        )
    (ARTIFACTS_DIR / "shadow_observation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    payload = run_shadow_observation(recent_n=6)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
