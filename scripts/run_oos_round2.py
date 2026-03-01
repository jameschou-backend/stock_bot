#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from app.models import Feature
from scripts.evaluate_experiment import ARTIFACTS_DIR, _monthly_rebalance_dates, _to_config, run_experiment


def _load_matrix(path: Path) -> Dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _numeric_metric(row: Dict[str, object], key: str) -> float:
    v = row.get("metrics", {}).get(key)
    return float(v) if isinstance(v, (int, float)) else float("-inf")


def _compute_split_dates(start_date: date, end_date: date) -> Dict[str, str]:
    with get_session() as session:
        rows = (
            session.query(Feature.trading_date)
            .filter(Feature.trading_date.between(start_date, end_date))
            .distinct()
            .order_by(Feature.trading_date)
            .all()
        )
    trading_dates = [r[0] for r in rows]
    rb_dates = _monthly_rebalance_dates(trading_dates)
    if len(rb_dates) < 4:
        raise ValueError("Not enough rebalance dates for 70/30 OOS split")
    split_idx = max(1, min(int(len(rb_dates) * 0.7), len(rb_dates) - 2))
    dev_last_rb = rb_dates[split_idx - 1]
    val_first_rb = rb_dates[split_idx]
    return {
        "split_rule": "rebalance_dates_count_70_30",
        "rebalance_dates_total": len(rb_dates),
        "development_start": start_date.isoformat(),
        "development_end": (val_first_rb - timedelta(days=1)).isoformat(),
        "validation_start": val_first_rb.isoformat(),
        "validation_end": end_date.isoformat(),
        "development_rebalance_dates": [d.isoformat() for d in rb_dates[:split_idx]],
        "validation_rebalance_dates": [d.isoformat() for d in rb_dates[split_idx:]],
        "development_last_rebalance": dev_last_rb.isoformat(),
    }


def run_oos(matrix_path: Path) -> Dict[str, object]:
    matrix = _load_matrix(matrix_path)
    base = load_config()
    topn = int(matrix.get("topn", 20))
    start_date = date.fromisoformat(str(matrix["start_date"]))
    end_date = date.fromisoformat(str(matrix["end_date"]))
    split = _compute_split_dates(start_date, end_date)
    dev_start = date.fromisoformat(split["development_start"])
    dev_end = date.fromisoformat(split["development_end"])
    val_start = date.fromisoformat(split["validation_start"])
    val_end = date.fromisoformat(split["validation_end"])
    ts = datetime.now().strftime("%Y%m%d")

    development_results: List[Dict[str, object]] = []
    for spec in matrix.get("experiments", []):
        cfg = _to_config(base, spec, topn=topn)
        exp_id = f"{spec['name']}_dev_{ts}"
        payload = run_experiment(
            experiment_id=exp_id,
            start_date=dev_start,
            end_date=dev_end,
            cfg=cfg,
            resume=False,
        )
        development_results.append(
            {
                "name": spec["name"],
                "experiment_id": exp_id,
                "selection_mode": cfg.selection_mode,
                "data_quality_mode": cfg.data_quality_mode,
                "weights_requested": cfg.multi_agent_weights,
                "metrics": payload.get("metrics", {}),
            }
        )

    best_by_return = max(development_results, key=lambda r: _numeric_metric(r, "total_return"))
    best_by_risk = max(development_results, key=lambda r: _numeric_metric(r, "sharpe"))
    chosen = best_by_risk

    model_spec = {"selection_mode": "model", "data_quality_mode": "research", "topn": topn}
    model_cfg = _to_config(base, model_spec, topn=topn)
    model_val_id = f"model_research_val_{ts}"
    model_val = run_experiment(
        experiment_id=model_val_id,
        start_date=val_start,
        end_date=val_end,
        cfg=model_cfg,
        resume=False,
    )

    chosen_cfg = _to_config(
        base,
        {
            "selection_mode": "multi_agent",
            "data_quality_mode": "research",
            "topn": topn,
            "multi_agent_weights": chosen["weights_requested"],
        },
        topn=topn,
    )
    chosen_val_id = f"{chosen['name']}_val_{ts}"
    chosen_val = run_experiment(
        experiment_id=chosen_val_id,
        start_date=val_start,
        end_date=val_end,
        cfg=chosen_cfg,
        resume=False,
    )

    model_val_ret = _numeric_metric({"metrics": model_val.get("metrics", {})}, "total_return")
    ma_val_ret = _numeric_metric({"metrics": chosen_val.get("metrics", {})}, "total_return")
    model_val_sharpe = _numeric_metric({"metrics": model_val.get("metrics", {})}, "sharpe")
    ma_val_sharpe = _numeric_metric({"metrics": chosen_val.get("metrics", {})}, "sharpe")
    overfit_suspected = bool((ma_val_ret + 1e-12) < model_val_ret and (ma_val_sharpe + 1e-12) < model_val_sharpe)

    payload = {
        "round2_matrix_file": str(matrix_path),
        "split": split,
        "development": {
            "results": development_results,
            "best_by_return": best_by_return,
            "best_by_risk_adjusted": best_by_risk,
            "chosen_for_validation": {
                "rule": "prefer_best_by_risk_adjusted_sharpe",
                "name": chosen["name"],
                "weights_requested": chosen["weights_requested"],
            },
        },
        "validation": {
            "model": {"experiment_id": model_val_id, "metrics": model_val.get("metrics", {})},
            "chosen_multi_agent": {
                "name": chosen["name"],
                "experiment_id": chosen_val_id,
                "weights_requested": chosen["weights_requested"],
                "metrics": chosen_val.get("metrics", {}),
            },
        },
        "overfit_assessment": {
            "suspected": overfit_suspected,
            "rule": "suspected_when_chosen_multi_agent_underperforms_model_on_return_and_sharpe",
        },
    }

    out_json = ARTIFACTS_DIR / "oos_summary.json"
    out_md = ARTIFACTS_DIR / "oos_summary.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# OOS Summary",
        "",
        "## Split",
        f"- split_rule: `{split['split_rule']}`",
        f"- development: `{split['development_start']} ~ {split['development_end']}`",
        f"- validation: `{split['validation_start']} ~ {split['validation_end']}`",
        f"- rebalance_count: `{split['rebalance_dates_total']}`",
        "",
        "## Development Results",
        "| name | total_return | sharpe | max_drawdown | turnover | picks_stability |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in development_results:
        m = r["metrics"]
        lines.append(
            f"| {r['name']} | {m.get('total_return')} | {m.get('sharpe')} | {m.get('max_drawdown')} | "
            f"{m.get('turnover')} | {m.get('picks_stability')} |"
        )
    lines.extend(
        [
            "",
            f"- development best by return: `{best_by_return['name']}` (total_return={best_by_return['metrics'].get('total_return')})",
            f"- development best by risk-adjusted: `{best_by_risk['name']}` (sharpe={best_by_risk['metrics'].get('sharpe')})",
            "",
            "## Validation Comparison",
            f"- model_research total_return: `{model_val.get('metrics', {}).get('total_return')}`",
            f"- model_research sharpe: `{model_val.get('metrics', {}).get('sharpe')}`",
            f"- chosen_multi_agent (`{chosen['name']}`) total_return: `{chosen_val.get('metrics', {}).get('total_return')}`",
            f"- chosen_multi_agent (`{chosen['name']}`) sharpe: `{chosen_val.get('metrics', {}).get('sharpe')}`",
            "",
            "## Overfit Assessment",
            f"- suspected_overfit: `{overfit_suspected}`",
            f"- rule: `{payload['overfit_assessment']['rule']}`",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    matrix_path = PROJECT_ROOT / "experiments" / "multi_agent_matrix_round2.yaml"
    payload = run_oos(matrix_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
