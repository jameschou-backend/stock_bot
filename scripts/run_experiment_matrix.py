#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from skills import multi_agent_selector
from scripts.agent_attribution_report import build_agent_attribution, write_outputs as write_attribution_outputs
from scripts.evaluate_experiment import ARTIFACTS_DIR, _to_config, run_experiment


def _load_matrix(path: Path) -> Dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_picks_df(experiment_id: str) -> pd.DataFrame:
    path = ARTIFACTS_DIR / f"experiment_picks_{experiment_id}.json"
    if not path.exists():
        return pd.DataFrame(columns=["date", "stock_id", "rank", "score"])
    data = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["date", "stock_id", "rank", "score"])
    df["date"] = df["date"].astype(str)
    df["stock_id"] = df["stock_id"].astype(str)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df


def _avg_weights_used(experiment_id: str) -> Dict[str, float]:
    path = ARTIFACTS_DIR / f"experiment_picks_{experiment_id}.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    agg: Dict[str, List[float]] = {}
    for r in rows:
        reason = r.get("reason_json", {}) or {}
        meta = reason.get("_selection_meta", {}) if isinstance(reason, dict) else {}
        wu = meta.get("weights_used", {}) if isinstance(meta, dict) else {}
        for k, v in wu.items():
            agg.setdefault(k, []).append(float(v))
    return {k: float(np.mean(vs)) for k, vs in agg.items() if vs}


def _is_invalid_metrics(metrics: Dict[str, object]) -> bool:
    for v in metrics.values():
        if isinstance(v, (float, np.floating)) and not np.isfinite(float(v)):
            return True
        if isinstance(v, str) and v.strip().lower() in {"inf", "-inf", "nan"}:
            return True
    return False


def _degenerate_multi_agent(experiment_id: str) -> tuple[bool, Dict[str, int], List[str]]:
    path = ARTIFACTS_DIR / f"experiment_picks_{experiment_id}.json"
    if not path.exists():
        return False, {}, []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return True, {}, ["empty picks rows"]
    avail_counts: List[int] = []
    reason_counter: Counter[str] = Counter()
    for r in rows:
        reason = (r.get("reason_json") or {})
        agents = reason.get("agents", {}) if isinstance(reason, dict) else {}
        available = []
        for name, out in agents.items():
            is_unavailable = bool((out or {}).get("unavailable"))
            if not is_unavailable:
                available.append(name)
            else:
                for msg in (out or {}).get("reasons", []):
                    reason_counter[f"{name}: {msg}"] += 1
        avail_counts.append(len(available))
    degenerate = (float(np.mean(avail_counts)) if avail_counts else 0.0) <= 1.0
    top_reasons = [f"{k} ({v})" for k, v in reason_counter.most_common(10)]
    return degenerate, dict(reason_counter), top_reasons


def _pairwise_compare(base_df: pd.DataFrame, target_df: pd.DataFrame, topn: int) -> Dict[str, float]:
    if base_df.empty or target_df.empty:
        return {"topn_overlap": 0.0, "rank_corr_spearman": 0.0}
    dates = sorted(set(base_df["date"]) & set(target_df["date"]))
    if not dates:
        return {"topn_overlap": 0.0, "rank_corr_spearman": 0.0}
    overlaps = []
    corrs = []
    for d in dates:
        a = base_df[base_df["date"] == d].sort_values("rank").head(topn)
        b = target_df[target_df["date"] == d].sort_values("rank").head(topn)
        sa, sb = set(a["stock_id"]), set(b["stock_id"])
        overlaps.append(len(sa & sb) / float(max(topn, 1)))
        merged = a[["stock_id", "rank"]].merge(b[["stock_id", "rank"]], on="stock_id", suffixes=("_a", "_b"))
        if len(merged) >= 2:
            corr = float(merged["rank_a"].corr(merged["rank_b"], method="spearman"))
            corrs.append(0.0 if np.isnan(corr) else corr)
    return {
        "topn_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "rank_corr_spearman": float(np.mean(corrs)) if corrs else 0.0,
    }


def _markdown_table(rows: List[Dict[str, object]]) -> List[str]:
    lines = [
        "| experiment_id | selection_mode | dq_mode | total_return | cagr | max_dd | sharpe | picks_stability | degraded_ratio | invalid_result | degenerate_multi_agent |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        m = r.get("metrics", {})
        lines.append(
            f"| {r['experiment_id']} | {r['selection_mode']} | {r['data_quality_mode']} | "
            f"{m.get('total_return')} | {m.get('cagr')} | {m.get('max_drawdown')} | {m.get('sharpe')} | "
            f"{m.get('picks_stability')} | {r.get('degraded_ratio')} | {r.get('invalid_result')} | {r.get('degenerate_multi_agent')} |"
        )
    return lines


def _write_multi_agent_debug(rows: List[Dict[str, object]]) -> None:
    md_path = ARTIFACTS_DIR / "debug_multi_agent_availability.md"
    ma_rows = [r for r in rows if r.get("selection_mode") == "multi_agent"]
    lines = [
        "# Debug Multi Agent Availability",
        "",
        "## Agent Required Columns",
    ]
    for agent, cols in multi_agent_selector.AGENT_REQUIRED_COLUMNS.items():
        lines.append(f"- {agent}: `{cols}`")

    lines.extend(["", "## Experiment Diagnostics"])
    for r in ma_rows:
        lines.extend(
            [
                f"### {r['experiment_id']}",
                f"- feature_columns_observed_count(raw): `{len(r.get('feature_columns_observed', []))}`",
                f"- feature_columns_input_selector_count: `{len(r.get('feature_columns_input_selector', []))}`",
                f"- degenerate_multi_agent: `{r.get('degenerate_multi_agent')}`",
                f"- weights_used_avg: `{r.get('weights_used')}`",
                "",
                "| agent | required_cols | present_in_feature_df |",
                "|---|---:|---:|",
            ]
        )
        observed = set(r.get("feature_columns_input_selector", []))
        for agent, cols in multi_agent_selector.AGENT_REQUIRED_COLUMNS.items():
            present = sum(1 for c in cols if c in observed)
            lines.append(f"| {agent} | {len(cols)} | {present} |")
        lines.append("")
        lines.append("Top unavailable reasons:")
        for msg in r.get("top_unavailable_reasons", []):
            lines.append(f"- {msg}")
        lines.append("")

    lines.extend(
        [
            "## Root Cause for `degraded_datasets=['raw_fundamentals']` but others unavailable",
            "- 主要原因是 `feature_df` 缺少 multi-agent 所需欄位，造成 `tech/flow/theme` 被判定為 `missing columns`，與 degraded dataset 無關。",
            "- 修正後改為在評估階段強制補齊 `FEATURE_COLUMNS + AGENT_REQUIRED_COLUMNS`，缺值以 imputation 補齊，避免「欄位不存在」誤判 unavailable。",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def run_matrix(matrix_path: Path, resume: bool | None = None) -> Dict[str, object]:
    matrix = _load_matrix(matrix_path)
    start_date = date.fromisoformat(matrix["start_date"])
    end_date = date.fromisoformat(matrix["end_date"])
    topn = int(matrix.get("topn", 20))
    base = load_config()
    experiments = matrix.get("experiments", [])

    results = []
    resume_flag = bool(matrix.get("resume", False)) if resume is None else bool(resume)

    for spec in experiments:
        name = str(spec["name"])
        ts = datetime.now().strftime("%Y%m%d")
        experiment_id = f"{name}_{ts}"
        cfg = _to_config(base, spec, topn=topn)
        eval_payload = run_experiment(
            experiment_id=experiment_id,
            start_date=start_date,
            end_date=end_date,
            cfg=cfg,
            resume=resume_flag,
        )
        attr_payload = build_agent_attribution(experiment_id)
        write_attribution_outputs(attr_payload)

        manifest_dir = Path(eval_payload["manifest_dir"])
        manifest_paths = sorted([str(p) for p in manifest_dir.glob("run_manifest_*.json")])
        results.append(
            {
                "experiment_id": experiment_id,
                "name": name,
                "selection_mode": cfg.selection_mode,
                "data_quality_mode": cfg.data_quality_mode,
                "weights_requested": cfg.multi_agent_weights if cfg.selection_mode == "multi_agent" else None,
                "weights_used": _avg_weights_used(experiment_id),
                "degraded_ratio": eval_payload.get("degraded_ratio", 0.0),
                "number_of_rebalance_dates": eval_payload.get("number_of_rebalance_dates", 0),
                "run_ids": [Path(p).stem.replace("run_manifest_", "") for p in manifest_paths],
                "manifest_paths": manifest_paths,
                "evaluation_json": str(ARTIFACTS_DIR / f"evaluation_{experiment_id}.json"),
                "attribution_json": str(ARTIFACTS_DIR / f"agent_attribution_{experiment_id}.json"),
                "metrics": eval_payload.get("metrics", {}),
                "invalid_result": bool(eval_payload.get("invalid_result", False))
                or _is_invalid_metrics(eval_payload.get("metrics", {})),
                "skipped_periods": eval_payload.get("skipped_periods", []),
                "feature_columns_observed": eval_payload.get("feature_columns_observed", []),
                "feature_columns_input_selector": eval_payload.get("feature_columns_input_selector", []),
            }
        )

        if cfg.selection_mode == "multi_agent":
            degenerate, reason_counts, top_reasons = _degenerate_multi_agent(experiment_id)
            results[-1]["degenerate_multi_agent"] = bool(degenerate)
            results[-1]["unavailable_reason_counts"] = reason_counts
            results[-1]["top_unavailable_reasons"] = top_reasons
        else:
            results[-1]["degenerate_multi_agent"] = False

    baseline_id = results[0]["experiment_id"] if results else None
    baseline_df = _load_picks_df(baseline_id) if baseline_id else pd.DataFrame()
    for r in results:
        comp = _pairwise_compare(baseline_df, _load_picks_df(r["experiment_id"]), topn=topn)
        r["comparison_vs_baseline"] = comp

    payload = {
        "matrix_file": str(matrix_path),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "topn": topn,
        "baseline_experiment_id": baseline_id,
        "experiments": results,
    }
    jp = ARTIFACTS_DIR / "experiment_matrix_summary.json"
    mp = ARTIFACTS_DIR / "experiment_matrix_summary.md"
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = ["# Experiment Matrix Summary", ""]
    lines.extend(_markdown_table(results))
    lines.extend(["", f"- baseline: `{baseline_id}`"])
    mp.write_text("\n".join(lines), encoding="utf-8")
    _write_multi_agent_debug(results)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment matrix for model/multi-agent evaluation")
    parser.add_argument("--matrix", default=str(PROJECT_ROOT / "experiments" / "multi_agent_matrix.yaml"))
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=None)
    args = parser.parse_args()
    payload = run_matrix(Path(args.matrix), resume=args.resume)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
