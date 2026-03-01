#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _load(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(v):
    return v if isinstance(v, (int, float)) else None


def _best(rows: List[Dict], key: str, reverse: bool = True):
    scored = [(r, _metric_value(r.get("metrics", {}).get(key))) for r in rows]
    scored = [(r, s) for r, s in scored if s is not None]
    if not scored:
        return None
    scored.sort(key=lambda x: x[1], reverse=reverse)
    return scored[0][0]


def render_latest() -> Path:
    summary = _load(ARTIFACTS_DIR / "experiment_matrix_summary.json")
    rows = summary.get("experiments", [])
    if not rows:
        raise ValueError("No experiments in matrix summary")

    best_perf = _best(rows, "total_return", reverse=True)
    most_stable = _best(rows, "picks_stability", reverse=True)
    resilient = None
    resilient_candidates = [r for r in rows if float(r.get("degraded_ratio", 0.0)) > 0]
    if resilient_candidates:
        resilient = _best(resilient_candidates, "total_return", reverse=True)

    lines = [
        "# Experiment Summary Latest",
        "",
        "## Performance Table",
        "",
        "| experiment_id | mode | dq | total_return | cagr | max_dd | sharpe | stability | degraded_ratio | overlap_vs_baseline | invalid_result | degenerate_multi_agent |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        m = r.get("metrics", {})
        comp = r.get("comparison_vs_baseline", {})
        lines.append(
            f"| {r['experiment_id']} | {r['selection_mode']} | {r['data_quality_mode']} | "
            f"{m.get('total_return')} | {m.get('cagr')} | {m.get('max_drawdown')} | {m.get('sharpe')} | "
            f"{m.get('picks_stability')} | {r.get('degraded_ratio')} | {comp.get('topn_overlap')} | "
            f"{r.get('invalid_result', False)} | {r.get('degenerate_multi_agent', False)} |"
        )

    lines.extend(
        [
            "",
            "## Model vs Multi-Agent",
            f"- model count: `{sum(1 for r in rows if r['selection_mode']=='model')}`",
            f"- multi_agent count: `{sum(1 for r in rows if r['selection_mode']=='multi_agent')}`",
            "",
            "## Strict vs Research",
            f"- strict count: `{sum(1 for r in rows if r['data_quality_mode']=='strict')}`",
            f"- research count: `{sum(1 for r in rows if r['data_quality_mode']=='research')}`",
            "",
            "## Weights Comparison",
            "- compare `weights_requested` / `weights_used` in `experiment_matrix_summary.json`",
            "",
            "## Agent Attribution Summary",
        ]
    )
    for r in rows:
        attr_path = ARTIFACTS_DIR / f"agent_attribution_{r['experiment_id']}.json"
        if not attr_path.exists():
            continue
        attr = _load(attr_path)
        top = attr.get("top_contributor_distribution", {})
        lines.append(f"- {r['experiment_id']}: top_contributor_distribution={top}")

    lines.extend(["", "## Conclusion"])
    if best_perf:
        lines.append(f"- 最佳績效組：`{best_perf['experiment_id']}`（total_return={best_perf['metrics'].get('total_return')}）")
    if most_stable:
        lines.append(f"- 最穩定組：`{most_stable['experiment_id']}`（picks_stability={most_stable['metrics'].get('picks_stability')}）")
    if resilient:
        lines.append(f"- degraded 下最韌性組：`{resilient['experiment_id']}`（total_return={resilient['metrics'].get('total_return')}）")
    else:
        lines.append("- degraded 下最韌性組：`N/A`（本批次無 degraded period）")

    out = ARTIFACTS_DIR / "experiment_summary_latest.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    out = render_latest()
    print(f"written: {out}")


if __name__ == "__main__":
    main()
