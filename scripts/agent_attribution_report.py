#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
AGENTS = ["tech", "flow", "margin", "fund", "theme"]


def _load_picks(experiment_id: str) -> List[Dict]:
    path = ARTIFACTS_DIR / f"experiment_picks_{experiment_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_agent_attribution(experiment_id: str) -> Dict[str, object]:
    rows = _load_picks(experiment_id)
    if not rows:
        return {"experiment_id": experiment_id, "error": "no picks data"}

    agent_stats = {a: {"signal": [], "confidence": [], "unavailable": 0, "total": 0, "final_scores": []} for a in AGENTS}
    support_counts = {a: {"support_selected": 0, "oppose_selected": 0, "selected_total": 0} for a in AGENTS}
    top_contributor_counter = Counter()
    degraded_unavailable = Counter()
    degraded_rows = []
    normal_rows = []

    for r in rows:
        reason = r.get("reason_json", {}) or {}
        agents = reason.get("agents", {}) if isinstance(reason, dict) else {}
        meta = reason.get("_selection_meta", {}) if isinstance(reason, dict) else {}
        weights_used = meta.get("weights_used", {}) if isinstance(meta, dict) else {}
        final_score = float(r.get("score", 0.0))
        contrib_map = {}

        for a in AGENTS:
            out = agents.get(a, {})
            if not out:
                continue
            sig = float(out.get("signal", 0))
            conf = float(out.get("confidence", 0))
            unavail = bool(out.get("unavailable", False))
            agent_stats[a]["total"] += 1
            agent_stats[a]["signal"].append(sig)
            agent_stats[a]["confidence"].append(conf)
            agent_stats[a]["final_scores"].append(final_score)
            if unavail:
                agent_stats[a]["unavailable"] += 1
            support_counts[a]["selected_total"] += 1
            if sig >= 1:
                support_counts[a]["support_selected"] += 1
            if sig <= -1:
                support_counts[a]["oppose_selected"] += 1
            score_component = (1.0 if sig == 2 else 0.5 if sig == 1 else -0.5 if sig == -1 else -1.0 if sig == -2 else 0.0) * conf
            contrib_map[a] = float(weights_used.get(a, 0.0)) * score_component
            if r.get("degraded_mode") and unavail:
                degraded_unavailable[a] += 1

        if contrib_map:
            top_agent = max(contrib_map.items(), key=lambda kv: abs(kv[1]))[0]
            top_contributor_counter[top_agent] += 1

        sig_snapshot = {a: float(agents.get(a, {}).get("signal", 0.0)) for a in AGENTS}
        if r.get("degraded_mode"):
            degraded_rows.append(sig_snapshot)
        else:
            normal_rows.append(sig_snapshot)

    out_stats = {}
    for a in AGENTS:
        st = agent_stats[a]
        total = st["total"] or 1
        corr = (
            float(pd.Series(st["signal"]).corr(pd.Series(st["final_scores"])))
            if len(st["signal"]) >= 2
            else 0.0
        )
        out_stats[a] = {
            "average_signal": float(np.mean(st["signal"])) if st["signal"] else 0.0,
            "average_confidence": float(np.mean(st["confidence"])) if st["confidence"] else 0.0,
            "unavailable_ratio": float(st["unavailable"] / total),
            "signal_final_score_correlation": corr if not np.isnan(corr) else 0.0,
            "support_selected_ratio": float(support_counts[a]["support_selected"] / max(support_counts[a]["selected_total"], 1)),
            "oppose_but_selected_ratio": float(support_counts[a]["oppose_selected"] / max(support_counts[a]["selected_total"], 1)),
        }

    degraded_delta = {}
    if degraded_rows and normal_rows:
        ddf = pd.DataFrame(degraded_rows)
        ndf = pd.DataFrame(normal_rows)
        for a in AGENTS:
            degraded_delta[a] = float(ddf[a].mean() - ndf[a].mean())
    else:
        degraded_delta = {a: 0.0 for a in AGENTS}

    payload = {
        "experiment_id": experiment_id,
        "agent_stats": out_stats,
        "top_contributor_distribution": dict(top_contributor_counter),
        "degraded_unavailable_counts": dict(degraded_unavailable),
        "degraded_vs_normal_signal_delta": degraded_delta,
        "rows": len(rows),
    }
    return payload


def write_outputs(payload: Dict[str, object]) -> None:
    experiment_id = payload["experiment_id"]
    jp = ARTIFACTS_DIR / f"agent_attribution_{experiment_id}.json"
    mp = ARTIFACTS_DIR / f"agent_attribution_{experiment_id}.md"
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# Agent Attribution {experiment_id}", ""]
    for a, st in payload.get("agent_stats", {}).items():
        lines.extend(
            [
                f"## {a}",
                f"- average_signal: `{st['average_signal']:.4f}`",
                f"- average_confidence: `{st['average_confidence']:.4f}`",
                f"- unavailable_ratio: `{st['unavailable_ratio']:.2%}`",
                f"- corr(signal,final_score): `{st['signal_final_score_correlation']:.4f}`",
                f"- support_selected_ratio: `{st['support_selected_ratio']:.2%}`",
                f"- oppose_but_selected_ratio: `{st['oppose_but_selected_ratio']:.2%}`",
                "",
            ]
        )
    lines.append(f"- top_contributor_distribution: `{payload.get('top_contributor_distribution', {})}`")
    lines.append(f"- degraded_unavailable_counts: `{payload.get('degraded_unavailable_counts', {})}`")
    mp.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-agent attribution report")
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    payload = build_agent_attribution(args.experiment_id)
    write_outputs(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
