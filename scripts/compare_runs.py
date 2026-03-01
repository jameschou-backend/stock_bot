#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _manifest_path_from_job_id(job_id: str) -> Path:
    return ARTIFACTS_DIR / f"run_manifest_daily_pick_{job_id}.json"


def _load_manifest(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_df(manifest: Dict) -> pd.DataFrame:
    rows = manifest.get("picks", [])
    if not rows:
        return pd.DataFrame(columns=["stock_id", "rank", "score"])
    df = pd.DataFrame(rows)
    df["stock_id"] = df["stock_id"].astype(str)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df


def compare_manifests(a: Dict, b: Dict) -> Dict[str, object]:
    dfa = _to_df(a)
    dfb = _to_df(b)
    set_a = set(dfa["stock_id"].tolist())
    set_b = set(dfb["stock_id"].tolist())

    def overlap_at(k: int) -> float:
        ka = set(dfa.sort_values("rank").head(k)["stock_id"].tolist())
        kb = set(dfb.sort_values("rank").head(k)["stock_id"].tolist())
        if k <= 0:
            return 0.0
        return len(ka & kb) / float(k)

    merged_rank = dfa[["stock_id", "rank"]].merge(dfb[["stock_id", "rank"]], on="stock_id", how="inner", suffixes=("_a", "_b"))
    merged_score = dfa[["stock_id", "score"]].merge(dfb[["stock_id", "score"]], on="stock_id", how="inner", suffixes=("_a", "_b"))
    rank_corr = float(merged_rank["rank_a"].corr(merged_rank["rank_b"], method="spearman")) if len(merged_rank) >= 2 else 0.0
    score_corr = float(merged_score["score_a"].corr(merged_score["score_b"], method="pearson")) if len(merged_score) >= 2 else 0.0

    return {
        "job_a": a.get("job_id"),
        "job_b": b.get("job_id"),
        "pick_date_a": a.get("pick_date"),
        "pick_date_b": b.get("pick_date"),
        "selection_mode_a": a.get("selection_mode"),
        "selection_mode_b": b.get("selection_mode"),
        "topn_overlap_10": overlap_at(10),
        "topn_overlap_20": overlap_at(20),
        "rank_correlation_spearman": rank_corr,
        "score_correlation": score_corr,
        "added_in_b": sorted(set_b - set_a),
        "removed_in_b": sorted(set_a - set_b),
        "dq_mode_a": a.get("data_quality_mode"),
        "dq_mode_b": b.get("data_quality_mode"),
        "degraded_mode_a": a.get("degraded_mode"),
        "degraded_mode_b": b.get("degraded_mode"),
        "degraded_datasets_a": a.get("degraded_datasets", []),
        "degraded_datasets_b": b.get("degraded_datasets", []),
    }


def render_markdown(result: Dict[str, object]) -> str:
    return "\n".join(
        [
            "# Compare Runs",
            "",
            f"- A: `{result['job_a']}` ({result['selection_mode_a']}, pick_date={result['pick_date_a']})",
            f"- B: `{result['job_b']}` ({result['selection_mode_b']}, pick_date={result['pick_date_b']})",
            "",
            "## Core Metrics",
            f"- Top10 overlap: `{result['topn_overlap_10']:.2%}`",
            f"- Top20 overlap: `{result['topn_overlap_20']:.2%}`",
            f"- Rank correlation (Spearman): `{result['rank_correlation_spearman']:.4f}`",
            f"- Score correlation: `{result['score_correlation']:.4f}`",
            "",
            "## Pick Changes",
            f"- Added in B: `{result['added_in_b']}`",
            f"- Removed in B: `{result['removed_in_b']}`",
            "",
            "## DQ / Degraded Differences",
            f"- dq_mode A/B: `{result['dq_mode_a']}` / `{result['dq_mode_b']}`",
            f"- degraded_mode A/B: `{result['degraded_mode_a']}` / `{result['degraded_mode_b']}`",
            f"- degraded_datasets A: `{result['degraded_datasets_a']}`",
            f"- degraded_datasets B: `{result['degraded_datasets_b']}`",
            "",
        ]
    )


def resolve_paths(args) -> Tuple[Path, Path]:
    if args.path_a and args.path_b:
        return Path(args.path_a), Path(args.path_b)
    if args.a and args.b:
        return _manifest_path_from_job_id(args.a), _manifest_path_from_job_id(args.b)
    raise ValueError("Provide either (--a --b) or (--path-a --path-b)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two daily_pick run manifests")
    parser.add_argument("--a", type=str, default=None, help="job_id A")
    parser.add_argument("--b", type=str, default=None, help="job_id B")
    parser.add_argument("--path-a", type=str, default=None, help="manifest path A")
    parser.add_argument("--path-b", type=str, default=None, help="manifest path B")
    args = parser.parse_args()

    path_a, path_b = resolve_paths(args)
    manifest_a = _load_manifest(path_a)
    manifest_b = _load_manifest(path_b)
    result = compare_manifests(manifest_a, manifest_b)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    md = render_markdown(result)
    stem_a = str(result["job_a"])
    stem_b = str(result["job_b"])
    out_path = ARTIFACTS_DIR / f"compare_{stem_a}_{stem_b}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
