from __future__ import annotations

import json

from scripts.compare_runs import compare_manifests


def test_compare_runs_smoke(tmp_path):
    a = {
        "job_id": "a1",
        "pick_date": "2026-02-27",
        "selection_mode": "model",
        "data_quality_mode": "research",
        "degraded_mode": False,
        "degraded_datasets": [],
        "picks": [
            {"stock_id": "2330", "rank": 1, "score": 0.9},
            {"stock_id": "2317", "rank": 2, "score": 0.8},
            {"stock_id": "2454", "rank": 3, "score": 0.7},
        ],
    }
    b = {
        "job_id": "b1",
        "pick_date": "2026-02-28",
        "selection_mode": "multi_agent",
        "data_quality_mode": "research",
        "degraded_mode": True,
        "degraded_datasets": ["raw_institutional"],
        "picks": [
            {"stock_id": "2330", "rank": 1, "score": 0.88},
            {"stock_id": "2303", "rank": 2, "score": 0.79},
            {"stock_id": "2454", "rank": 3, "score": 0.71},
        ],
    }
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    pa.write_text(json.dumps(a), encoding="utf-8")
    pb.write_text(json.dumps(b), encoding="utf-8")

    result = compare_manifests(a, b)
    assert "topn_overlap_10" in result
    assert "rank_correlation_spearman" in result
    assert result["degraded_mode_b"] is True
