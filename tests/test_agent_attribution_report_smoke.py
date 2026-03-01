from __future__ import annotations

import json

from scripts import agent_attribution_report


def test_agent_attribution_report_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_attribution_report, "ARTIFACTS_DIR", tmp_path)
    exp_id = "exp_smoke"
    rows = [
        {
            "experiment_id": exp_id,
            "date": "2026-01-01",
            "stock_id": "2330",
            "rank": 1,
            "score": 0.8,
            "degraded_mode": False,
            "reason_json": {
                "_selection_meta": {"weights_used": {"tech": 0.5, "flow": 0.5}},
                "agents": {
                    "tech": {"signal": 2, "confidence": 0.9, "unavailable": False},
                    "flow": {"signal": 1, "confidence": 0.7, "unavailable": False},
                    "margin": {"signal": 0, "confidence": 0.0, "unavailable": True},
                    "fund": {"signal": 0, "confidence": 0.0, "unavailable": True},
                    "theme": {"signal": 0, "confidence": 0.0, "unavailable": True},
                },
            },
        }
    ]
    (tmp_path / f"experiment_picks_{exp_id}.json").write_text(json.dumps(rows), encoding="utf-8")
    payload = agent_attribution_report.build_agent_attribution(exp_id)
    agent_attribution_report.write_outputs(payload)
    assert "agent_stats" in payload
    assert (tmp_path / f"agent_attribution_{exp_id}.json").exists()
    assert (tmp_path / f"agent_attribution_{exp_id}.md").exists()
