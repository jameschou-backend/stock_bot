from __future__ import annotations

import json
from pathlib import Path

from scripts import run_experiment_matrix


def test_run_experiment_matrix_smoke(tmp_path, monkeypatch):
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        "\n".join(
            [
                'start_date: "2026-01-01"',
                'end_date: "2026-02-28"',
                "topn: 5",
                "experiments:",
                '  - name: "m1"',
                '    selection_mode: "model"',
                '    data_quality_mode: "research"',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_experiment_matrix, "ARTIFACTS_DIR", tmp_path)

    def _fake_run_experiment(experiment_id, start_date, end_date, cfg, resume=False):
        (tmp_path / f"evaluation_{experiment_id}.json").write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "metrics": {"total_return": 0.1, "cagr": 0.2, "max_drawdown": -0.05, "sharpe": 1.1, "picks_stability": 0.6},
                    "degraded_ratio": 0.1,
                    "number_of_rebalance_dates": 2,
                    "manifest_dir": str(tmp_path / "manifests" / experiment_id),
                    "skipped_periods": [],
                }
            ),
            encoding="utf-8",
        )
        exp_picks = [
            {"date": "2026-01-01", "stock_id": "2330", "rank": 1, "score": 0.5, "reason_json": {"_selection_meta": {"weights_used": {}}}},
            {"date": "2026-02-01", "stock_id": "2317", "rank": 1, "score": 0.4, "reason_json": {"_selection_meta": {"weights_used": {}}}},
        ]
        (tmp_path / f"experiment_picks_{experiment_id}.json").write_text(json.dumps(exp_picks), encoding="utf-8")
        manifest_dir = Path(str(tmp_path / "manifests" / experiment_id))
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / f"run_manifest_{experiment_id}_2026-01-01.json").write_text("{}", encoding="utf-8")
        return {
            "experiment_id": experiment_id,
            "metrics": {"total_return": 0.1, "cagr": 0.2, "max_drawdown": -0.05, "sharpe": 1.1, "picks_stability": 0.6},
            "degraded_ratio": 0.1,
            "number_of_rebalance_dates": 2,
            "manifest_dir": str(manifest_dir),
            "skipped_periods": [],
        }

    monkeypatch.setattr(run_experiment_matrix, "run_experiment", _fake_run_experiment)
    monkeypatch.setattr(run_experiment_matrix, "build_agent_attribution", lambda experiment_id: {"experiment_id": experiment_id, "agent_stats": {}})
    monkeypatch.setattr(run_experiment_matrix, "write_attribution_outputs", lambda payload: None)

    payload = run_experiment_matrix.run_matrix(matrix_path, resume=False)
    assert "experiments" in payload
    assert (tmp_path / "experiment_matrix_summary.json").exists()
    assert (tmp_path / "experiment_matrix_summary.md").exists()
