from __future__ import annotations

import numpy as np

from skills import train_ranker


def test_compute_validation_metrics_contains_required_keys():
    val_dates = np.array(
        [
            "2026-01-01",
            "2026-01-01",
            "2026-01-02",
            "2026-01-02",
        ]
    )
    val_y = np.array([0.03, -0.01, 0.02, 0.01], dtype=float)
    preds = np.array([0.8, 0.1, 0.6, 0.3], dtype=float)

    metrics = train_ranker._compute_validation_metrics(
        val_dates=val_dates,
        val_y=val_y,
        preds=preds,
        topk_list=[10, 20],
    )

    assert metrics["v"] == 1
    assert "ic_spearman" in metrics
    assert "topk" in metrics
    assert "hitrate" in metrics
    assert "pred_stats" in metrics
    assert "k10" in metrics["topk"]
    assert "k20" in metrics["topk"]
    assert "k10" in metrics["hitrate"]
    assert "k20" in metrics["hitrate"]

    pred_stats = metrics["pred_stats"]
    for key in ["mean", "std", "min", "max"]:
        assert key in pred_stats
