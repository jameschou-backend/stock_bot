from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from skills.daily_pick import _research_score_candidates, _should_use_research_fallback


def test_research_mode_degraded_execution():
    config = SimpleNamespace(data_quality_mode="research")
    dq_ctx = {"degraded_mode": True, "degraded_datasets": ["raw_institutional"]}
    assert _should_use_research_fallback(config, dq_ctx) is True

    # institutional 變動不應影響 research fallback 分數（只用 tech + liquidity）
    feature_df = pd.DataFrame(
        {
            "ret_20": [0.1, 0.1],
            "breakout_20": [0.05, 0.05],
            "amt_ratio_20": [1.2, 1.2],
            "foreign_net_20": [1000000, -1000000],
        }
    )
    scores = _research_score_candidates(feature_df)
    assert np.isclose(scores.iloc[0], scores.iloc[1])
