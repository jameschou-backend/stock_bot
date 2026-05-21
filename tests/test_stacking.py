"""Stacking ensemble 單元測試（合成資料，無 DB）。

驗證：
- 三個 base model 都能訓練
- predict 回傳 0~1 percentile（rank-averaged）
- by_group 做 per-group rank
- model correlation matrix 形狀正確
- 缺特徵 / 空 base models 報錯
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skills.stacking import (
    StackingEnsemble,
    _rank_to_percentile,
    base_model_correlation,
    train_stacking_ensemble,
)


def _synth_data(n=600, n_features=8, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 1, size=(n, n_features)),
                     columns=[f"f{i}" for i in range(n_features)])
    # y 與 f0 + f1 弱相關
    y = 0.5 * X["f0"] + 0.3 * X["f1"] + rng.normal(0, 0.5, n)
    return X, y.to_numpy()


# ──────────────────────────────────────────────
# Rank percentile helper
# ──────────────────────────────────────────────

class TestRankPercentile:
    def test_global_rank(self):
        scores = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        pct = _rank_to_percentile(scores)
        # 排序 [1, 2, 3, 4, 5] 對應 percentile [0, 0.25, 0.5, 0.75, 1.0]
        # rankdata 給 [1, 5, 3, 2, 4] → percentile [0, 1.0, 0.5, 0.25, 0.75]
        expected = np.array([0.0, 1.0, 0.5, 0.25, 0.75])
        np.testing.assert_allclose(pct, expected, atol=1e-9)

    def test_per_group_rank(self):
        scores = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        groups = np.array(["A", "A", "B", "B", "B"])
        pct = _rank_to_percentile(scores, by_group=groups)
        # A 組 (1, 5): rank → 0.5, 1.0
        # B 組 (3, 2, 4): rank → 0.667, 0.333, 1.0
        assert 0 <= pct.min() <= pct.max() <= 1
        # A 組內 5 比 1 大
        assert pct[1] > pct[0]
        # B 組內 4 最大
        assert pct[4] >= pct[2] and pct[4] >= pct[3]

    def test_singleton(self):
        pct = _rank_to_percentile(np.array([1.0]))
        assert pct[0] == 0.0


# ──────────────────────────────────────────────
# train_stacking_ensemble
# ──────────────────────────────────────────────

class TestTrainStackingEnsemble:
    def test_trains_three_models(self):
        X, y = _synth_data(n=600, seed=42)
        val_X, val_y = _synth_data(n=200, seed=43)
        ens = train_stacking_ensemble(X, y, val_X, val_y)
        assert isinstance(ens, StackingEnsemble)
        # 至少 LightGBM 應該成功；另兩個若 import OK 也應該訓練
        assert "lightgbm" in ens.engines_used
        assert ens.feature_names == [f"f{i}" for i in range(8)]
        assert ens.n_samples == 600

    def test_predict_in_unit_interval(self):
        X, y = _synth_data(seed=1)
        val_X, val_y = _synth_data(n=200, seed=2)
        ens = train_stacking_ensemble(X, y, val_X, val_y)
        pred = ens.predict(val_X)
        assert pred.shape == (len(val_X),)
        assert (pred >= 0).all() and (pred <= 1).all()

    def test_predict_with_groups(self):
        X, y = _synth_data(seed=3)
        val_X, _ = _synth_data(n=200, seed=4)
        ens = train_stacking_ensemble(X, y, X, y)
        groups = np.array(["A"] * 100 + ["B"] * 100)
        pred = ens.predict(val_X, by_group=groups)
        # 每組內 rank 應該 0~1
        for g in np.unique(groups):
            sub = pred[groups == g]
            assert sub.min() >= 0 and sub.max() <= 1

    def test_disable_two_models(self):
        X, y = _synth_data(n=400, seed=5)
        val_X, val_y = _synth_data(n=100, seed=6)
        ens = train_stacking_ensemble(
            X, y, val_X, val_y,
            use_xgboost=False, use_catboost=False,
        )
        assert ens.engines_used == ["lightgbm"]
        assert len(ens.base_models) == 1

    def test_predict_rejects_missing_features(self):
        X, y = _synth_data(seed=7)
        ens = train_stacking_ensemble(X, y, X, y)
        bad_X = X.drop(columns=["f0", "f1"])
        with pytest.raises(ValueError, match="缺欄位"):
            ens.predict(bad_X)


# ──────────────────────────────────────────────
# Correlation diagnostic
# ──────────────────────────────────────────────

class TestBaseModelCorrelation:
    def test_correlation_matrix_shape(self):
        X, y = _synth_data(n=500, seed=10)
        val_X, val_y = _synth_data(n=200, seed=11)
        ens = train_stacking_ensemble(X, y, val_X, val_y)
        corr = base_model_correlation(ens, val_X)
        n = len(ens.engines_used)
        assert corr.shape == (n, n)
        # 對角線為 1
        np.testing.assert_allclose(np.diag(corr.values), [1.0] * n)
        # 對稱
        np.testing.assert_allclose(corr.values, corr.values.T)
        # 三個 GBDT 對相同資料的預測通常相關性 > 0.5
        if n >= 2:
            off_diag = corr.values[np.triu_indices(n, k=1)]
            assert (off_diag > 0).all(), f"base models 應正相關 (got {off_diag})"
