"""Multi-Horizon ensemble 單元測試。"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from skills.multi_horizon import (
    DEFAULT_HORIZONS,
    MultiHorizonEnsemble,
    compute_multi_horizon_labels,
    horizon_model_correlation,
    train_multi_horizon_ensemble,
)


def _make_panel(n_stocks=3, n_days=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    start = date(2024, 1, 1)
    for i in range(n_stocks):
        sid = f"100{i}"
        walk = 100 + np.cumsum(rng.normal(0.05, 1, n_days))
        for d_idx, c in enumerate(walk):
            rows.append({
                "stock_id": sid,
                "trading_date": start + timedelta(days=d_idx),
                "close": float(c),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# compute_multi_horizon_labels
# ──────────────────────────────────────────────

class TestComputeMultiHorizonLabels:
    def test_basic_horizons(self):
        df = _make_panel(n_stocks=2, n_days=30)
        out = compute_multi_horizon_labels(df, horizons=[5, 10, 20])
        for h in (5, 10, 20):
            assert f"future_ret_{h}" in out.columns
        # 每股最後 h 個 row 對應 horizon=h 的 label 應為 NaN
        for sid, g in out.groupby("stock_id"):
            assert g["future_ret_5"].iloc[-5:].isna().all()
            assert g["future_ret_10"].iloc[-10:].isna().all()
            assert g["future_ret_20"].iloc[-20:].isna().all()

    def test_label_formula(self):
        """future_ret_h[t] = close[t+h] / close[t] - 1"""
        df = pd.DataFrame({
            "stock_id": ["A"] * 5,
            "trading_date": [date(2024, 1, i+1) for i in range(5)],
            "close": [100.0, 110, 120, 132, 145.2],
        })
        out = compute_multi_horizon_labels(df, horizons=[1, 2])
        # day0: r1 = 110/100-1 = 0.1; r2 = 120/100-1 = 0.2
        assert abs(out.iloc[0]["future_ret_1"] - 0.1) < 1e-9
        assert abs(out.iloc[0]["future_ret_2"] - 0.2) < 1e-9
        # day3 (close=132): r1 = 145.2/132-1 = 0.1; r2 = NaN (no day5 data)
        assert abs(out.iloc[3]["future_ret_1"] - 0.1) < 1e-9
        assert pd.isna(out.iloc[3]["future_ret_2"])

    def test_zero_close_safe(self):
        df = pd.DataFrame({
            "stock_id": ["A"] * 3,
            "trading_date": [date(2024, 1, i+1) for i in range(3)],
            "close": [0.0, 100, 110],
        })
        out = compute_multi_horizon_labels(df, horizons=[1])
        # close=0 should give NaN return（avoid div-by-zero）
        assert pd.isna(out.iloc[0]["future_ret_1"])

    def test_rejects_missing_columns(self):
        with pytest.raises(ValueError, match="缺欄位"):
            compute_multi_horizon_labels(pd.DataFrame({"stock_id": ["A"]}))

    def test_rejects_empty_horizons(self):
        df = _make_panel(n_days=10)
        with pytest.raises(ValueError, match="horizons"):
            compute_multi_horizon_labels(df, horizons=[])


# ──────────────────────────────────────────────
# train_multi_horizon_ensemble
# ──────────────────────────────────────────────

def _synth_train_val(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 1, (n, 6)), columns=[f"f{i}" for i in range(6)])
    # Labels：4 個 horizons 都跟 f0/f1 弱相關（不同係數）
    label_panel = pd.DataFrame({
        "future_ret_5": 0.5 * X["f0"] + 0.3 * X["f1"] + rng.normal(0, 0.5, n),
        "future_ret_10": 0.3 * X["f0"] + 0.4 * X["f1"] + rng.normal(0, 0.5, n),
        "future_ret_20": 0.2 * X["f0"] + 0.5 * X["f1"] + rng.normal(0, 0.5, n),
        "future_ret_40": 0.1 * X["f0"] + 0.6 * X["f1"] + rng.normal(0, 0.5, n),
    })
    return X, label_panel


class TestTrainMultiHorizon:
    def test_trains_all_horizons(self):
        train_X, train_lp = _synth_train_val(n=500, seed=42)
        val_X, val_lp = _synth_train_val(n=200, seed=43)
        ens = train_multi_horizon_ensemble(
            train_X, train_lp, val_X, val_lp, horizons=[5, 10, 20, 40]
        )
        assert isinstance(ens, MultiHorizonEnsemble)
        assert sorted(ens.horizon_models.keys()) == [5, 10, 20, 40]
        assert ens.feature_names == [f"f{i}" for i in range(6)]

    def test_predict_in_unit_interval(self):
        train_X, train_lp = _synth_train_val(seed=1)
        val_X, val_lp = _synth_train_val(n=200, seed=2)
        ens = train_multi_horizon_ensemble(train_X, train_lp, val_X, val_lp,
                                            horizons=[5, 10, 20])
        pred = ens.predict(val_X)
        assert pred.shape == (len(val_X),)
        assert (pred >= 0).all() and (pred <= 1).all()

    def test_skips_missing_horizon(self):
        """label_panel 缺 future_ret_40 → 該 horizon 跳過。"""
        train_X, train_lp = _synth_train_val(seed=3)
        val_X, val_lp = _synth_train_val(n=200, seed=4)
        # 不含 future_ret_40
        train_lp = train_lp.drop(columns=["future_ret_40"])
        val_lp = val_lp.drop(columns=["future_ret_40"])
        ens = train_multi_horizon_ensemble(train_X, train_lp, val_X, val_lp,
                                            horizons=[5, 10, 20, 40])
        # 40 應該被跳過
        assert 40 not in ens.horizon_models
        assert sorted(ens.horizon_models.keys()) == [5, 10, 20]

    def test_rejects_mismatched_length(self):
        train_X = pd.DataFrame({"f0": [1.0, 2.0, 3.0]})
        train_lp = pd.DataFrame({"future_ret_5": [0.1, 0.2]})  # 長度 2 vs 3
        val_X = pd.DataFrame({"f0": [1.0]})
        val_lp = pd.DataFrame({"future_ret_5": [0.1]})
        with pytest.raises(ValueError, match="長度不一致"):
            train_multi_horizon_ensemble(train_X, train_lp, val_X, val_lp,
                                          horizons=[5])

    def test_correlation_diagnostic(self):
        train_X, train_lp = _synth_train_val(n=500, seed=5)
        val_X, val_lp = _synth_train_val(n=200, seed=6)
        ens = train_multi_horizon_ensemble(train_X, train_lp, val_X, val_lp,
                                            horizons=[5, 10, 20])
        corr = horizon_model_correlation(ens, val_X)
        assert corr.shape == (3, 3)
        # 對角線 1.0
        np.testing.assert_allclose(np.diag(corr.values), [1.0, 1.0, 1.0])
