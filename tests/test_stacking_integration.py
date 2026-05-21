"""Stage 6.1 integration test：確認 use_stacking 開關 + invariants 不破。"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from skills.backtest import (
    BacktestPipeline,
    WalkForwardConfig,
    _StackingAdapter,
    run_backtest,
)


class TestStackingConfig:
    def test_default_is_disabled(self):
        wf = WalkForwardConfig()
        assert wf.use_stacking is False
        assert 0.05 <= wf.stacking_val_frac <= 0.5

    def test_run_backtest_wrapper_accepts_kwarg(self):
        sig = inspect.signature(run_backtest)
        assert "use_stacking" in sig.parameters
        assert sig.parameters["use_stacking"].default is False
        assert "stacking_val_frac" in sig.parameters
        assert sig.parameters["stacking_val_frac"].default == 0.20

    def test_pipeline_has_train_method(self):
        assert hasattr(BacktestPipeline, "_train_stacking_for_period")


class TestStackingAdapter:
    def test_adapter_passthrough_non_stacking(self):
        class Dummy:
            def predict(self, X):
                return np.zeros(len(X))
        a = _StackingAdapter(model=Dummy(), feature_names=["a", "b"], is_stacking=False)
        out = a.predict(np.array([[1, 2], [3, 4]]))
        assert out.shape == (2,)

    def test_adapter_wraps_ndarray_to_df_when_stacking(self):
        # 用 mock StackingEnsemble：predict 收 DataFrame，回傳行數的常數
        class MockEnsemble:
            def __init__(self):
                self.last_X = None
            def predict(self, X):
                # 應該是 DataFrame；若不是 backtest 端會壞
                assert isinstance(X, pd.DataFrame), f"got {type(X)}"
                self.last_X = X
                return np.full(len(X), 0.5)
        m = MockEnsemble()
        a = _StackingAdapter(model=m, feature_names=["a", "b", "c"], is_stacking=True)
        out = a.predict(np.array([[1, 2, 3], [4, 5, 6]]))
        assert out.shape == (2,)
        assert m.last_X is not None
        assert list(m.last_X.columns) == ["a", "b", "c"]


class TestStackingLambdarankExclusion:
    """use_stacking + use_lambdarank 互斥（regressor vs ranker target type 不同）。"""

    def test_init_raises_when_both_set(self):
        from unittest.mock import MagicMock
        wf = WalkForwardConfig(use_stacking=True, use_lambdarank=True)
        with pytest.raises(ValueError, match="use_stacking 與 use_lambdarank 互斥"):
            BacktestPipeline(MagicMock(), MagicMock(), wf)
