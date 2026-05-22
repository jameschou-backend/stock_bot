"""Stage 9.2 Optuna search 基本檢查（不跑真 backtest）。"""
from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest


def test_optuna_importable():
    """optuna 套件可正常 import。"""
    optuna = importlib.import_module("optuna")
    assert optuna is not None


def test_search_script_importable():
    """scripts.optuna_search 模組可 import 不爆。"""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    mod = importlib.import_module("scripts.optuna_search")
    assert hasattr(mod, "objective")
    assert hasattr(mod, "DEFAULT_SEARCH_DIMS")
    assert "topn" in mod.DEFAULT_SEARCH_DIMS
    assert "vol_target_pct" in mod.DEFAULT_SEARCH_DIMS


def test_search_dims_have_valid_ranges():
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    mod = importlib.import_module("scripts.optuna_search")
    dims = mod.DEFAULT_SEARCH_DIMS
    assert all(isinstance(v, list) and len(v) >= 2 for v in dims.values()
               if isinstance(v, list))
    # min_avg_turnover 是 tuple (low, high)（單位：億元）
    lo, hi = dims["min_avg_turnover"]
    assert lo < hi
    assert 0 < lo < hi < 10  # sanity: 億元範圍合理


def test_objective_signature():
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    mod = importlib.import_module("scripts.optuna_search")
    import inspect
    sig = inspect.signature(mod.objective)
    params = list(sig.parameters.keys())
    assert "trial" in params
    assert "months" in params
    assert "mlflow_experiment" in params
