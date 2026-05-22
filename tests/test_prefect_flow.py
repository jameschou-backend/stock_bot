"""Stage 9.3 Prefect flow 結構測試（不跑實際 daily pipeline）。"""
from __future__ import annotations

import importlib
import inspect

import pytest


def test_prefect_importable():
    prefect = importlib.import_module("prefect")
    assert hasattr(prefect, "flow")
    assert hasattr(prefect, "task")


def test_flow_module_importable():
    mod = importlib.import_module("pipelines.prefect_flow")
    assert hasattr(mod, "stockbot_daily_flow")
    assert hasattr(mod, "task_core_ingest")
    assert hasattr(mod, "task_optional_ingest")
    assert hasattr(mod, "task_build_artifacts")
    assert hasattr(mod, "task_train")
    assert hasattr(mod, "task_pick_and_report")


def test_flow_callable_and_signature():
    from pipelines.prefect_flow import stockbot_daily_flow
    # Prefect Flow 物件有 .fn 屬性指向原 function
    sig = inspect.signature(stockbot_daily_flow)
    assert "skip_ingest" in sig.parameters
    assert sig.parameters["skip_ingest"].default is False


def test_core_ingest_retry_config():
    """core_ingest 應該有 retries=3。"""
    from pipelines.prefect_flow import task_core_ingest
    # Prefect Task 物件有 .retries
    assert task_core_ingest.retries == 3


def test_optional_ingest_retry_config():
    from pipelines.prefect_flow import task_optional_ingest
    assert task_optional_ingest.retries == 1


def test_train_no_retries():
    """train 失敗不重試（避免 wasted compute）。"""
    from pipelines.prefect_flow import task_train
    assert task_train.retries == 0


def test_helper_check_functions_exist():
    from pipelines import prefect_flow as pf
    assert callable(pf._check_prices_exist)
    assert callable(pf._check_features_exist)
    assert callable(pf._check_labels_exist)
    assert callable(pf._run_skill_safe)
