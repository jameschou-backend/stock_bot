"""Stage 9.1 MLflow tracking tests：確認 opt-in、disabled 時 no-op、import 不破。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from skills import mlflow_tracking as mt


def test_is_available_returns_bool():
    assert isinstance(mt.is_available(), bool)


def test_start_mlflow_run_disabled_yields_none():
    with mt.start_mlflow_run(enabled=False) as run:
        assert run is None


def test_log_helpers_noop_on_none_run():
    """所有 log_* 在 mlflow_run=None 時應 silently no-op，不 raise。"""
    mt.log_params({"a": 1, "b": "x"}, mlflow_run=None)
    mt.log_metrics({"sharpe": 1.5}, mlflow_run=None)
    mt.log_artifact("/nonexistent/path.json", mlflow_run=None)
    mt.log_dict({"k": "v"}, "out.json", mlflow_run=None)
    mt.log_backtest_result({"summary": {}}, mlflow_run=None)
    # 無 assert 失敗 = OK


@pytest.mark.skipif(not mt.is_available(), reason="mlflow 未安裝")
def test_full_run_writes_metrics_to_tempdir(tmp_path):
    """端到端：起 mlflow run 寫指標，確認檔案落地。"""
    import mlflow
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    with mt.start_mlflow_run(
        experiment_name="test_exp",
        run_name="test_run",
        tracking_uri=tracking_uri,
        enabled=True,
    ) as run:
        assert run is not None
        mt.log_params({"alpha": 0.1, "model": "lgb"}, mlflow_run=run)
        mt.log_metrics({"sharpe": 1.5, "mdd": -0.27}, mlflow_run=run)
        # log_backtest_result with synthetic
        result = {
            "summary": {
                "sharpe_ratio": 1.5, "max_drawdown": -0.27,
                "win_rate": 0.45, "total_trades": 100,
            },
            "periods": [{"return": 0.05}, {"return": -0.02}],
        }
        mt.log_backtest_result(result, mlflow_run=run)

    # 驗證 mlruns 目錄真的有東西
    mlruns_dir = tmp_path / "mlruns"
    assert mlruns_dir.exists()
    # mlflow 結構：mlruns/<exp_id>/<run_id>/metrics, params, ...
    files = list(mlruns_dir.rglob("metrics/sharpe"))
    assert len(files) > 0, "sharpe metric 未寫入"


def test_serialize_value_handles_various_types():
    from datetime import date
    assert mt._serialize_value(None) == "None"
    assert mt._serialize_value(42) == 42
    assert mt._serialize_value(3.14) == 3.14
    assert mt._serialize_value("x") == "x"
    assert mt._serialize_value(True) is True
    assert mt._serialize_value(date(2026, 5, 22)) == "2026-05-22"
    assert mt._serialize_value([1, 2, 3]) == "[1, 2, 3]"
