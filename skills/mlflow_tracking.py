"""Stage 9.1: MLflow tracking helper for run_backtest.

設計原則：
  1. **opt-in 且失敗不破壞 backtest**：mlflow import 或 logging 失敗都 silently skip。
  2. **不污染 backtest core**：mlflow 邏輯集中於此 module，backtest.py 只 call
     `start_mlflow_run(...)` / `log_backtest_result(...)`。
  3. **tracking_uri 預設 file:./mlruns/**：本機 SQLite-backed，無 server 依賴。
  4. **param/metric 分離**：WalkForwardConfig fields → params；summary → metrics。
  5. **artifact 條件式紀錄**：避免大檔案（如 trades_log）灌爆 mlruns/。

使用：
    with start_mlflow_run("baseline_10y", config_dict) as run:
        result = run_backtest(...)
        log_backtest_result(result, mlflow_run=run)

或 CLI：
    python scripts/run_backtest.py --months 120 ... --mlflow-experiment my_exp
"""
from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from datetime import date as _date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


DEFAULT_EXPERIMENT = "stock_bot_backtest"
DEFAULT_TRACKING_URI = "file:./mlruns"


def _git_sha() -> Optional[str]:
    """取得當前 git commit SHA（前 8 碼）。失敗回 None。"""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=Path(__file__).resolve().parent.parent,
        )
        return out.decode().strip()
    except Exception:
        return None


def _serialize_value(v: Any) -> Any:
    """把 dataclass field 序列化成 mlflow 支援的 param 型別（str/int/float/bool）。

    None → "None"（string）；list/tuple/dict → str(v)；其他保持原樣。
    """
    if v is None:
        return "None"
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, _date):
        return v.isoformat()
    if isinstance(v, (list, tuple, dict)):
        return str(v)
    return str(v)


def is_available() -> bool:
    """mlflow 是否可用（套件已裝）。"""
    return _HAS_MLFLOW


@contextlib.contextmanager
def start_mlflow_run(
    experiment_name: str = DEFAULT_EXPERIMENT,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    enabled: bool = True,
):
    """Context manager 啟動 mlflow run。失敗 silently 退化為 no-op。

    Args:
        experiment_name: mlflow experiment 名（自動建立）
        run_name: run 名（預設 timestamp + git sha）
        tracking_uri: 預設 file:./mlruns
        tags: 額外 tag dict
        enabled: False 時 yield None（給 CLI flag 控制）
    Yields:
        active_run 物件，或 None
    """
    if not enabled or not _HAS_MLFLOW:
        yield None
        return

    try:
        mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        sha = _git_sha()
        if run_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{ts}_{sha or 'nogit'}"
        full_tags = {"git_sha": sha or "unknown"}
        if tags:
            full_tags.update(tags)
        with mlflow.start_run(run_name=run_name, tags=full_tags) as run:
            logger.info("[mlflow] started run %s (experiment=%s)", run_name, experiment_name)
            yield run
    except Exception as exc:
        logger.warning("[mlflow] disabled (start failed): %s", exc)
        yield None


def log_params(params: Dict[str, Any], mlflow_run=None) -> None:
    """把參數 dict 寫入 mlflow params。run=None 時 no-op。"""
    if mlflow_run is None or not _HAS_MLFLOW:
        return
    try:
        clean = {}
        for k, v in params.items():
            if k.startswith("_") or callable(v):
                continue
            clean[k] = _serialize_value(v)
        # mlflow log_params 限制 key 長度 250、value 長度 6000
        clean = {k[:250]: str(v)[:6000] for k, v in clean.items()}
        mlflow.log_params(clean)
    except Exception as exc:
        logger.warning("[mlflow] log_params failed: %s", exc)


def log_metrics(metrics: Dict[str, float], mlflow_run=None) -> None:
    """寫入 mlflow metrics；非數字 / NaN 自動跳過。"""
    if mlflow_run is None or not _HAS_MLFLOW:
        return
    try:
        import math
        clean = {}
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except (ValueError, TypeError):
                continue
            if math.isnan(fv) or math.isinf(fv):
                continue
            clean[k[:250]] = fv
        if clean:
            mlflow.log_metrics(clean)
    except Exception as exc:
        logger.warning("[mlflow] log_metrics failed: %s", exc)


def log_artifact(path: str, mlflow_run=None) -> None:
    """把檔案複製到 mlflow run 的 artifacts。檔案不存在會 skip。"""
    if mlflow_run is None or not _HAS_MLFLOW:
        return
    try:
        if not os.path.exists(path):
            logger.debug("[mlflow] artifact missing: %s", path)
            return
        mlflow.log_artifact(path)
    except Exception as exc:
        logger.warning("[mlflow] log_artifact failed for %s: %s", path, exc)


def log_dict(d: Dict[str, Any], artifact_file: str, mlflow_run=None) -> None:
    """把 dict 以 json 格式寫成 artifact。"""
    if mlflow_run is None or not _HAS_MLFLOW:
        return
    try:
        mlflow.log_dict(d, artifact_file)
    except Exception as exc:
        logger.warning("[mlflow] log_dict failed: %s", exc)


def log_backtest_result(result: Dict, mlflow_run=None) -> None:
    """把 run_backtest() 的 result dict 標準化寫入 mlflow。

    Args:
        result: run_backtest 回傳的 dict（含 summary, periods, equity_curve, ...）
        mlflow_run: start_mlflow_run yielded 物件（None = no-op）
    """
    if mlflow_run is None or not _HAS_MLFLOW or not result:
        return
    try:
        summary = result.get("summary") or {}
        # ── metrics ──
        metrics_keys = [
            "cumulative_return", "annual_return", "excess_return",
            "max_drawdown", "sharpe_ratio", "calmar_ratio",
            "win_rate", "profit_factor",
            "total_trades", "stoploss_triggers",
            "benchmark_cumulative_return", "benchmark_annual_return",
        ]
        metrics = {k: summary.get(k) for k in metrics_keys if k in summary}
        # 從 periods 推算累積（如果 summary 缺）
        if metrics.get("cumulative_return") is None and result.get("periods"):
            cum = 1.0
            for p in result["periods"]:
                r = p.get("return")
                if r is not None:
                    cum *= (1 + float(r))
            metrics["cumulative_return"] = cum - 1
        log_metrics(metrics, mlflow_run=mlflow_run)

        # ── 把 summary 本身存成 artifact（給人類閱讀）──
        log_dict(summary, "summary.json", mlflow_run=mlflow_run)

        # ── periods 也存（給跨 run 比較）──
        if result.get("periods"):
            log_dict({"periods": result["periods"]}, "periods.json", mlflow_run=mlflow_run)

        # ── 報表 HTML（如果存在）──
        report_html = "artifacts/backtest_report.html"
        log_artifact(report_html, mlflow_run=mlflow_run)
    except Exception as exc:
        logger.warning("[mlflow] log_backtest_result failed: %s", exc)
