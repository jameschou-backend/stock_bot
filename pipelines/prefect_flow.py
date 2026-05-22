"""Stage 9.3: Prefect-orchestrated daily pipeline.

把既有 `pipelines/daily_pipeline.py` 的 17 個 step 包成 5 個 @task 邏輯群組，
保留 run_daily_pipeline() 向後相容。

設計重點：
  1. **不重寫 skill 邏輯**：每個 task 呼叫既有 skill module.run()，不複製程式碼
  2. **logical grouping**：避免 17 個過細 task；按失敗復原策略分 5 群
  3. **retry policy**：core_ingest=3 重試（外部 API 暫時失敗常見）；其他=1
  4. **optional 群組失敗不阻斷主流程**（與既有 _run_optional_skill 對齊）
  5. **fail-fast checkpoint**：build 群組內含 prices/features/labels 三個 assertion
  6. **與 ai_assist 共存**：既有 daily_pipeline.run_skill 內已有 AI 失敗分析，
     prefect retry 在那之前；retry 用盡後 ai_assist 才觸發

使用：
    python -m pipelines.prefect_flow            # 本地直接跑（無 server）
    prefect server start                        # 啟動 Web UI (port 4200)
    prefect deploy ...                          # 部署 scheduled flow

注意：與 `make pipeline` 並存。`make pipeline` 走原 run_daily_pipeline()，
Prefect 走 stockbot_daily_flow()，兩者呼叫相同的底層 skill。
"""
from __future__ import annotations

import logging
from typing import Optional

from prefect import flow, get_run_logger, task
from prefect.tasks import exponential_backoff

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _run_skill_safe(skill_module_name: str, optional: bool = False) -> str:
    """執行 skill module 的 .run(config, session)，回傳 'ok' / 'skipped' / 'failed'。

    Args:
        skill_module_name: 例如 "skills.ingest_prices"
        optional: True 時失敗不 raise（仍 log）

    Returns:
        狀態字串供下游 task 串接 / log
    """
    import importlib
    from app.config import load_config
    from app.db import get_session

    pf_log = get_run_logger()
    try:
        mod = importlib.import_module(skill_module_name)
        with get_session() as session:
            mod.run(load_config(), session)
        pf_log.info(f"[{skill_module_name}] ok")
        return "ok"
    except Exception as exc:
        if optional:
            pf_log.warning(f"[{skill_module_name}] optional failed: {exc}")
            return "failed"
        pf_log.error(f"[{skill_module_name}] failed: {exc}")
        raise


def _check_prices_exist(min_rows: int = 100) -> bool:
    """近 7 天 raw_prices 至少 N 筆（與 daily_pipeline._check_prices_exist 對齊）。"""
    from datetime import timedelta
    from sqlalchemy import func, select
    from app.db import get_session
    from app.models import RawPrice

    with get_session() as session:
        from datetime import date
        cutoff = date.today() - timedelta(days=7)
        count = session.execute(
            select(func.count()).select_from(RawPrice).where(RawPrice.trading_date >= cutoff)
        ).scalar() or 0
    return count >= min_rows


def _check_features_exist(min_rows: int = 50) -> bool:
    from datetime import timedelta, date
    from sqlalchemy import func, select
    from app.db import get_session
    from app.models import Feature

    with get_session() as session:
        cutoff = date.today() - timedelta(days=7)
        count = session.execute(
            select(func.count()).select_from(Feature).where(Feature.trading_date >= cutoff)
        ).scalar() or 0
    return count >= min_rows


def _check_labels_exist(min_rows: int = 50) -> bool:
    from datetime import timedelta, date
    from sqlalchemy import func, select
    from app.db import get_session
    from app.models import Label

    with get_session() as session:
        cutoff = date.today() - timedelta(days=30)
        count = session.execute(
            select(func.count()).select_from(Label).where(Label.trading_date >= cutoff)
        ).scalar() or 0
    return count >= min_rows


# ──────────────────────────────────────────────────────────────
# Tasks（5 個邏輯群組）
# ──────────────────────────────────────────────────────────────

@task(
    name="core_ingest",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=30),
    retry_jitter_factor=0.5,
)
def task_core_ingest() -> dict:
    """必要 ingest：bootstrap / stock_master / calendar / prices / inst / corp_actions。

    任一失敗會 retry 3 次（30s/60s/120s backoff）。重試完仍失敗 → raise。
    """
    pf_log = get_run_logger()
    pf_log.info("=== core_ingest 開始 ===")
    statuses = {}
    for skill in [
        "skills.bootstrap_history",
        "skills.ingest_stock_master",
        "skills.ingest_trading_calendar",
        "skills.ingest_prices",
        "skills.ingest_institutional",
        "skills.ingest_corporate_actions",
    ]:
        statuses[skill] = _run_skill_safe(skill, optional=False)
    pf_log.info("=== core_ingest 完成 ===")
    return statuses


@task(name="optional_ingest", retries=1, retry_delay_seconds=30)
def task_optional_ingest(_upstream: Optional[dict] = None) -> dict:
    """選用 ingest：margin/fund/theme/gov_bank/fear_greed/per/(+sponsor 5)。

    每個 skill 獨立 try/except，單一失敗不影響其他；整 task 不 raise。
    """
    pf_log = get_run_logger()
    pf_log.info("=== optional_ingest 開始 ===")
    statuses = {}
    base_optional = [
        "skills.ingest_margin_short",
        "skills.ingest_fundamental",
        "skills.ingest_theme_flow",
        "skills.ingest_gov_bank",
        "skills.ingest_fear_greed",
        "skills.ingest_per",
    ]
    for skill in base_optional:
        statuses[skill] = _run_skill_safe(skill, optional=True)

    # Sponsor per-stock 重型 ingest（受 SPONSOR_INGEST=off 控制）
    import os
    if os.getenv("SPONSOR_INGEST", "on").lower() != "off":
        for skill in [
            "skills.ingest_broker_trades",
            "skills.ingest_holding_dist",
            "skills.ingest_kbar_features",
            "skills.ingest_securities_lending",
            "skills.ingest_quarterly_fundamental",
        ]:
            statuses[skill] = _run_skill_safe(skill, optional=True)
    else:
        pf_log.info("[skip-sponsor-ingest] SPONSOR_INGEST=off")
    pf_log.info("=== optional_ingest 完成 ===")
    return statuses


@task(name="build_artifacts", retries=1, retry_delay_seconds=60)
def task_build_artifacts(_upstream: Optional[dict] = None, skip_ingest: bool = False) -> dict:
    """資料品質檢查 + features + labels（含 3 個 checkpoint）。"""
    pf_log = get_run_logger()
    pf_log.info("=== build_artifacts 開始 ===")
    statuses = {}

    statuses["data_quality"] = _run_skill_safe("skills.data_quality", optional=False)

    if not skip_ingest and not _check_prices_exist():
        raise RuntimeError(
            "[checkpoint] ingest_prices 後 raw_prices 近 7 天 < 100 筆 — FinMind API 異常?")

    statuses["build_features"] = _run_skill_safe("skills.build_features", optional=False)
    if not _check_features_exist():
        raise RuntimeError("[checkpoint] build_features 後近 7 天 < 50 筆")

    statuses["build_labels"] = _run_skill_safe("skills.build_labels", optional=False)
    if not _check_labels_exist():
        raise RuntimeError("[checkpoint] build_labels 後近 30 天 < 50 筆")

    pf_log.info("=== build_artifacts 完成 ===")
    return statuses


@task(name="train", retries=0)
def task_train(_upstream: Optional[dict] = None) -> dict:
    """條件式重訓模型（依 config.train.frequency_days 決定）。"""
    pf_log = get_run_logger()
    from app.config import load_config

    from pipelines.daily_pipeline import _should_train
    cfg = load_config()
    if not _should_train(cfg):
        pf_log.info("[train] 跳過（未到重訓週期）")
        return {"trained": False}
    pf_log.info("=== train 開始 ===")
    _run_skill_safe("skills.train_ranker", optional=False)
    pf_log.info("=== train 完成 ===")
    return {"trained": True}


@task(name="pick_and_report", retries=1, retry_delay_seconds=30)
def task_pick_and_report(_upstream: Optional[dict] = None) -> dict:
    """每日選股 + 出報表（一旦 build_artifacts 完成必跑）。"""
    pf_log = get_run_logger()
    pf_log.info("=== pick_and_report 開始 ===")
    statuses = {
        "daily_pick": _run_skill_safe("skills.daily_pick", optional=False),
        "export_report": _run_skill_safe("skills.export_report", optional=True),
    }
    pf_log.info("=== pick_and_report 完成 ===")
    return statuses


# ──────────────────────────────────────────────────────────────
# Flow
# ──────────────────────────────────────────────────────────────

@flow(name="stockbot_daily")
def stockbot_daily_flow(skip_ingest: bool = False) -> dict:
    """Stockbot 每日 pipeline（Prefect 編排版）。

    與 pipelines.daily_pipeline.run_daily_pipeline() 邏輯等價但加入：
      - core_ingest retry 3x（30s/60s/120s exponential backoff）
      - optional_ingest 失敗不阻斷下游
      - 3 個 checkpoint（prices / features / labels）失敗時 fail-fast
      - 自動 Web UI 進度追蹤（如有 prefect server）

    Args:
        skip_ingest: True 時跳過所有 ingest，直接 build + report
    """
    pf_log = get_run_logger()
    pf_log.info(f"stockbot_daily_flow start (skip_ingest={skip_ingest})")

    if not skip_ingest:
        core_status = task_core_ingest()
        opt_status = task_optional_ingest(core_status)
        upstream = {"core": core_status, "optional": opt_status}
    else:
        upstream = {"core": "skipped", "optional": "skipped"}

    build_status = task_build_artifacts(upstream, skip_ingest=skip_ingest)
    train_status = task_train(build_status)
    pick_status = task_pick_and_report(train_status)

    pf_log.info("stockbot_daily_flow done")
    return {
        "core_ingest": upstream.get("core"),
        "optional_ingest": upstream.get("optional"),
        "build": build_status,
        "train": train_status,
        "pick_report": pick_status,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip-ingest", action="store_true")
    args = p.parse_args()
    stockbot_daily_flow(skip_ingest=args.skip_ingest)
