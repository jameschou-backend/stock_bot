"""每日 Pipeline DAG 定義。

將 daily_pipeline.py 的線性流程重組為 DAG，
讓相互獨立的 ingest 節點並行執行，縮短端到端時間。

執行時間估計（I/O 密集型節點並行後）：
    舊（線性）: ~14 min
    新（DAG）:  ~9-10 min（ingest 層並行 → 節省 ~4 min）

DAG 結構：
    Layer 1: bootstrap_history
    Layer 2: ingest_stock_master
    Layer 3: ingest_trading_calendar
    Layer 4: ingest_prices ∥ ingest_institutional ∥ ingest_corporate_actions
             ∥ ingest_margin_short ∥ ingest_fundamental ∥ ingest_theme_flow
    Layer 5: data_quality_check
    Layer 6: build_features ∥ build_labels
    Layer 7: train_ranker（conditional: 週一或 force_train）
    Layer 8: daily_pick

注意事項：
    - ingest_margin_short / ingest_fundamental / ingest_theme_flow 標記為 optional=True
    - train_ranker 使用 condition=_should_train 決定是否執行
    - 各節點使用獨立 DB session（DAGExecutor 保證）
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pipelines.dag_executor import DAGExecutor, DAGNode


def _should_train(config) -> bool:
    """True if model training should run today."""
    if getattr(config, "force_train", False):
        return True
    from app.db import get_session
    from app.models import ModelVersion

    with get_session() as session:
        latest = session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).first()
        if latest is None:
            return True
    today = datetime.now(ZoneInfo(config.tz)).date()
    return today.weekday() == 0  # Monday


def build_dag(skip_ingest: bool = False) -> DAGExecutor:
    """Build and return the daily pipeline DAGExecutor.

    Args:
        skip_ingest: True → 跳過所有 ingest 節點，直接從 data_quality 開始
                     （資料已是最新時使用，等同 --skip-ingest 旗標）。
    """
    # ── 延遲匯入（避免頂層 import 時拉入所有依賴）──────────────────────────────
    from skills import (
        bootstrap_history,
        build_features,
        build_labels,
        daily_pick,
        data_quality,
        ingest_corporate_actions,
        ingest_fundamental,
        ingest_institutional,
        ingest_margin_short,
        ingest_prices,
        ingest_stock_master,
        ingest_theme_flow,
        ingest_trading_calendar,
        train_ranker,
    )

    if skip_ingest:
        # 跳過 ingest：從 data_quality 開始，無上游依賴
        nodes = [
            DAGNode("data_quality",  data_quality.run,  deps=[]),
            DAGNode("build_features", build_features.run, deps=["data_quality"]),
            DAGNode("build_labels",   build_labels.run,   deps=["data_quality"]),
            DAGNode(
                "train_ranker",
                train_ranker.run,
                deps=["build_features", "build_labels"],
                condition=_should_train,
            ),
            DAGNode(
                "daily_pick",
                daily_pick.run,
                deps=["build_features", "train_ranker"],
            ),
        ]
    else:
        # 完整 DAG（含 ingest 層）
        nodes = [
            # ── Layer 1：初始化歷史資料（必須最先，只有第一次才實際執行）
            DAGNode("bootstrap_history",  bootstrap_history.run,  deps=[]),

            # ── Layer 2：股票基本資料
            DAGNode("ingest_stock_master", ingest_stock_master.run, deps=["bootstrap_history"]),

            # ── Layer 3：交易日曆
            DAGNode(
                "ingest_trading_calendar",
                ingest_trading_calendar.run,
                deps=["ingest_stock_master"],
            ),

            # ── Layer 4：各類資料並行 ingest（皆依賴 trading_calendar 確保日期基礎存在）
            DAGNode("ingest_prices",           ingest_prices.run,           deps=["ingest_trading_calendar"]),
            DAGNode("ingest_institutional",    ingest_institutional.run,    deps=["ingest_trading_calendar"]),
            DAGNode("ingest_corporate_actions",ingest_corporate_actions.run, deps=["ingest_trading_calendar"]),
            DAGNode("ingest_margin_short",     ingest_margin_short.run,     deps=["ingest_trading_calendar"], optional=True),
            DAGNode("ingest_fundamental",      ingest_fundamental.run,      deps=["ingest_trading_calendar"], optional=True),
            DAGNode("ingest_theme_flow",       ingest_theme_flow.run,       deps=["ingest_trading_calendar"], optional=True),

            # ── Layer 5：資料品質（所有 ingest 完成後才執行）
            DAGNode(
                "data_quality",
                data_quality.run,
                deps=[
                    "ingest_prices",
                    "ingest_institutional",
                    "ingest_corporate_actions",
                    "ingest_margin_short",
                    "ingest_fundamental",
                    "ingest_theme_flow",
                ],
            ),

            # ── Layer 6：特徵與標籤可並行（互不依賴）
            DAGNode("build_features", build_features.run, deps=["data_quality"]),
            DAGNode("build_labels",   build_labels.run,   deps=["data_quality"]),

            # ── Layer 7：訓練（conditional）
            DAGNode(
                "train_ranker",
                train_ranker.run,
                deps=["build_features", "build_labels"],
                condition=_should_train,
            ),

            # ── Layer 8：每日選股
            DAGNode(
                "daily_pick",
                daily_pick.run,
                deps=["build_features", "train_ranker"],
            ),
        ]

    return DAGExecutor(nodes, max_workers=4)
