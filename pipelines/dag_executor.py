"""DAG execution engine for the daily pipeline.

並行執行相互獨立的 ingest 節點（I/O 密集型），最大化資料抓取吞吐量，
同時保留嚴格的拓樸順序，確保依賴節點在上游完成後才啟動。

設計原則：
  - 每個並行節點使用獨立的 DB session（避免跨 thread 共用 SQLAlchemy session 問題）
  - 拓樸排序決定執行層級；同層節點以 ThreadPoolExecutor 並行執行
  - 節點失敗：optional=True → 記錄 warning 並繼續；optional=False → 傳播錯誤
  - 若前置依賴失敗，後續節點自動跳過（標記為 SKIPPED）
  - 支援 condition 函式：condition(config) → False 時跳過節點（不算失敗）

使用範例（見 dag_daily.py）：
    from pipelines.dag_executor import DAGNode, DAGExecutor

    nodes = [
        DAGNode("bootstrap",  bootstrap.run,  deps=[]),
        DAGNode("prices",     prices.run,     deps=["bootstrap"]),
        DAGNode("features",   features.run,   deps=["prices"]),
    ]
    executor = DAGExecutor(nodes, max_workers=4)
    executor.run(config)
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED  = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class DAGNode:
    """單一 pipeline 節點定義。

    Attributes:
        name:      唯一識別名稱（用於 deps 引用）。
        runner:    Callable(config, session) → dict，與現有 skill.run() 簽名相容。
        deps:      前置依賴節點名稱清單（空白 = 無依賴）。
        optional:  True → 失敗時僅記錄 warning，不中斷下游節點。
        condition: Callable(config) → bool；回傳 False 時跳過此節點。
                   None 表示永遠執行。
    """
    name: str
    runner: Callable
    deps: List[str] = field(default_factory=list)
    optional: bool = False
    condition: Optional[Callable] = None


@dataclass
class NodeResult:
    name: str
    status: NodeStatus
    elapsed_s: float = 0.0
    result: Optional[dict] = None
    error: Optional[Exception] = None


class DAGExecutor:
    """拓樸 DAG 執行引擎（分層並行）。

    執行邏輯：
        1. 拓樸排序 → 層次（同層可並行）
        2. 逐層執行：同層節點用 ThreadPoolExecutor 並發啟動
        3. 各節點使用獨立 DB session（get_session() in worker thread）
        4. 依賴節點失敗 → 跳過後續節點（SKIPPED）
        5. 所有層完成後回傳 execution summary

    Args:
        nodes:       DAGNode 清單（無需預排序）。
        max_workers: ThreadPoolExecutor 最大線程數（預設 4，I/O 密集型建議 4-8）。
    """

    def __init__(self, nodes: List[DAGNode], max_workers: int = 4) -> None:
        self.nodes: Dict[str, DAGNode] = {n.name: n for n in nodes}
        self.max_workers = max_workers
        self._validate()

    # ── 驗證 ──────────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        """驗證：無循環依賴、所有 deps 均已定義。"""
        for node in self.nodes.values():
            for dep in node.deps:
                if dep not in self.nodes:
                    raise ValueError(
                        f"DAGNode '{node.name}' dep '{dep}' not found in node list."
                    )
        # 循環偵測（DFS）
        visited: Set[str] = set()
        path: Set[str] = set()

        def dfs(name: str) -> None:
            if name in path:
                raise ValueError(f"Cycle detected involving node '{name}'.")
            if name in visited:
                return
            path.add(name)
            for dep in self.nodes[name].deps:
                dfs(dep)
            path.discard(name)
            visited.add(name)

        for name in self.nodes:
            dfs(name)

    # ── 拓樸排序 ──────────────────────────────────────────────────────────────

    def _topological_levels(self) -> List[List[str]]:
        """Kahn's algorithm → 按層次返回節點名稱（同層可並行）。"""
        in_degree: Dict[str, int] = {n: 0 for n in self.nodes}
        dependents: Dict[str, List[str]] = defaultdict(list)
        for node in self.nodes.values():
            for dep in node.deps:
                in_degree[node.name] += 1
                dependents[dep].append(node.name)

        queue: deque[str] = deque(n for n, d in in_degree.items() if d == 0)
        levels: List[List[str]] = []

        while queue:
            level = list(queue)
            queue.clear()
            levels.append(level)
            next_candidates: Set[str] = set()
            for name in level:
                for child in dependents[name]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_candidates.add(child)
            queue.extend(sorted(next_candidates))  # sorted for deterministic order

        return levels

    # ── 主執行入口 ─────────────────────────────────────────────────────────────

    def run(self, config) -> Dict[str, NodeResult]:
        """執行整個 DAG。

        Args:
            config: pipeline config 物件（傳入各節點 runner）。

        Returns:
            {node_name: NodeResult} 字典，包含每個節點的執行狀態。

        Raises:
            RuntimeError: 若有非 optional 節點失敗。
        """
        from app.db import get_session

        levels = self._topological_levels()
        results: Dict[str, NodeResult] = {}
        failed_nodes: Set[str] = set()  # 記錄失敗/跳過節點（用於下游 skip 傳播）

        logger.info(
            "[DAGExecutor] 開始執行：%d 個節點，%d 層，max_workers=%d",
            len(self.nodes),
            len(levels),
            self.max_workers,
        )

        for level_idx, level in enumerate(levels):
            logger.info(
                "[DAGExecutor] Level %d/%d: %s",
                level_idx + 1,
                len(levels),
                level,
            )

            # ── 篩選本層需要執行的節點 ──
            to_run: List[str] = []
            for name in level:
                node = self.nodes[name]

                # 若任意依賴失敗/跳過，本節點也跳過
                deps_failed = any(dep in failed_nodes for dep in node.deps)
                if deps_failed:
                    results[name] = NodeResult(
                        name=name,
                        status=NodeStatus.SKIPPED,
                        error=RuntimeError(
                            "Skipped: dependency failed — "
                            + ", ".join(d for d in node.deps if d in failed_nodes)
                        ),
                    )
                    logger.warning("[DAGExecutor] SKIPPED: %s (dep failed)", name)
                    failed_nodes.add(name)
                    continue

                # condition 函式檢查
                if node.condition is not None and not node.condition(config):
                    results[name] = NodeResult(name=name, status=NodeStatus.SKIPPED)
                    logger.info("[DAGExecutor] SKIPPED (condition=False): %s", name)
                    continue

                to_run.append(name)

            if not to_run:
                continue

            # ── 執行本層節點 ──
            if len(to_run) == 1:
                # 單節點：直接在主線程跑，省去 executor overhead
                name = to_run[0]
                result = _run_node_worker(
                    name, self.nodes[name].runner, config, get_session
                )
                results[name] = result
                if result.status == NodeStatus.FAILED and not self.nodes[name].optional:
                    failed_nodes.add(name)
            else:
                # 多節點：ThreadPoolExecutor 並行
                futures: Dict = {}
                with ThreadPoolExecutor(
                    max_workers=min(self.max_workers, len(to_run)),
                    thread_name_prefix="dag-worker",
                ) as executor:
                    for name in to_run:
                        future = executor.submit(
                            _run_node_worker,
                            name,
                            self.nodes[name].runner,
                            config,
                            get_session,
                        )
                        futures[future] = name

                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = NodeResult(
                                name=name,
                                status=NodeStatus.FAILED,
                                error=exc,
                            )
                        results[name] = result
                        if (
                            result.status == NodeStatus.FAILED
                            and not self.nodes[name].optional
                        ):
                            failed_nodes.add(name)

        # ── 執行摘要 ──
        _log_summary(results)

        # 非 optional 失敗則拋出
        hard_failures = [
            name for name, r in results.items()
            if r.status == NodeStatus.FAILED and not self.nodes[name].optional
        ]
        if hard_failures:
            raise RuntimeError(
                f"DAG pipeline 失敗節點：{hard_failures}。"
                "請查看上方 log 確認錯誤詳情。"
            )

        return results


# ── module-level helpers（可在 worker thread 中呼叫，無需 self）────────────────

def _run_node_worker(
    name: str,
    runner: Callable,
    config,
    get_session_fn,
) -> NodeResult:
    """在 worker 線程中執行單一節點（使用獨立 DB session）。

    Args:
        name:           節點名稱（logging 用）。
        runner:         skill.run(config, session) → dict。
        config:         pipeline config 物件。
        get_session_fn: app.db.get_session（context manager factory）。
    """
    t0 = time.perf_counter()
    logger.info("[DAGExecutor] START: %s", name)
    try:
        with get_session_fn() as session:
            result_data = runner(config, session)
        elapsed = time.perf_counter() - t0
        logger.info("[DAGExecutor] SUCCESS: %s (%.1fs)", name, elapsed)
        return NodeResult(
            name=name,
            status=NodeStatus.SUCCESS,
            elapsed_s=elapsed,
            result=result_data if isinstance(result_data, dict) else {},
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error(
            "[DAGExecutor] FAILED: %s (%.1fs) — %s", name, elapsed, exc,
            exc_info=True,
        )
        return NodeResult(
            name=name,
            status=NodeStatus.FAILED,
            elapsed_s=elapsed,
            error=exc,
        )


def _log_summary(results: Dict[str, NodeResult]) -> None:
    total = len(results)
    ok    = sum(1 for r in results.values() if r.status == NodeStatus.SUCCESS)
    skip  = sum(1 for r in results.values() if r.status == NodeStatus.SKIPPED)
    fail  = sum(1 for r in results.values() if r.status == NodeStatus.FAILED)
    logger.info(
        "[DAGExecutor] 完成：total=%d  success=%d  skipped=%d  failed=%d",
        total, ok, skip, fail,
    )
    _icons = {
        NodeStatus.SUCCESS: "✅",
        NodeStatus.FAILED:  "❌",
        NodeStatus.SKIPPED: "⚠️",
        NodeStatus.RUNNING: "⏳",
        NodeStatus.PENDING: "…",
    }
    for name, r in results.items():
        logger.info(
            "  %s %-35s %.1fs",
            _icons.get(r.status, "?"),
            name,
            r.elapsed_s,
        )
