"""Tests for pipelines/dag_executor.py."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from pipelines.dag_executor import DAGExecutor, DAGNode, NodeStatus


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_runner(name: str, result: dict | None = None, raise_exc: Exception | None = None):
    """Return a runner callable that either succeeds or raises."""
    def runner(config, session):
        if raise_exc is not None:
            raise raise_exc
        return result or {"node": name}
    runner.__name__ = name
    return runner


@contextmanager
def _mock_session():
    yield MagicMock()


def _run(nodes, config=None):
    """Helper: build executor, patch get_session, run."""
    executor = DAGExecutor(nodes, max_workers=2)
    with patch("app.db.get_session", side_effect=_mock_session):
        return executor.run(config or MagicMock())


# ── validation tests ──────────────────────────────────────────────────────────

def test_unknown_dep_raises():
    nodes = [DAGNode("a", _make_runner("a"), deps=["nonexistent"])]
    with pytest.raises(ValueError, match="not found"):
        DAGExecutor(nodes)


def test_cycle_detection():
    nodes = [
        DAGNode("a", _make_runner("a"), deps=["b"]),
        DAGNode("b", _make_runner("b"), deps=["a"]),
    ]
    with pytest.raises(ValueError, match="[Cc]ycle"):
        DAGExecutor(nodes)


# ── topological levels ────────────────────────────────────────────────────────

def test_topological_levels_linear():
    nodes = [
        DAGNode("a", _make_runner("a"), deps=[]),
        DAGNode("b", _make_runner("b"), deps=["a"]),
        DAGNode("c", _make_runner("c"), deps=["b"]),
    ]
    executor = DAGExecutor(nodes)
    levels = executor._topological_levels()
    assert levels == [["a"], ["b"], ["c"]]


def test_topological_levels_parallel():
    nodes = [
        DAGNode("root",  _make_runner("root"),  deps=[]),
        DAGNode("child1",_make_runner("child1"), deps=["root"]),
        DAGNode("child2",_make_runner("child2"), deps=["root"]),
        DAGNode("leaf",  _make_runner("leaf"),  deps=["child1", "child2"]),
    ]
    executor = DAGExecutor(nodes)
    levels = executor._topological_levels()
    assert levels[0] == ["root"]
    assert set(levels[1]) == {"child1", "child2"}
    assert levels[2] == ["leaf"]


# ── execution tests ───────────────────────────────────────────────────────────

def test_all_success():
    nodes = [
        DAGNode("a", _make_runner("a"), deps=[]),
        DAGNode("b", _make_runner("b"), deps=["a"]),
    ]
    results = _run(nodes)
    assert results["a"].status == NodeStatus.SUCCESS
    assert results["b"].status == NodeStatus.SUCCESS


def test_optional_failure_does_not_block_downstream():
    """Optional node failure: downstream node still runs (not skipped).

    Design rationale: optional=True means the node is best-effort.
    When it fails, we don't add it to failed_nodes, so downstream nodes
    that depend on it still proceed (they may handle missing data gracefully).
    This is the correct behavior for truly optional data sources.
    """
    nodes = [
        DAGNode("a", _make_runner("a"), deps=[]),
        DAGNode("optional_b", _make_runner("b", raise_exc=RuntimeError("boom")),
                deps=["a"], optional=True),
        DAGNode("c", _make_runner("c"), deps=["optional_b"]),
    ]
    # optional_b fails → no RuntimeError raised at pipeline level
    # c depends on optional_b, but since optional_b failure doesn't propagate,
    # c still runs (SUCCESS)
    results = _run(nodes)
    assert results["optional_b"].status == NodeStatus.FAILED
    assert results["c"].status == NodeStatus.SUCCESS


def test_optional_failure_sibling_still_runs():
    """When optional node fails, sibling nodes at same level still run."""
    nodes = [
        DAGNode("root", _make_runner("root"), deps=[]),
        DAGNode("optional_a", _make_runner("a", raise_exc=RuntimeError("boom")),
                deps=["root"], optional=True),
        DAGNode("b", _make_runner("b"), deps=["root"]),
    ]
    # b does not depend on optional_a, so it should run fine
    results = _run(nodes)
    assert results["optional_a"].status == NodeStatus.FAILED
    assert results["b"].status == NodeStatus.SUCCESS


def test_required_failure_blocks_downstream_and_raises():
    nodes = [
        DAGNode("a", _make_runner("a"), deps=[]),
        DAGNode("b", _make_runner("b", raise_exc=RuntimeError("hard fail")), deps=["a"]),
        DAGNode("c", _make_runner("c"), deps=["b"]),
    ]
    with pytest.raises(RuntimeError, match="DAG pipeline"):
        _run(nodes)


def test_condition_false_skips_node():
    nodes = [
        DAGNode("a", _make_runner("a"), deps=[]),
        DAGNode("cond_b", _make_runner("b"), deps=["a"], condition=lambda cfg: False),
        DAGNode("c", _make_runner("c"), deps=["cond_b"]),
    ]
    # cond_b skipped; c depends on cond_b — but SKIPPED ≠ FAILED, so c may run
    # Actually in current implementation, SKIPPED propagates like failure for downstream
    results = _run(nodes)
    assert results["cond_b"].status == NodeStatus.SKIPPED


def test_condition_true_runs_node():
    nodes = [
        DAGNode("a", _make_runner("a"), deps=[]),
        DAGNode("b", _make_runner("b"), deps=["a"], condition=lambda cfg: True),
    ]
    results = _run(nodes)
    assert results["b"].status == NodeStatus.SUCCESS


def test_parallel_nodes_all_run():
    execution_order = []

    def make_tracker(name):
        def runner(config, session):
            execution_order.append(name)
            return {}
        return runner

    nodes = [
        DAGNode("root", make_tracker("root"), deps=[]),
        DAGNode("p1",   make_tracker("p1"),   deps=["root"]),
        DAGNode("p2",   make_tracker("p2"),   deps=["root"]),
        DAGNode("p3",   make_tracker("p3"),   deps=["root"]),
    ]
    results = _run(nodes)
    assert results["p1"].status == NodeStatus.SUCCESS
    assert results["p2"].status == NodeStatus.SUCCESS
    assert results["p3"].status == NodeStatus.SUCCESS
    # root must have run before parallel nodes
    assert execution_order[0] == "root"


def test_elapsed_s_recorded():
    nodes = [DAGNode("a", _make_runner("a"), deps=[])]
    results = _run(nodes)
    assert results["a"].elapsed_s >= 0.0


def test_result_dict_stored():
    nodes = [DAGNode("a", _make_runner("a", result={"rows": 42}), deps=[])]
    results = _run(nodes)
    assert results["a"].result == {"rows": 42}
