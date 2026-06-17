"""skills.io_utils 原子寫入與安全讀取測試。

涵蓋 state / portfolio 檔的核心防護：round-trip、建父目錄、無 temp 殘留、
不存在回 default、損毀 fail-loud、損毀 fallback 到 .bak、覆寫。
"""
from __future__ import annotations

import json

import pytest

from skills.io_utils import atomic_write_json, safe_read_json


def test_atomic_write_then_read_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    obj = {"positions": {"2330": {"entry_price": 855.0}}, "last_run_date": "2026-06-13"}
    atomic_write_json(p, obj)
    assert safe_read_json(p) == obj


def test_write_creates_parent_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "state.json"
    atomic_write_json(p, {"x": 1})
    assert p.exists()


def test_no_tmp_files_left_behind(tmp_path):
    p = tmp_path / "state.json"
    atomic_write_json(p, {"x": 1})
    leftover = [f.name for f in tmp_path.iterdir() if f.name != "state.json"]
    assert leftover == []


def test_read_missing_returns_default(tmp_path):
    p = tmp_path / "missing.json"
    assert safe_read_json(p, default={"positions": []}) == {"positions": []}


def test_read_corrupt_raises_when_no_bak(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{ this is not valid json", encoding="utf-8")
    with pytest.raises(ValueError):
        safe_read_json(p, default={})


def test_read_corrupt_falls_back_to_bak(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{ truncated", encoding="utf-8")
    bak = tmp_path / "state.json.bak_20260606"
    bak.write_text(json.dumps({"positions": {"2330": {}}}), encoding="utf-8")
    assert safe_read_json(p, default={}) == {"positions": {"2330": {}}}


def test_atomic_write_overwrites_existing(tmp_path):
    p = tmp_path / "state.json"
    atomic_write_json(p, {"v": 1})
    atomic_write_json(p, {"v": 2})
    assert safe_read_json(p) == {"v": 2}
