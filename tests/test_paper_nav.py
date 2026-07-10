"""paper_nav.py 測試：NAV 計算邏輯（picks+prices fixture）與冪等性。"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from scripts.paper_nav import (
    BASE_NOTE,
    DEFAULT_CONFIG_VERSION,
    compute_nav_series,
    load_nav_file,
    merge_rows,
    monthly_rebalance_dates,
    resolve_config_version,
    write_nav_file,
)

CV = "test-v1"


def _mk_picks(rows):
    return pd.DataFrame(rows, columns=["pick_date", "stock_id"])


def _mk_prices(rows):
    return pd.DataFrame(rows, columns=["trading_date", "stock_id", "close"])


# ──────────────────────────────────────────────────────────
# NAV 計算邏輯
# ──────────────────────────────────────────────────────────

class TestComputeNavSeries:
    def test_basic_two_stock_nav(self):
        """再平衡日等權進場後持股數固定：nav = sum(shares * close)。"""
        picks = _mk_picks([
            (date(2026, 1, 5), "1101"),
            (date(2026, 1, 5), "2330"),
        ])
        prices = _mk_prices([
            (date(2026, 1, 5), "1101", 100.0),
            (date(2026, 1, 5), "2330", 50.0),
            (date(2026, 1, 6), "1101", 110.0),  # +10%
            (date(2026, 1, 6), "2330", 45.0),   # -10%
            (date(2026, 1, 7), "1101", 120.0),
            (date(2026, 1, 7), "2330", 60.0),
        ])
        rows = compute_nav_series(picks, prices, date(2026, 1, 1), CV)

        assert [r["date"] for r in rows] == ["2026-01-05", "2026-01-06", "2026-01-07"]
        # 錨點：再平衡日 nav = 1.0
        assert rows[0]["nav"] == pytest.approx(1.0)
        assert rows[0]["holdings_n"] == 2
        # shares: 1101 → 0.5/100, 2330 → 0.5/50
        # d2: 0.005*110 + 0.01*45 = 0.55 + 0.45 = 1.0
        assert rows[1]["nav"] == pytest.approx(1.0)
        # d3: 0.005*120 + 0.01*60 = 0.6 + 0.6 = 1.2（權重漂移後非等權平均報酬）
        assert rows[2]["nav"] == pytest.approx(1.2)

    def test_buy_and_hold_weight_drift(self):
        """月內權重自然漂移：不是每日重設等權（區分 shares 模型 vs 每日均值報酬）。"""
        picks = _mk_picks([(date(2026, 1, 5), "1101"), (date(2026, 1, 5), "2330")])
        prices = _mk_prices([
            (date(2026, 1, 5), "1101", 100.0),
            (date(2026, 1, 5), "2330", 100.0),
            (date(2026, 1, 6), "1101", 200.0),  # 1101 翻倍 → 權重 2/3
            (date(2026, 1, 6), "2330", 100.0),
            (date(2026, 1, 7), "1101", 220.0),  # +10%
            (date(2026, 1, 7), "2330", 100.0),  # 0%
        ])
        rows = compute_nav_series(picks, prices, date(2026, 1, 1), CV)
        # d2 nav = 1.5；d3 = 0.005*220 + 0.005*100 = 1.6
        # （若每日重設等權會是 1.5 * (1 + 0.05) = 1.575 ≠ 1.6）
        assert rows[1]["nav"] == pytest.approx(1.5)
        assert rows[2]["nav"] == pytest.approx(1.6)

    def test_monthly_rebalance_switches_holdings(self):
        """每月第一個 pick 日換股；同月其後 picks 不改變持倉。"""
        picks = _mk_picks([
            (date(2026, 1, 5), "1101"),
            (date(2026, 1, 6), "9999"),  # 同月非首日 pick：必須被忽略
            (date(2026, 2, 2), "2330"),  # 2 月首 pick 日：換股
        ])
        prices = _mk_prices([
            (date(2026, 1, 5), "1101", 100.0),
            (date(2026, 1, 6), "1101", 110.0),
            (date(2026, 1, 6), "9999", 10.0),
            (date(2026, 2, 2), "1101", 130.0),
            (date(2026, 2, 2), "2330", 500.0),
            (date(2026, 2, 3), "1101", 999.0),  # 已出場，不應影響 nav
            (date(2026, 2, 3), "2330", 550.0),
        ])
        rows = compute_nav_series(picks, prices, date(2026, 1, 1), CV)
        by_date = {r["date"]: r for r in rows}

        # 1/6：仍持有 1101（9999 非月首 pick），nav = 110/100
        assert by_date["2026-01-06"]["nav"] == pytest.approx(1.10)
        assert by_date["2026-01-06"]["holdings_n"] == 1
        # 2/2 再平衡：舊倉 1101 先 mark 到 130 → nav=1.3，再全數換入 2330（等權進場）
        assert by_date["2026-02-02"]["nav"] == pytest.approx(1.30)
        assert "rebalance" in by_date["2026-02-02"]["notes"]
        # 2/3：只跟 2330 走：1.3 * 550/500 = 1.43（1101 漲到 999 無影響）
        assert by_date["2026-02-03"]["nav"] == pytest.approx(1.43)
        assert by_date["2026-02-03"]["holdings_n"] == 1

    def test_missing_price_ffill(self):
        """停牌/缺價日沿用最後收盤價 mark，notes 標注。"""
        picks = _mk_picks([(date(2026, 1, 5), "1101"), (date(2026, 1, 5), "2330")])
        prices = _mk_prices([
            (date(2026, 1, 5), "1101", 100.0),
            (date(2026, 1, 5), "2330", 50.0),
            (date(2026, 1, 6), "1101", 110.0),
            # 2330 於 1/6 停牌（無列）
            (date(2026, 1, 7), "1101", 110.0),
            (date(2026, 1, 7), "2330", 52.0),
        ])
        rows = compute_nav_series(picks, prices, date(2026, 1, 1), CV)
        by_date = {r["date"]: r for r in rows}
        # 1/6: 0.005*110 + 0.01*50(前收) = 1.05
        assert by_date["2026-01-06"]["nav"] == pytest.approx(1.05)
        assert "沿用前收" in by_date["2026-01-06"]["notes"]
        assert "2330" in by_date["2026-01-06"]["notes"]
        # 1/7 恢復交易，不再標注
        assert "沿用前收" not in by_date["2026-01-07"]["notes"]

    def test_entry_skips_stock_without_price(self):
        """進場日無任何價格的股票剔除，其餘等權；notes 標注略過。"""
        picks = _mk_picks([
            (date(2026, 1, 5), "1101"),
            (date(2026, 1, 5), "2330"),
            (date(2026, 1, 5), "4444"),  # 無任何價格
        ])
        prices = _mk_prices([
            (date(2026, 1, 5), "1101", 100.0),
            (date(2026, 1, 5), "2330", 50.0),
            (date(2026, 1, 6), "1101", 110.0),
            (date(2026, 1, 6), "2330", 55.0),
        ])
        rows = compute_nav_series(picks, prices, date(2026, 1, 1), CV)
        assert rows[0]["holdings_n"] == 2
        assert "略過" in rows[0]["notes"] and "4444" in rows[0]["notes"]
        # 兩檔各 +10% → nav 1.1
        assert rows[1]["nav"] == pytest.approx(1.10)

    def test_start_date_filters_old_picks(self):
        """start 之前的 picks（歷史錯位期）不入 forward 紀錄。"""
        picks = _mk_picks([
            (date(2026, 1, 5), "1101"),   # start 之前 → 排除
            (date(2026, 2, 2), "2330"),
        ])
        prices = _mk_prices([
            (date(2026, 1, 5), "1101", 100.0),
            (date(2026, 2, 2), "2330", 500.0),
            (date(2026, 2, 3), "2330", 510.0),
        ])
        rows = compute_nav_series(picks, prices, date(2026, 2, 1), CV)
        assert [r["date"] for r in rows] == ["2026-02-02", "2026-02-03"]
        assert rows[0]["nav"] == pytest.approx(1.0)
        assert rows[1]["nav"] == pytest.approx(1.02)

    def test_every_row_carries_bias_note_and_version(self):
        """每行 notes 必含未還原/低估之偏差聲明；config_version 正確寫入。"""
        picks = _mk_picks([(date(2026, 1, 5), "1101")])
        prices = _mk_prices([
            (date(2026, 1, 5), "1101", 100.0),
            (date(2026, 1, 6), "1101", 101.0),
        ])
        rows = compute_nav_series(picks, prices, date(2026, 1, 1), "v9-test")
        assert rows
        for r in rows:
            assert BASE_NOTE in r["notes"]
            assert "未還原" in r["notes"] and "低估" in r["notes"]
            assert r["config_version"] == "v9-test"

    def test_empty_inputs(self):
        assert compute_nav_series(_mk_picks([]), _mk_prices([]), date(2026, 1, 1), CV) == []
        picks = _mk_picks([(date(2026, 1, 5), "1101")])
        assert compute_nav_series(picks, _mk_prices([]), date(2026, 1, 1), CV) == []


def test_monthly_rebalance_dates():
    ds = [date(2026, 1, 6), date(2026, 1, 5), date(2026, 1, 20), date(2026, 2, 2)]
    assert monthly_rebalance_dates(ds) == [date(2026, 1, 5), date(2026, 2, 2)]


# ──────────────────────────────────────────────────────────
# 冪等性 / 合併 / 檔案 round-trip
# ──────────────────────────────────────────────────────────

def _row(d: str, nav: float, cv: str = CV) -> dict:
    return {"date": d, "nav": nav, "holdings_n": 2, "config_version": cv, "notes": BASE_NOTE}


class TestMergeIdempotency:
    def test_same_day_rerun_overwrites_last_row(self):
        existing = [_row("2026-01-05", 1.0), _row("2026-01-06", 1.05)]
        new = [_row("2026-01-05", 1.0), _row("2026-01-06", 1.07)]  # 同日重跑，尾行值更新
        merged, warnings = merge_rows(existing, new)
        assert [r["date"] for r in merged] == ["2026-01-05", "2026-01-06"]
        assert merged[-1]["nav"] == pytest.approx(1.07)  # 覆蓋當日行
        assert warnings == []

    def test_frozen_rows_keep_first_write_config_version(self):
        """歷史行凍結：config_version 保留 first-write 值，可按配置切段。"""
        existing = [_row("2026-01-05", 1.0, cv="v1-old"), _row("2026-01-06", 1.05, cv="v1-old")]
        new = [
            _row("2026-01-05", 1.0, cv="v2-new"),
            _row("2026-01-06", 1.05, cv="v2-new"),
            _row("2026-01-07", 1.10, cv="v2-new"),
        ]
        merged, _ = merge_rows(existing, new)
        by_date = {r["date"]: r for r in merged}
        assert by_date["2026-01-05"]["config_version"] == "v1-old"   # 凍結
        assert by_date["2026-01-06"]["config_version"] == "v2-new"   # 最後一行：覆蓋
        assert by_date["2026-01-07"]["config_version"] == "v2-new"   # 新增
        assert len(merged) == 3

    def test_gap_self_healing(self):
        """漏跑數日：下次執行自動補齊缺日。"""
        existing = [_row("2026-01-05", 1.0)]
        new = [_row("2026-01-05", 1.0), _row("2026-01-06", 1.02), _row("2026-01-07", 1.04)]
        merged, _ = merge_rows(existing, new)
        assert [r["date"] for r in merged] == ["2026-01-05", "2026-01-06", "2026-01-07"]

    def test_drift_warning_keeps_frozen_value(self):
        """上游 DB 回溯修改 → 警告，但凍結行保持 first-write 值。"""
        existing = [_row("2026-01-05", 1.0), _row("2026-01-06", 1.05)]
        new = [_row("2026-01-05", 1.5), _row("2026-01-06", 1.6)]  # 重放值大幅漂移
        merged, warnings = merge_rows(existing, new)
        by_date = {r["date"]: r for r in merged}
        assert by_date["2026-01-05"]["nav"] == pytest.approx(1.0)  # 凍結值保留
        assert len(warnings) == 1 and "2026-01-05" in warnings[0]
        # 最後一行照樣被覆蓋（同日重跑語意）
        assert by_date["2026-01-06"]["nav"] == pytest.approx(1.6)

    def test_empty_existing(self):
        new = [_row("2026-01-06", 1.05), _row("2026-01-05", 1.0)]
        merged, warnings = merge_rows([], new)
        assert [r["date"] for r in merged] == ["2026-01-05", "2026-01-06"]
        assert warnings == []

    def test_full_pipeline_rerun_is_idempotent(self, tmp_path):
        """compute → merge → write 連跑兩次，檔案內容 byte-identical。"""
        picks = _mk_picks([(date(2026, 1, 5), "1101"), (date(2026, 1, 5), "2330")])
        prices = _mk_prices([
            (date(2026, 1, 5), "1101", 100.0),
            (date(2026, 1, 5), "2330", 50.0),
            (date(2026, 1, 6), "1101", 110.0),
            (date(2026, 1, 6), "2330", 45.0),
        ])
        nav_path = tmp_path / "nav.jsonl"

        for _ in range(2):
            new_rows = compute_nav_series(picks, prices, date(2026, 1, 1), CV)
            merged, _ = merge_rows(load_nav_file(nav_path), new_rows)
            write_nav_file(nav_path, merged)

        lines = nav_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2  # 不因重跑而重複
        first = json.loads(lines[0])
        assert first["date"] == "2026-01-05"
        assert first["nav"] == pytest.approx(1.0)
        assert set(first) == {"date", "nav", "holdings_n", "config_version", "notes"}


def test_write_and_load_roundtrip(tmp_path):
    path = tmp_path / "sub" / "nav.jsonl"  # 目錄不存在 → 自動建立
    rows = [_row("2026-01-05", 1.0), _row("2026-01-06", 1.05)]
    write_nav_file(path, rows)
    assert load_nav_file(path) == rows
    assert not path.with_suffix(".jsonl.tmp").exists()  # 原子寫入不留 tmp


def test_config_version_env(monkeypatch):
    monkeypatch.delenv("PAPER_NAV_CONFIG_VERSION", raising=False)
    assert resolve_config_version() == DEFAULT_CONFIG_VERSION
    monkeypatch.setenv("PAPER_NAV_CONFIG_VERSION", "v3-exdate-patch")
    assert resolve_config_version() == "v3-exdate-patch"
