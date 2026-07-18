"""誠實日報（telegram_bot --strategy daily-brief）helper 測試。

只測純函式（nav 摘要 / IPO 過濾 / 處置股 diff / 訊息渲染），不碰 DB / 網路。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from scripts.telegram_bot import (
    HONEST_BANNER,
    _load_nav_records,
    _summarize_nav,
    _select_ipo_actionable,
    _load_latest_ipo_scan,
    _load_disposition_pair,
    _new_disposition_ids,
    _disposition_names,
    _render_daily_brief,
)


# ── nav ──────────────────────────────────────────────────────

def _write_nav(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_load_nav_records_missing_file(tmp_path):
    assert _load_nav_records(tmp_path / "nope.jsonl") == []


def test_load_nav_records_skips_corrupt_lines(tmp_path):
    p = tmp_path / "nav.jsonl"
    p.write_text(
        '{"date": "2026-07-03", "nav": 1.0}\n'
        "not-json\n"
        '{"date": "2026-07-04", "nav": 1.01}\n',
        encoding="utf-8",
    )
    recs = _load_nav_records(p)
    assert [r["date"] for r in recs] == ["2026-07-03", "2026-07-04"]


def test_summarize_nav_empty():
    assert _summarize_nav([]) is None


def test_summarize_nav_short_history_falls_back_to_earliest():
    recs = [
        {"date": "2026-07-03", "nav": 1.0, "holdings_n": 5, "config_version": "v2"},
        {"date": "2026-07-09", "nav": 0.99, "holdings_n": 5, "config_version": "v2"},
    ]
    out = _summarize_nav(recs)
    assert out["nav"] == pytest.approx(0.99)
    assert out["date"] == "2026-07-09"
    assert out["base_date"] == "2026-07-03"  # 不足 30 天 → 最早一筆
    assert out["chg_30d"] == pytest.approx(0.99 / 1.0 - 1)


def test_summarize_nav_uses_record_at_or_before_30d_cutoff():
    recs = [
        {"date": "2026-06-01", "nav": 1.00, "config_version": "v2"},
        {"date": "2026-06-10", "nav": 1.05, "config_version": "v2"},
        {"date": "2026-06-20", "nav": 1.10, "config_version": "v2"},
        {"date": "2026-07-15", "nav": 1.20, "config_version": "v2"},
    ]
    out = _summarize_nav(recs)
    # cutoff = 06-15 → 基準取 06-10（<= cutoff 的最近一筆）
    assert out["base_date"] == "2026-06-10"
    assert out["chg_30d"] == pytest.approx(1.20 / 1.05 - 1)


# ── ipo ──────────────────────────────────────────────────────

_TODAY = date(2026, 7, 18)


def _ipo(sid, discount, sub_end, name="測試"):
    return {"stock_id": sid, "name": name, "discount": discount,
            "sub_start": "2026-07-15", "sub_end": sub_end, "draw_date": "2026-07-22"}


def test_select_ipo_actionable_filters_discount_and_deadline():
    items = [
        _ipo("1111", 0.30, "2026-07-20"),   # 入選
        _ipo("2222", 0.05, "2026-07-20"),   # 折價不足
        _ipo("3333", 0.50, "2026-07-17"),   # 已截止
        _ipo("4444", None, "2026-07-20"),   # 折價未知（無市價）不列入
        _ipo("5555", 0.15, None),           # sub_end 缺 → 保守排除
    ]
    out = _select_ipo_actionable(items, _TODAY)
    assert [it["stock_id"] for it in out] == ["1111"]


def test_select_ipo_actionable_sorted_by_discount_desc():
    items = [_ipo("1111", 0.12, "2026-07-20"), _ipo("2222", 0.40, "2026-07-20")]
    out = _select_ipo_actionable(items, _TODAY)
    assert [it["stock_id"] for it in out] == ["2222", "1111"]


def test_load_latest_ipo_scan_picks_newest(tmp_path):
    (tmp_path / "scan_2026-07-01.json").write_text(
        json.dumps({"scan_date": "2026-07-01", "items": []}), encoding="utf-8")
    (tmp_path / "scan_2026-07-10.json").write_text(
        json.dumps({"scan_date": "2026-07-10", "items": []}), encoding="utf-8")
    scan = _load_latest_ipo_scan(tmp_path)
    assert scan["scan_date"] == "2026-07-10"


def test_load_latest_ipo_scan_empty_dir(tmp_path):
    assert _load_latest_ipo_scan(tmp_path) is None


# ── disposition ──────────────────────────────────────────────

def test_new_disposition_ids_diff():
    latest = {"disposition": ["1111", "2222", "3333"]}
    prev = {"disposition": ["1111"]}
    assert _new_disposition_ids(latest, prev) == ["2222", "3333"]


def test_new_disposition_ids_no_prev_returns_empty():
    # 首日無前一份可比 → 不誤報全量為「新增」
    assert _new_disposition_ids({"disposition": ["1111"]}, None) == []


def test_disposition_names_only_four_digit():
    dispo = {"records": [
        {"stock_id": "1515", "name": "力山"},
        {"stock_id": "052974", "name": "權證不算"},
    ]}
    names = _disposition_names(dispo)
    assert names == {"1515": "力山"}


def test_load_disposition_pair(tmp_path):
    (tmp_path / "2026-07-16.json").write_text(
        json.dumps({"as_of": "2026-07-16", "disposition": ["1111"]}), encoding="utf-8")
    (tmp_path / "2026-07-18.json").write_text(
        json.dumps({"as_of": "2026-07-18", "disposition": ["1111", "2222"]}), encoding="utf-8")
    latest, prev = _load_disposition_pair(tmp_path)
    assert latest["as_of"] == "2026-07-18"
    assert prev["as_of"] == "2026-07-16"


# ── render ───────────────────────────────────────────────────

def _render_minimal(**overrides):
    kwargs = dict(
        today=_TODAY,
        picks_info={"picks": [
            {"pick_date": "2026-07-17", "stock_id": "2330",
             "name": "台積電", "score": 0.1234},
        ], "total": 3},
        nav={"nav": 0.9927, "date": "2026-07-09", "chg_30d": -0.0073,
             "base_date": "2026-07-03", "holdings_n": 5, "config_version": "v2-20260703"},
        ipo_items=[_ipo("1717", 0.534, "2026-07-20", name="長興")],
        ipo_scan_date="2026-07-18",
        dispo_new=["1515"],
        dispo_names={"1515": "力山"},
        dispo_total=21,
        sentinel={"sanity_ok": True, "mismatch": 0, "error": None,
                  "last_job": "export_report success（2026-07-17 19:33:40）"},
    )
    kwargs.update(overrides)
    return _render_daily_brief(**kwargs)


def test_render_daily_brief_contains_all_sections_in_order():
    msg = _render_minimal()
    assert HONEST_BANNER in msg
    idx = [msg.index(k) for k in ("① ", "② ", "③ ", "④ ", "⑤ ")]
    assert idx == sorted(idx)
    assert "2330 台積電" in msg
    assert "紙上追蹤" in msg
    assert "0.9927" in msg
    assert "1717 長興" in msg and "+53.4%" in msg
    assert "1515 力山" in msg
    assert "✅ pick 特徵一致性抽驗通過" in msg


def test_render_daily_brief_escapes_html_in_names():
    msg = _render_minimal(
        picks_info={"picks": [
            {"pick_date": "2026-07-17", "stock_id": "9999",
             "name": "<b>bad</b>", "score": 0.1},
        ], "total": 1},
        ipo_items=[_ipo("8888", 0.20, "2026-07-20", name="<i>evil</i>")],
    )
    assert "<b>bad</b>" not in msg
    assert "&lt;b&gt;bad&lt;/b&gt;" in msg
    assert "&lt;i&gt;evil&lt;/i&gt;" in msg


def test_render_daily_brief_handles_all_missing_sections():
    msg = _render_daily_brief(
        today=_TODAY, picks_info=None, nav=None,
        ipo_items=[], ipo_scan_date=None,
        dispo_new=[], dispo_names={}, dispo_total=None,
        sentinel={"sanity_ok": None, "mismatch": None,
                  "error": "DB down", "last_job": None},
    )
    assert "picks 表無資料" in msg
    assert "無 NAV 紀錄" in msg
    assert "無折價" in msg
    assert "無處置股快取" in msg
    assert "哨兵無法執行" in msg


def test_render_daily_brief_sentinel_mismatch_alert():
    msg = _render_minimal(sentinel={"sanity_ok": False, "mismatch": 3,
                                    "error": None, "last_job": None})
    assert "🚨" in msg and "MISMATCH 3 筆" in msg
