#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(v):
    return float(v) if isinstance(v, (int, float)) and np.isfinite(float(v)) else None


def _status_text(v: bool | None) -> str:
    if v is True:
        return "pass"
    if v is False:
        return "fail"
    return "n/a"


def _evaluate_a_stability(shadow: Dict[str, object]) -> tuple[bool, Dict[str, object]]:
    windows = [shadow["windows"].get("1m", {}), shadow["windows"].get("3m", {}), shadow["windows"].get("6m", {})]
    checked = []
    pass_count = 0
    qualified_windows = 0
    for w in windows:
        if not w.get("available"):
            continue
        rb_n = len(w.get("period", {}).get("rebalance_dates", []))
        qualifies = rb_n >= 6
        if qualifies:
            qualified_windows += 1
        mm = w.get("model_metrics", {})
        am = w.get("multi_agent_metrics", {})
        mr = _num(mm.get("total_return"))
        ar = _num(am.get("total_return"))
        ms = _num(mm.get("sharpe"))
        ass = _num(am.get("sharpe"))
        mdd_m = _num(mm.get("max_drawdown"))
        mdd_a = _num(am.get("max_drawdown"))
        cond_ret = (ar is not None and mr is not None and ar >= mr - abs(mr) * 0.05)
        cond_sharpe = (ass is not None and ms is not None and ass >= ms - 0.20)
        cond_mdd = (mdd_a is not None and mdd_m is not None and mdd_a <= mdd_m + 0.03)
        window_pass = bool(cond_ret and cond_sharpe and cond_mdd)
        if qualifies and window_pass:
            pass_count += 1
        checked.append(
            {
                "window": f"{w.get('window_months')}m",
                "rebalance_dates": rb_n,
                "qualified_for_A": qualifies,
                "ret_check": cond_ret,
                "sharpe_check": cond_sharpe,
                "mdd_check": cond_mdd,
                "window_pass": window_pass,
            }
        )
    overall = qualified_windows >= 3 and pass_count >= 2
    return overall, {"checked_windows": checked, "qualified_windows": qualified_windows, "passed_windows": pass_count}


def _evaluate_b_regime(regime: Dict[str, object]) -> tuple[bool, Dict[str, object]]:
    perf = regime.get("performance_by_regime", {})
    major_rows = []
    non_negative = 0
    total = len(regime.get("per_period", []))
    for key in ["trend_regime", "vol_regime"]:
        for r in perf.get(key, []):
            if _num(r.get("multi_agent_minus_model")) is not None and r["multi_agent_minus_model"] >= 0:
                non_negative += 1
            ratio = (r.get("periods", 0) / total) if total > 0 else 0.0
            major_rows.append({**r, "source": key, "period_ratio": ratio})
    no_major_bad = all(not (r["period_ratio"] >= 0.3 and _num(r.get("multi_agent_minus_model")) is not None and r["multi_agent_minus_model"] < -0.03) for r in major_rows)
    pass_b = non_negative >= 2 and no_major_bad
    return pass_b, {"non_negative_regimes": non_negative, "no_major_bad_underperform": no_major_bad, "rows": major_rows}


def _evaluate_c_operational(shadow: Dict[str, object], fund_align: Dict[str, object]) -> tuple[bool, Dict[str, object]]:
    w6 = shadow["windows"].get("6m", {})
    model_ok = not bool(w6.get("model_invalid_result", True))
    ma_ok = not bool(w6.get("multi_agent_invalid_result", True))
    degraded_ok = _num((w6.get("multi_agent_metrics") or {}).get("degraded_period_ratio"))
    degraded_ok = bool(degraded_ok is not None and degraded_ok <= 0.2)
    rows = fund_align.get("rows", [])
    yoy_ok = bool(np.mean([r["after_non_null"]["fund_revenue_yoy"] for r in rows]) >= 0.8) if rows else False
    mom_ok = bool(np.mean([r["after_non_null"]["fund_revenue_mom"] for r in rows]) >= 0.8) if rows else False
    pass_c = bool(model_ok and ma_ok and degraded_ok and yoy_ok and mom_ok)
    return pass_c, {
        "model_invalid_result_ok": model_ok,
        "multi_agent_invalid_result_ok": ma_ok,
        "degraded_ratio_ok": degraded_ok,
        "fund_yoy_coverage_ok": yoy_ok,
        "fund_mom_coverage_ok": mom_ok,
    }


def _evaluate_d_tradability(shadow: Dict[str, object]) -> tuple[bool, Dict[str, object]]:
    gap = _load_json(ARTIFACTS_DIR / "tradability_gap_analysis.json")
    liq_obs = _load_json(ARTIFACTS_DIR / "liquidity_observability_report.json") if (ARTIFACTS_DIR / "liquidity_observability_report.json").exists() else {}
    w6 = gap.get("windows", {}).get("6m", {})
    summary = w6.get("summary", {}) if w6.get("available") else {}

    model_turnover = _num(summary.get("model_turnover"))
    ma_turnover = _num(summary.get("multi_agent_turnover"))
    avg_overlap = _num(summary.get("avg_overlap_jaccard"))
    low_liq_ratio = _num(summary.get("avg_low_liquidity_ratio"))
    replace_ratio = _num(summary.get("avg_replace_ratio"))

    d1_turnover = bool(model_turnover is not None and ma_turnover is not None and ma_turnover <= model_turnover + 0.12)
    d2_overlap = bool(avg_overlap is not None and avg_overlap >= 0.05)
    d3_observability_ready = bool(liq_obs.get("sufficient_for_d3_liquidity", False))
    d3_liquidity = bool(d3_observability_ready and low_liq_ratio is not None and low_liq_ratio <= 0.30)
    d3_status = "pass" if d3_liquidity else ("unknown_observability" if not d3_observability_ready else "fail")
    d4_switching_risk = bool(replace_ratio is not None and replace_ratio <= 0.80)

    w1 = gap.get("windows", {}).get("1m", {}).get("summary", {})
    w3 = gap.get("windows", {}).get("3m", {}).get("summary", {})
    trend = {
        "turnover_gap_1m": (_num(w1.get("multi_agent_turnover")) or 0.0) - (_num(w1.get("model_turnover")) or 0.0),
        "turnover_gap_3m": (_num(w3.get("multi_agent_turnover")) or 0.0) - (_num(w3.get("model_turnover")) or 0.0),
        "turnover_gap_6m": (ma_turnover or 0.0) - (model_turnover or 0.0),
        "overlap_1m": _num(w1.get("avg_overlap_jaccard")),
        "overlap_3m": _num(w3.get("avg_overlap_jaccard")),
        "overlap_6m": avg_overlap,
    }

    prev_d1 = bool(_num(w3.get("multi_agent_turnover")) is not None and _num(w3.get("model_turnover")) is not None and (_num(w3.get("multi_agent_turnover")) <= _num(w3.get("model_turnover")) + 0.12))
    prev_d2 = bool(_num(w3.get("avg_overlap_jaccard")) is not None and _num(w3.get("avg_overlap_jaccard")) >= 0.05)
    prev_d3 = bool(d3_observability_ready and _num(w3.get("avg_low_liquidity_ratio")) is not None and _num(w3.get("avg_low_liquidity_ratio")) <= 0.30)
    prev_d4 = bool(_num(w3.get("avg_replace_ratio")) is not None and _num(w3.get("avg_replace_ratio")) <= 0.80)

    readability_rows = [
        {
            "criteria": "D1_turnover",
            "current_status": _status_text(d1_turnover),
            "previous_status": _status_text(prev_d1),
            "trend": "improving" if trend["turnover_gap_6m"] < trend["turnover_gap_3m"] else "flat_or_worse",
            "blocker": "" if d1_turnover else "multi-agent turnover 高於容忍閾值",
            "notes": f"gap_3m={trend['turnover_gap_3m']:.4f}, gap_6m={trend['turnover_gap_6m']:.4f}",
        },
        {
            "criteria": "D2_overlap",
            "current_status": _status_text(d2_overlap),
            "previous_status": _status_text(prev_d2),
            "trend": "improving" if (trend["overlap_6m"] or 0.0) > (trend["overlap_3m"] or 0.0) else "flat_or_worse",
            "blocker": "" if d2_overlap else "model 與 multi-agent picks 重疊不足",
            "notes": f"overlap_3m={trend['overlap_3m']}, overlap_6m={trend['overlap_6m']}",
        },
        {
            "criteria": "D3_liquidity",
            "current_status": d3_status,
            "previous_status": _status_text(prev_d3),
            "trend": "observable" if d3_observability_ready else "unknown",
            "blocker": "" if d3_liquidity else ("資料觀測不足" if not d3_observability_ready else "低流動性比率超標"),
            "notes": f"avg_low_liq_ratio_6m={low_liq_ratio}",
        },
        {
            "criteria": "D4_switching_risk",
            "current_status": _status_text(d4_switching_risk),
            "previous_status": _status_text(prev_d4),
            "trend": "improving" if (_num(w3.get("avg_replace_ratio")) or 0.0) > (replace_ratio or 0.0) else "flat_or_worse",
            "blocker": "" if d4_switching_risk else "平均持股替換比例過高",
            "notes": f"replace_ratio_3m={_num(w3.get('avg_replace_ratio'))}, replace_ratio_6m={replace_ratio}",
        },
    ]

    pass_d = bool(d1_turnover and d2_overlap and d3_liquidity and d4_switching_risk)
    return pass_d, {
        "D1_turnover": d1_turnover,
        "D2_overlap": d2_overlap,
        "D3_liquidity": d3_liquidity,
        "D3_liquidity_status": d3_status,
        "D3_observability_ready": d3_observability_ready,
        "D4_switching_risk": d4_switching_risk,
        "values": {
            "model_turnover_6m": model_turnover,
            "multi_agent_turnover_6m": ma_turnover,
            "avg_overlap_6m": avg_overlap,
            "avg_low_liquidity_ratio_6m": low_liq_ratio,
            "avg_replace_ratio_6m": replace_ratio,
        },
        "improvement_trend": trend,
        "readability_rows": readability_rows,
    }


def update_tracker() -> Dict[str, object]:
    shadow = _load_json(ARTIFACTS_DIR / "shadow_monitor_latest.json")
    regime = _load_json(ARTIFACTS_DIR / "market_regime_analysis.json")
    fund_align = _load_json(ARTIFACTS_DIR / "debug_fund_alignment_round4.json")

    a_pass, a_detail = _evaluate_a_stability(shadow)
    b_pass, b_detail = _evaluate_b_regime(regime)
    c_pass, c_detail = _evaluate_c_operational(shadow, fund_align)
    d_pass, d_detail = _evaluate_d_tradability(shadow)
    transition = _load_json(ARTIFACTS_DIR / "promotion_transition_simulation.json") if (ARTIFACTS_DIR / "promotion_transition_simulation.json").exists() else {}

    checks = [
        {"id": "A_stability", "pass": a_pass, "detail": a_detail},
        {"id": "B_regime_consistency", "pass": b_pass, "detail": b_detail},
        {"id": "C_operational_stability", "pass": c_pass, "detail": c_detail},
        {"id": "D_tradability", "pass": d_pass, "detail": d_detail},
    ]
    passed = sum(1 for c in checks if c["pass"])
    missing = [c["id"] for c in checks if not c["pass"]]
    payload = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "total_checks": len(checks),
        "passed_checks": passed,
        "all_passed": passed == len(checks),
        "checks": checks,
        "missing_conditions": missing,
        "transition_policy_if_promoted": (transition.get("policy_recommendation") or {}).get(
            "default_policy_when_all_pass", "C_gradual_adoption"
        ),
    }
    (ARTIFACTS_DIR / "promotion_tracker.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Promotion Tracker",
        "",
        f"- updated_at: `{payload['updated_at']}`",
        f"- passed/total: `{passed}/{len(checks)}`",
        f"- all_passed: `{payload['all_passed']}`",
        "",
        "| check | pass |",
        "|---|---|",
    ]
    for c in checks:
        lines.append(f"| {c['id']} | {c['pass']} |")
    lines.extend(["", "## Missing Conditions"])
    for m in missing:
        lines.append(f"- {m}")
    d_detail = next((c["detail"] for c in checks if c["id"] == "D_tradability"), {})
    lines.extend(
        [
            "",
            "## D Tradability Sub-Checks",
            f"- D1_turnover: `{d_detail.get('D1_turnover')}`",
            f"- D2_overlap: `{d_detail.get('D2_overlap')}`",
            f"- D3_liquidity: `{d_detail.get('D3_liquidity')}`",
            f"- D3_liquidity_status: `{d_detail.get('D3_liquidity_status')}`",
            f"- D3_observability_ready: `{d_detail.get('D3_observability_ready')}`",
            f"- D4_switching_risk: `{d_detail.get('D4_switching_risk')}`",
            "",
            "## D Trend Readability",
            "| criteria | current_status | previous_status | trend | blocker | notes |",
            "|---|---|---|---|---|---|",
            "",
        ]
    )
    for r in d_detail.get("readability_rows", []):
        lines.append(
            f"| {r.get('criteria')} | {r.get('current_status')} | {r.get('previous_status')} | {r.get('trend')} | {r.get('blocker')} | {r.get('notes')} |"
        )
    lines.extend(
        [
            "",
            "## Transition Policy (If Promoted)",
            f"- default_policy_when_all_pass: `{payload.get('transition_policy_if_promoted')}`",
        ]
    )
    lines.extend(["", "## Check Details"])
    for c in checks:
        lines.append(f"- {c['id']}: `{c['detail']}`")
    (ARTIFACTS_DIR / "promotion_tracker.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    payload = update_tracker()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
