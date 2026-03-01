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


def _num(v) -> float | None:
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _status_change(current: str, previous: str) -> str:
    if current == previous:
        return "no_change"
    if previous in {"fail", "unknown_observability"} and current == "pass":
        return "improved"
    if previous == "pass" and current in {"fail", "unknown_observability"}:
        return "worsened"
    return "changed"


def _to_status(v) -> str:
    if isinstance(v, bool):
        return "pass" if v else "fail"
    if isinstance(v, str):
        return v
    return "n/a"


def _fmt_num(v) -> str:
    x = _num(v)
    return f"{x:.6f}" if x is not None else "N/A"


def _interpret_delta(metric: str, prev_v, cur_v) -> str:
    p = _num(prev_v)
    c = _num(cur_v)
    if p is None or c is None:
        return "N/A"
    d = c - p
    if abs(d) < 1e-12:
        return "flat"
    higher_better = {"return", "sharpe", "mdd", "picks_stability", "overlap", "promotion_pass_ratio"}
    lower_better = {"turnover"}
    if metric in higher_better:
        return "improved" if d > 0 else "worsened"
    if metric in lower_better:
        return "improved" if d < 0 else "worsened"
    return "flat"


def _delta_row(metric: str, prev_v, cur_v) -> Dict[str, str]:
    p = _num(prev_v)
    c = _num(cur_v)
    if p is None or c is None:
        delta = "N/A"
    else:
        delta = f"{(c - p):.6f}"
    return {
        "metric": metric,
        "previous_month": _fmt_num(prev_v),
        "current_month": _fmt_num(cur_v),
        "delta": delta,
        "interpretation": _interpret_delta(metric, prev_v, cur_v),
    }


def render_monthly_shadow_review() -> Dict[str, object]:
    shadow = _load_json(ARTIFACTS_DIR / "shadow_monitor_latest.json")
    tracker = _load_json(ARTIFACTS_DIR / "promotion_tracker.json")
    regime = _load_json(ARTIFACTS_DIR / "market_regime_analysis.json")
    tradability = _load_json(ARTIFACTS_DIR / "tradability_gap_analysis.json")
    transition = _load_json(ARTIFACTS_DIR / "promotion_transition_simulation.json")

    w1 = shadow.get("windows", {}).get("1m", {})
    model_1m = w1.get("model_metrics", {})
    ma_1m = w1.get("multi_agent_metrics", {})
    d_detail = next((c.get("detail", {}) for c in tracker.get("checks", []) if c.get("id") == "D_tradability"), {})
    readability_rows = d_detail.get("readability_rows", [])
    criteria_changes = []
    for r in readability_rows:
        cur = _to_status(r.get("current_status"))
        prev = _to_status(r.get("previous_status"))
        criteria_changes.append(
            {
                "criteria": r.get("criteria"),
                "current_status": cur,
                "previous_status": prev,
                "change": _status_change(cur, prev),
                "blocker": r.get("blocker") or "",
            }
        )

    trend_rows = regime.get("performance_by_regime", {}).get("trend_regime", [])
    vol_rows = regime.get("performance_by_regime", {}).get("vol_regime", [])
    strongest = sorted(
        trend_rows + vol_rows,
        key=lambda r: (_num(r.get("multi_agent_minus_model")) if _num(r.get("multi_agent_minus_model")) is not None else -999.0),
        reverse=True,
    )
    strongest_regime = strongest[0] if strongest else {}

    tr6 = tradability.get("windows", {}).get("6m", {}).get("summary", {})
    transition_reco = transition.get("policy_recommendation", {})
    recommended_now = transition_reco.get("recommended_now", "B_blended_70_30")

    passed = tracker.get("passed_checks", 0)
    total = tracker.get("total_checks", 0)
    d2 = bool(d_detail.get("D2_overlap", False))
    d4 = bool(d_detail.get("D4_switching_risk", False))
    d3_ready = bool(d_detail.get("D3_observability_ready", False))
    if tracker.get("all_passed", False):
        action = "trigger promotion review"
        action_reason = "所有 promotion criteria 已通過"
    elif passed >= 2 and d3_ready and recommended_now.startswith("B_"):
        action = "prepare blended pilot"
        action_reason = "觀測與操作穩定性可接受，可先用 blended 低風險試點"
    elif d2 is False or d4 is False:
        action = "continue shadow"
        action_reason = "tradability 關鍵缺口仍在 overlap / switching risk"
    else:
        action = "hold"
        action_reason = "尚未形成足夠升級訊號，維持觀察"

    prev_payload = None
    prev_path = ARTIFACTS_DIR / "monthly_shadow_review_latest.json"
    if prev_path.exists():
        try:
            prev_payload = _load_json(prev_path)
        except Exception:
            prev_payload = None

    current_promotion_ratio = (float(passed) / float(total)) if total else None
    prev_kpi = ((prev_payload or {}).get("monthly_kpi_comparison") or {})
    prev_model = (prev_kpi.get("model") or {})
    prev_ma = (prev_kpi.get("multi_agent") or {})
    prev_promo = ((prev_payload or {}).get("promotion_criteria_change_summary") or {})
    prev_passed = prev_promo.get("passed_checks")
    prev_total = prev_promo.get("total_checks")
    prev_promotion_ratio = (float(prev_passed) / float(prev_total)) if isinstance(prev_passed, int) and isinstance(prev_total, int) and prev_total > 0 else None

    mom_delta_rows = [
        _delta_row("return", prev_ma.get("total_return"), _num(ma_1m.get("total_return"))),
        _delta_row("sharpe", prev_ma.get("sharpe"), ma_1m.get("sharpe")),
        _delta_row("mdd", prev_ma.get("max_drawdown"), _num(ma_1m.get("max_drawdown"))),
        _delta_row("turnover", prev_ma.get("turnover"), _num(ma_1m.get("turnover"))),
        _delta_row("picks_stability", prev_ma.get("picks_stability"), _num(ma_1m.get("picks_stability"))),
        _delta_row("overlap", prev_kpi.get("avg_picks_overlap_jaccard"), _num(w1.get("avg_picks_overlap_jaccard"))),
        _delta_row("promotion_pass_ratio", prev_promotion_ratio, current_promotion_ratio),
    ]

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "month_window": w1.get("period", {}),
        "monthly_kpi_comparison": {
            "model": {
                "total_return": _num(model_1m.get("total_return")),
                "sharpe": model_1m.get("sharpe"),
                "max_drawdown": _num(model_1m.get("max_drawdown")),
                "turnover": _num(model_1m.get("turnover")),
                "picks_stability": _num(model_1m.get("picks_stability")),
            },
            "multi_agent": {
                "total_return": _num(ma_1m.get("total_return")),
                "sharpe": ma_1m.get("sharpe"),
                "max_drawdown": _num(ma_1m.get("max_drawdown")),
                "turnover": _num(ma_1m.get("turnover")),
                "picks_stability": _num(ma_1m.get("picks_stability")),
            },
            "avg_picks_overlap_jaccard": _num(w1.get("avg_picks_overlap_jaccard")),
        },
        "promotion_criteria_change_summary": {
            "passed_checks": passed,
            "total_checks": total,
            "changes": criteria_changes,
        },
        "month_over_month_delta": {
            "previous_generated_at": (prev_payload or {}).get("generated_at"),
            "rows": mom_delta_rows,
        },
        "regime_summary": {
            "strongest_relative_regime": strongest_regime,
            "non_negative_regime_count": _num(
                next(
                    (c.get("detail", {}).get("non_negative_regimes") for c in tracker.get("checks", []) if c.get("id") == "B_regime_consistency"),
                    None,
                )
            ),
        },
        "tradability_transition_summary": {
            "D2_overlap": d2,
            "D3_liquidity_status": d_detail.get("D3_liquidity_status"),
            "D4_switching_risk": d4,
            "avg_overlap_6m": _num(tr6.get("avg_overlap_jaccard")),
            "avg_replace_ratio_6m": _num(tr6.get("avg_replace_ratio")),
            "default_policy_if_promoted": tracker.get("transition_policy_if_promoted"),
            "recommended_now": recommended_now,
        },
        "recommended_action": {
            "action": action,
            "reason": action_reason,
            "action_candidates": ["continue shadow", "prepare blended pilot", "trigger promotion review", "hold"],
        },
    }
    (ARTIFACTS_DIR / "monthly_shadow_review_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines: List[str] = [
        "# Monthly Shadow Review",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- month_window: `{payload['month_window']}`",
        "",
        "## 本月 model vs multi-agent 關鍵指標比較",
        "| metric | model | multi_agent |",
        "|---|---:|---:|",
    ]
    for m in ["total_return", "max_drawdown", "turnover", "picks_stability"]:
        lines.append(f"| {m} | {payload['monthly_kpi_comparison']['model'].get(m)} | {payload['monthly_kpi_comparison']['multi_agent'].get(m)} |")
    lines.extend(
        [
            f"| sharpe | {payload['monthly_kpi_comparison']['model'].get('sharpe')} | {payload['monthly_kpi_comparison']['multi_agent'].get('sharpe')} |",
            "",
            "## Promotion criteria 變化摘要",
            f"- passed_checks: `{passed}/{total}`",
            "| criteria | current_status | previous_status | change | blocker |",
            "|---|---|---|---|---|",
        ]
    )
    for r in criteria_changes:
        lines.append(
            f"| {r['criteria']} | {r['current_status']} | {r['previous_status']} | {r['change']} | {r['blocker']} |"
        )
    lines.extend(
        [
            "",
            "## Month-over-Month Delta",
            f"- previous_generated_at: `{payload['month_over_month_delta']['previous_generated_at'] or 'N/A'}`",
            "| metric | previous_month | current_month | delta | interpretation |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for r in mom_delta_rows:
        lines.append(
            f"| {r['metric']} | {r['previous_month']} | {r['current_month']} | {r['delta']} | {r['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Regime 表現摘要",
            f"- strongest_relative_regime: `{payload['regime_summary']['strongest_relative_regime']}`",
            f"- non_negative_regime_count: `{payload['regime_summary']['non_negative_regime_count']}`",
            "",
            "## Tradability / Transition 狀態摘要",
            f"- D2_overlap: `{d2}`",
            f"- D3_liquidity_status: `{d_detail.get('D3_liquidity_status')}`",
            f"- D4_switching_risk: `{d4}`",
            f"- avg_overlap_6m: `{_num(tr6.get('avg_overlap_jaccard'))}`",
            f"- avg_replace_ratio_6m: `{_num(tr6.get('avg_replace_ratio'))}`",
            f"- recommended_now: `{recommended_now}`",
            f"- default_policy_if_promoted: `{tracker.get('transition_policy_if_promoted')}`",
            "",
            "## 建議 Action",
            f"- action: `{action}`",
            f"- reason: {action_reason}",
            "- action_candidates: `continue shadow` / `prepare blended pilot` / `trigger promotion review` / `hold`",
        ]
    )
    (ARTIFACTS_DIR / "monthly_shadow_review_latest.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    payload = render_monthly_shadow_review()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
