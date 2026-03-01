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


def _max_drawdown(period_returns: List[float]) -> float:
    if not period_returns:
        return 0.0
    equity = np.cumprod([1.0 + r for r in period_returns])
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))


def _sharpe(period_returns: List[float]) -> float | str:
    if len(period_returns) < 2:
        return "N/A: <2 periods"
    std = float(np.std(period_returns, ddof=1))
    if std <= 0:
        return "N/A: zero std"
    return float((np.mean(period_returns) / std) * np.sqrt(12))


def _simulate_scheme(
    name: str,
    model_returns: List[float],
    ma_returns: List[float],
    model_turnover: float,
    ma_turnover: float,
    replace_ratio: float,
    overlap: float,
    ma_weights: List[float],
) -> Dict[str, object]:
    comb_ret = [(1.0 - w) * mr + w * ar for mr, ar, w in zip(model_returns, ma_returns, ma_weights)]
    total_return = float(np.prod([1.0 + r for r in comb_ret]) - 1.0) if comb_ret else 0.0
    avg_turnover = float(np.mean([(1.0 - w) * model_turnover + w * ma_turnover for w in ma_weights])) if ma_weights else 0.0
    switch_cost = abs(ma_weights[0] if ma_weights else 0.0) * replace_ratio
    if len(ma_weights) >= 2:
        switch_cost += float(np.mean(np.abs(np.diff(ma_weights)))) * replace_ratio
    replacement_rate = float(np.mean([w * replace_ratio for w in ma_weights])) if ma_weights else 0.0
    switching_risk_proxy = switch_cost + replacement_rate
    return {
        "scheme": name,
        "multi_agent_weights_over_time": ma_weights,
        "total_return": total_return,
        "sharpe": _sharpe(comb_ret),
        "max_drawdown": _max_drawdown(comb_ret),
        "turnover_impact": avg_turnover,
        "overlap_proxy": float((1.0 - np.mean(ma_weights)) + np.mean(ma_weights) * overlap) if ma_weights else overlap,
        "replacement_rate": replacement_rate,
        "switching_risk_proxy": switching_risk_proxy,
    }


def simulate_transition() -> Dict[str, object]:
    shadow = _load_json(ARTIFACTS_DIR / "shadow_monitor_latest.json")
    tradability = _load_json(ARTIFACTS_DIR / "tradability_gap_analysis.json")
    w6_shadow = shadow.get("windows", {}).get("6m", {})
    w6_tr = tradability.get("windows", {}).get("6m", {})
    per = w6_shadow.get("per_period", [])
    model_returns = [_num(p.get("model_return")) or 0.0 for p in per]
    ma_returns = [_num(p.get("multi_agent_return")) or 0.0 for p in per]

    model_turnover = _num((w6_shadow.get("model_metrics") or {}).get("turnover")) or 0.0
    ma_turnover = _num((w6_shadow.get("multi_agent_metrics") or {}).get("turnover")) or 0.0
    tr_summary = w6_tr.get("summary", {})
    replace_ratio = _num(tr_summary.get("avg_replace_ratio")) or 1.0
    overlap = _num(tr_summary.get("avg_overlap_jaccard")) or 0.0

    n = len(model_returns)
    hard = _simulate_scheme(
        "A_hard_switch",
        model_returns,
        ma_returns,
        model_turnover,
        ma_turnover,
        replace_ratio,
        overlap,
        [1.0] * n,
    )
    blend_7030 = _simulate_scheme(
        "B_blended_70_30",
        model_returns,
        ma_returns,
        model_turnover,
        ma_turnover,
        replace_ratio,
        overlap,
        [0.3] * n,
    )
    blend_5050 = _simulate_scheme(
        "B_blended_50_50",
        model_returns,
        ma_returns,
        model_turnover,
        ma_turnover,
        replace_ratio,
        overlap,
        [0.5] * n,
    )

    gradual_weights: List[float] = []
    w = 0.3
    streak = 0
    for mr, ar in zip(model_returns, ma_returns):
        if ar >= mr:
            streak += 1
        else:
            streak = 0
        if streak >= 2:
            w = min(1.0, w + 0.2)
        elif ar < mr - 0.03:
            w = max(0.3, w - 0.1)
        gradual_weights.append(round(w, 4))
    gradual = _simulate_scheme(
        "C_gradual_adoption",
        model_returns,
        ma_returns,
        model_turnover,
        ma_turnover,
        replace_ratio,
        overlap,
        gradual_weights,
    )

    model_base_total = _num((w6_shadow.get("model_metrics") or {}).get("total_return")) or 0.0
    schemes = [hard, blend_7030, blend_5050, gradual]
    for s in schemes:
        s["delta_vs_model_total_return"] = float(s["total_return"] - model_base_total)

    def _is_pass(s: Dict[str, object]) -> bool:
        sharpe = _num(s.get("sharpe"))
        return bool(
            s["delta_vs_model_total_return"] >= -0.05
            and sharpe is not None
            and sharpe >= 1.0
            and s["switching_risk_proxy"] <= 1.2
        )

    pass_map = {s["scheme"]: _is_pass(s) for s in schemes}
    all_primary_pass = all(pass_map.get(k, False) for k in ["A_hard_switch", "B_blended_70_30", "C_gradual_adoption"])

    recommended = min(
        schemes,
        key=lambda s: (
            0 if _is_pass(s) else 1,
            float(s["switching_risk_proxy"]),
            -float(s["total_return"]),
        ),
    )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "baseline": {
            "model_total_return_6m": model_base_total,
            "model_turnover_6m": model_turnover,
            "multi_agent_turnover_6m": ma_turnover,
            "avg_overlap_6m": overlap,
            "avg_replace_ratio_6m": replace_ratio,
        },
        "schemes": schemes,
        "pass_assessment": pass_map,
        "policy_recommendation": {
            "recommended_now": recommended["scheme"],
            "reason": "在滿足基本績效門檻下，切換風險代理值最低",
            "default_policy_when_all_pass": "C_gradual_adoption",
            "all_primary_pass": all_primary_pass,
        },
    }
    (ARTIFACTS_DIR / "promotion_transition_simulation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Promotion Transition Simulation",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        "",
        "## Baseline",
        f"- model_total_return_6m: {model_base_total:.4f}",
        f"- turnover(model/ma): {model_turnover:.4f}/{ma_turnover:.4f}",
        f"- overlap_6m: {overlap:.4f}",
        f"- replace_ratio_6m: {replace_ratio:.4f}",
        "",
        "## Scheme Comparison",
    ]
    for s in schemes:
        lines.extend(
            [
                f"### {s['scheme']}",
                f"- total_return: {s['total_return']:.4f}",
                f"- sharpe: {s['sharpe']}",
                f"- max_drawdown: {s['max_drawdown']:.4f}",
                f"- turnover_impact: {s['turnover_impact']:.4f}",
                f"- overlap_proxy: {s['overlap_proxy']:.4f}",
                f"- replacement_rate: {s['replacement_rate']:.4f}",
                f"- switching_risk_proxy: {s['switching_risk_proxy']:.4f}",
                f"- delta_vs_model_total_return: {s['delta_vs_model_total_return']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy Recommendation",
            f"- recommended_now: `{payload['policy_recommendation']['recommended_now']}`",
            f"- default_policy_when_all_pass: `{payload['policy_recommendation']['default_policy_when_all_pass']}`",
            f"- reason: {payload['policy_recommendation']['reason']}",
        ]
    )
    (ARTIFACTS_DIR / "promotion_transition_simulation.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    payload = simulate_transition()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
