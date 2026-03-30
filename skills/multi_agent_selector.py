from __future__ import annotations

from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


AGENT_NAMES = ["tech", "flow", "margin", "fund", "theme"]
SIGNAL_TO_SCORE = {-2: -1.0, -1: -0.5, 0: 0.0, 1: 0.5, 2: 1.0}

# 使用 FeatureStore 實際存在的欄位（2026-03-27 確認）
AGENT_REQUIRED_COLUMNS = {
    "tech":   ["ret_20", "boll_pct", "rsi_14", "macd_hist", "drawdown_60", "vol_ratio_20"],
    "flow":   ["foreign_net_20", "trust_net_20", "three_instn_net_20b", "chip_flow_intensity_20"],
    "margin": ["short_balance_chg_20", "margin_short_ratio"],
    "fund":   ["fund_revenue_yoy", "fund_revenue_mom", "fund_revenue_yoy_accel"],
    "theme":  ["theme_return_20", "theme_turnover_ratio"],
}

# Supervisor 分組：三個獨立視角
AGENT_GROUPS = {
    "technical":     ["tech"],
    "institutional": ["flow", "margin"],
    "fundamental":   ["fund", "theme"],
}

_CONFLICT_WATCH = 0.35   # group 信號 std > 此值 → WATCH（-20% confidence）
_CONFLICT_SKIP  = 0.65   # group 信號 std > 此值 → CONFLICT（-40% confidence）


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    std = float(vals.std(skipna=True) or 0.0)
    if std == 0:
        return pd.Series(0.0, index=series.index)
    mean = float(vals.mean(skipna=True))
    return (vals - mean) / std


def _risk_flags(row: pd.Series) -> List[str]:
    flags: List[str] = []
    drawdown_60 = pd.to_numeric(row.get("drawdown_60"), errors="coerce")
    amt_ratio_20 = pd.to_numeric(row.get("amt_ratio_20"), errors="coerce")
    if pd.notna(drawdown_60) and float(drawdown_60) < -0.25:
        flags.append("high_drawdown")
    if pd.notna(amt_ratio_20) and float(amt_ratio_20) < 0.7:
        flags.append("liquidity_risk")
    return flags


def _base_output(ticker: str, agent: str) -> Dict[str, object]:
    return {
        "ticker": ticker,
        "agent": agent,
        "signal": 0,
        "confidence": 0.0,
        "reasons": [],
        "risk_flags": [],
        "unavailable": False,
    }


def _mark_unavailable(output: Dict[str, object], reason: str) -> Dict[str, object]:
    output["unavailable"] = True
    output["reasons"] = [reason]
    output["signal"] = 0
    output["confidence"] = 0.0
    return output


def _tech_agent(row: pd.Series, z_map: Dict[str, pd.Series]) -> Dict[str, object]:
    ticker = str(row["stock_id"])
    out = _base_output(ticker, "tech")
    missing = [c for c in AGENT_REQUIRED_COLUMNS["tech"] if c not in row.index]
    if missing:
        return _mark_unavailable(out, f"missing columns: {missing}")

    ret_20  = float(pd.to_numeric(row.get("ret_20"),   errors="coerce") or 0.0)
    boll    = float(pd.to_numeric(row.get("boll_pct"), errors="coerce") or 0.5)
    macd    = float(pd.to_numeric(row.get("macd_hist"),errors="coerce") or 0.0)
    drawdown= float(pd.to_numeric(row.get("drawdown_60"), errors="coerce") or 0.0)
    rsi     = float(pd.to_numeric(row.get("rsi_14"),   errors="coerce") or 50.0)
    z_ret   = float(z_map["ret_20"].loc[row.name] if row.name in z_map["ret_20"].index else 0.0)

    # boll_pct > 0.8 ≈ 接近上軌（突破訊號代理）
    near_breakout = boll > 0.8
    if ret_20 > 0.06 and near_breakout and macd > 0 and drawdown > -0.18:
        signal = 2
    elif ret_20 > 0 and macd >= 0:
        signal = 1
    elif ret_20 < -0.08 and macd < 0 and drawdown < -0.28:
        signal = -2
    elif ret_20 < 0:
        signal = -1
    else:
        signal = 0

    conf = _clip01((abs(z_ret) + (1.0 if near_breakout else 0.0) + (1.0 if macd > 0 else 0.0)) / 4.0)
    out["signal"] = signal
    out["confidence"] = conf
    out["reasons"] = [f"ret_20={ret_20:.4f}", f"boll_pct={boll:.2f}", f"macd_hist={macd:.4f}", f"rsi_14={rsi:.1f}"]
    out["risk_flags"] = _risk_flags(row)
    return out


def _flow_agent(row: pd.Series, z_map: Dict[str, pd.Series], dq_ctx: Dict[str, object]) -> Dict[str, object]:
    ticker = str(row["stock_id"])
    out = _base_output(ticker, "flow")
    if "raw_institutional" in set(dq_ctx.get("degraded_datasets", [])):
        return _mark_unavailable(out, "raw_institutional degraded")
    missing = [c for c in AGENT_REQUIRED_COLUMNS["flow"] if c not in row.index]
    if missing:
        return _mark_unavailable(out, f"missing columns: {missing}")

    foreign   = float(pd.to_numeric(row.get("foreign_net_20"),       errors="coerce") or 0.0)
    trust     = float(pd.to_numeric(row.get("trust_net_20"),          errors="coerce") or 0.0)
    three_ins = float(pd.to_numeric(row.get("three_instn_net_20b"),   errors="coerce") or 0.0)
    intensity = float(pd.to_numeric(row.get("chip_flow_intensity_20"),errors="coerce") or 0.0)
    z_foreign = float(z_map["foreign_net_20"].loc[row.name] if row.name in z_map["foreign_net_20"].index else 0.0)

    if foreign > 0 and intensity > 0.002 and z_foreign > 0.5:
        signal = 2
    elif foreign > 0 or three_ins > 0:
        signal = 1
    elif foreign < 0 and intensity < -0.002 and z_foreign < -0.5:
        signal = -2
    elif foreign < 0:
        signal = -1
    else:
        signal = 0

    conf = _clip01((abs(z_foreign) + min(abs(intensity) * 100, 2.0)) / 4.0)
    out["signal"] = signal
    out["confidence"] = conf
    out["reasons"] = [
        f"foreign_net_20={foreign:.2f}",
        f"trust_net_20={trust:.2f}",
        f"three_instn_net_20b={three_ins:.2f}",
        f"chip_flow_intensity_20={intensity:.6f}",
    ]
    out["risk_flags"] = _risk_flags(row)
    return out


def _margin_agent(row: pd.Series, dq_ctx: Dict[str, object], ratio_median: float) -> Dict[str, object]:
    ticker = str(row["stock_id"])
    out = _base_output(ticker, "margin")
    if "raw_margin_short" in set(dq_ctx.get("degraded_datasets", [])):
        return _mark_unavailable(out, "raw_margin_short degraded")
    missing = [c for c in AGENT_REQUIRED_COLUMNS["margin"] if c not in row.index]
    if missing:
        return _mark_unavailable(out, f"missing columns: {missing}")

    short_chg = float(pd.to_numeric(row.get("short_balance_chg_20"), errors="coerce") or 0.0)
    ratio     = float(pd.to_numeric(row.get("margin_short_ratio"),   errors="coerce") or 0.0)
    short_chg5= float(pd.to_numeric(row.get("short_balance_chg_5"), errors="coerce") or 0.0)

    # 空單餘額下降（回補）+ 融資比率低 → 多頭訊號
    if short_chg < 0 and ratio < 0.25:
        signal = 2
    elif short_chg < 0:
        signal = 1
    elif short_chg > 0 and ratio > 0.35:
        signal = -2
    elif short_chg > 0:
        signal = -1
    else:
        signal = 0

    conf = _clip01(min(abs(ratio - ratio_median) / max(abs(ratio_median), 0.1), 1.0))
    out["signal"] = signal
    out["confidence"] = conf
    out["reasons"] = [
        f"short_balance_chg_20={short_chg:.4f}",
        f"short_balance_chg_5={short_chg5:.4f}",
        f"margin_short_ratio={ratio:.4f}",
    ]
    out["risk_flags"] = _risk_flags(row)
    return out


def _fund_agent(row: pd.Series, dq_ctx: Dict[str, object]) -> Dict[str, object]:
    ticker = str(row["stock_id"])
    out = _base_output(ticker, "fund")
    if "raw_fundamentals" in set(dq_ctx.get("degraded_datasets", [])):
        return _mark_unavailable(out, "raw_fundamentals degraded")
    missing = [c for c in AGENT_REQUIRED_COLUMNS["fund"] if c not in row.index]
    if missing:
        return _mark_unavailable(out, f"missing columns: {missing}")

    yoy   = float(pd.to_numeric(row.get("fund_revenue_yoy"),      errors="coerce") or 0.0)
    mom   = float(pd.to_numeric(row.get("fund_revenue_mom"),      errors="coerce") or 0.0)
    accel = float(pd.to_numeric(row.get("fund_revenue_yoy_accel"),errors="coerce") or 0.0)

    # yoy > 0.15 且加速 → 強基本面
    if yoy > 0.15 and accel > 0:
        signal = 2
    elif yoy > 0:
        signal = 1
    elif yoy < -0.1 and accel < 0:
        signal = -2
    elif yoy < 0:
        signal = -1
    else:
        signal = 0

    conf = _clip01(min((abs(yoy) * 2.5 + abs(accel) * 2.0 + abs(mom)), 1.0))
    out["signal"] = signal
    out["confidence"] = conf
    out["reasons"] = [
        f"fund_revenue_yoy={yoy:.4f}",
        f"fund_revenue_yoy_accel={accel:.4f}",
        f"fund_revenue_mom={mom:.4f}",
    ]
    out["risk_flags"] = _risk_flags(row)
    return out


def _theme_agent(row: pd.Series, dq_ctx: Dict[str, object]) -> Dict[str, object]:
    ticker = str(row["stock_id"])
    out = _base_output(ticker, "theme")
    if "raw_theme_flow" in set(dq_ctx.get("degraded_datasets", [])):
        return _mark_unavailable(out, "raw_theme_flow degraded")
    missing = [c for c in AGENT_REQUIRED_COLUMNS["theme"] if c not in row.index]
    if missing:
        return _mark_unavailable(out, f"missing columns: {missing}")

    ret20         = float(pd.to_numeric(row.get("theme_return_20"),   errors="coerce") or 0.0)
    turnover_ratio= float(pd.to_numeric(row.get("theme_turnover_ratio"), errors="coerce") or 0.0)

    if ret20 > 0.05 and turnover_ratio > 0.5:
        signal = 2
    elif ret20 > 0:
        signal = 1
    elif ret20 < -0.05 and turnover_ratio > 0.5:
        signal = -2
    elif ret20 < 0:
        signal = -1
    else:
        signal = 0

    conf = _clip01(min((abs(ret20) * 4 + abs(turnover_ratio) * 2), 1.0))
    out["signal"] = signal
    out["confidence"] = conf
    out["reasons"] = [f"theme_return_20={ret20:.4f}", f"theme_turnover_ratio={turnover_ratio:.4f}"]
    out["risk_flags"] = _risk_flags(row)
    return out


# ── Supervisor ────────────────────────────────────────────────────────────────

def _supervisor(agent_map: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """分析三個維度訊號是否一致，輸出 verdict / confidence_adj / explanation。

    PASS     → 三組方向一致（std < 0.35），confidence_adj = 1.0
    WATCH    → 輕度分歧（0.35 ≤ std < 0.65），confidence_adj = 0.80
    CONFLICT → 嚴重分歧（std ≥ 0.65），confidence_adj = 0.60
    """
    group_signals: Dict[str, float] = {}
    group_detail:  Dict[str, list]  = {}

    for group_name, agents in AGENT_GROUPS.items():
        sigs = []
        for a in agents:
            if a in agent_map and not bool(agent_map[a].get("unavailable")):
                sigs.append(float(SIGNAL_TO_SCORE.get(int(agent_map[a].get("signal", 0)), 0.0)))
        if sigs:
            group_signals[group_name] = float(np.mean(sigs))
            group_detail[group_name]  = sigs
        else:
            group_signals[group_name] = 0.0
            group_detail[group_name]  = []

    vals = list(group_signals.values())
    if len(vals) >= 2:
        conflict_score = float(np.std(vals, ddof=0))
    else:
        conflict_score = 0.0

    if conflict_score >= _CONFLICT_SKIP:
        verdict        = "CONFLICT"
        confidence_adj = 0.60
    elif conflict_score >= _CONFLICT_WATCH:
        verdict        = "WATCH"
        confidence_adj = 0.80
    else:
        verdict        = "PASS"
        confidence_adj = 1.0

    # 中文說明（供 /why 顯示）
    dir_map = {v: ("看多" if v > 0.1 else ("看空" if v < -0.1 else "中立"))
               for v in group_signals.values()}
    explanation_parts = []
    for gname, sig in group_signals.items():
        label = {"technical": "技術面", "institutional": "法人籌碼", "fundamental": "基本面"}.get(gname, gname)
        direction = "看多" if sig > 0.1 else ("看空" if sig < -0.1 else "中立")
        explanation_parts.append(f"{label}{direction}({sig:+.2f})")
    explanation = "；".join(explanation_parts)

    return {
        "verdict":        verdict,
        "confidence_adj": confidence_adj,
        "conflict_score": round(conflict_score, 4),
        "group_signals":  group_signals,
        "explanation":    explanation,
    }


def explain_stock(
    row: pd.Series,
    dq_ctx: Dict[str, object],
    z_map: Dict[str, pd.Series],
    ratio_median: float,
) -> Dict[str, object]:
    """為單支股票產生完整的多智能體分析報告（供 /why TG 指令使用）。

    Returns dict with:
        ticker, supervisor (verdict/confidence_adj/group_signals/explanation/conflict_score),
        agents (per-agent signal/confidence/reasons)
    """
    ticker = str(row.get("stock_id", ""))
    tech   = _tech_agent(row, z_map)
    flow   = _flow_agent(row, z_map, dq_ctx)
    margin = _margin_agent(row, dq_ctx, ratio_median)
    fund   = _fund_agent(row, dq_ctx)
    theme  = _theme_agent(row, dq_ctx)
    agent_map = {"tech": tech, "flow": flow, "margin": margin, "fund": fund, "theme": theme}
    supervisor = _supervisor(agent_map)

    return {
        "ticker":     ticker,
        "supervisor": supervisor,
        "agents":     agent_map,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_weights(weights: Dict[str, float], available_agents: List[str]) -> Dict[str, float]:
    valid = {k: float(weights.get(k, 0.0)) for k in available_agents}
    valid = {k: max(v, 0.0) for k, v in valid.items()}
    total = float(sum(valid.values()))
    if total <= 0:
        eq = 1.0 / len(available_agents) if available_agents else 0.0
        return {k: eq for k in available_agents}
    return {k: float(v / total) for k, v in valid.items()}


def _summarize_agent_dump(outputs: Dict[str, Dict[str, Dict[str, object]]]) -> Dict[str, object]:
    agent_summary: Dict[str, Dict[str, object]] = {}
    for agent in AGENT_NAMES:
        rows = [stock_outputs[agent] for stock_outputs in outputs.values() if agent in stock_outputs]
        if not rows:
            agent_summary[agent] = {"unavailable_count": 0, "avg_confidence": 0.0, "signal_distribution": {}}
            continue
        unavailable_count = sum(1 for r in rows if bool(r.get("unavailable")))
        confs = [float(r.get("confidence", 0.0)) for r in rows]
        dist: Dict[str, int] = {}
        for r in rows:
            key = str(int(r.get("signal", 0)))
            dist[key] = dist.get(key, 0) + 1
        agent_summary[agent] = {
            "unavailable_count": unavailable_count,
            "avg_confidence":    float(np.mean(confs)) if confs else 0.0,
            "signal_distribution": dist,
        }

    # Supervisor 統計
    pass_c = watch_c = conflict_c = 0
    for ticker_outputs in outputs.values():
        sv = _supervisor(ticker_outputs)
        if sv["verdict"] == "PASS":
            pass_c += 1
        elif sv["verdict"] == "WATCH":
            watch_c += 1
        else:
            conflict_c += 1

    return {
        "agents": agent_summary,
        "supervisor": {
            "PASS": pass_c,
            "WATCH": watch_c,
            "CONFLICT": conflict_c,
        },
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_multi_agent_selection(
    feature_df: pd.DataFrame,
    stock_ids: pd.Series,
    pick_date: date,
    topn: int,
    config,
    dq_ctx: dict,
    selection_meta: dict,
    model_score_map: Dict[str, float] | None = None,
) -> Tuple[pd.DataFrame, dict]:
    if feature_df.empty or stock_ids.empty:
        return pd.DataFrame(columns=["stock_id", "score", "reason_json"]), {"outputs": {}, "summary": {}}

    df = feature_df.copy().reset_index(drop=True)
    df["stock_id"] = stock_ids.astype(str).reset_index(drop=True)
    z_map = {
        "ret_20":       _zscore(df.get("ret_20",       pd.Series(0.0, index=df.index))),
        "foreign_net_20": _zscore(df.get("foreign_net_20", pd.Series(0.0, index=df.index))),
    }
    ratio_median = float(pd.to_numeric(df.get("margin_short_ratio"), errors="coerce").median(skipna=True) or 0.2)

    requested_weights = dict(getattr(config, "multi_agent_weights", {}) or {})
    for a in AGENT_NAMES:
        requested_weights.setdefault(a, 0.0)

    model_alignment_weight = float(getattr(config, "ma_model_alignment_weight", 0.0))
    model_z_map: Dict[str, float] = {}
    if model_alignment_weight > 0.0 and model_score_map:
        raw_scores = pd.Series(model_score_map)
        std = float(raw_scores.std(skipna=True) or 0.0)
        if std > 0:
            mean = float(raw_scores.mean(skipna=True))
            model_z_map = {k: float((v - mean) / std) for k, v in model_score_map.items()}

    outputs: Dict[str, Dict[str, Dict[str, object]]] = {}
    rows: List[Dict[str, object]] = []

    for idx, row in df.iterrows():
        ticker = str(row["stock_id"])
        tech   = _tech_agent(row, z_map)
        flow   = _flow_agent(row, z_map, dq_ctx)
        margin = _margin_agent(row, dq_ctx, ratio_median)
        fund   = _fund_agent(row, dq_ctx)
        theme  = _theme_agent(row, dq_ctx)
        agent_map = {"tech": tech, "flow": flow, "margin": margin, "fund": fund, "theme": theme}
        outputs[ticker] = agent_map

        # Supervisor 衝突檢測
        supervisor = _supervisor(agent_map)
        confidence_adj = supervisor["confidence_adj"]

        available    = [a for a, o in agent_map.items() if not bool(o.get("unavailable"))]
        weights_used = _normalize_weights(requested_weights, available)
        final_score  = 0.0
        risk_flags   = set()
        for name, out in agent_map.items():
            signal = int(out.get("signal", 0))
            conf   = float(out.get("confidence", 0.0))
            risk_flags.update(out.get("risk_flags", []))
            if name not in weights_used:
                continue
            final_score += float(weights_used[name]) * float(SIGNAL_TO_SCORE.get(signal, 0.0) * conf)

        # 套用 supervisor 信心調整
        final_score *= confidence_adj

        if "high_drawdown" in risk_flags:
            final_score *= 0.9
        if "liquidity_risk" in risk_flags:
            final_score *= 0.9

        if model_alignment_weight > 0.0 and ticker in model_z_map:
            final_score += model_alignment_weight * model_z_map[ticker]

        meta = dict(selection_meta or {})
        meta.update({
            "pick_date":             pick_date.isoformat(),
            "selection_mode":        "multi_agent",
            "dq_ctx":                dict(dq_ctx or {}),
            "weights_used":          weights_used,
            "weights_requested":     requested_weights,
            "model_alignment_weight":model_alignment_weight,
            "supervisor":            supervisor,
        })
        rows.append({
            "stock_id":    ticker,
            "score":       float(final_score),
            "reason_json": {
                "_selection_meta": meta,
                "agents":  agent_map,
                "explain": (
                    f"multi-agent score={final_score:.6f} "
                    f"({supervisor['verdict']}, conf_adj={confidence_adj:.2f})"
                ),
            },
        })

    picks_df = (
        pd.DataFrame(rows)
        .sort_values(["score", "stock_id"], ascending=[False, True])
        .head(topn)
        .reset_index(drop=True)
    )
    return picks_df, {"outputs": outputs, "summary": _summarize_agent_dump(outputs)}
