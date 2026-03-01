from __future__ import annotations

from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


AGENT_NAMES = ["tech", "flow", "margin", "fund", "theme"]
SIGNAL_TO_SCORE = {-2: -1.0, -1: -0.5, 0: 0.0, 1: 0.5, 2: 1.0}
AGENT_REQUIRED_COLUMNS = {
    "tech": ["ret_20", "breakout_20", "rsi_14", "macd_hist", "drawdown_60", "vol_ratio_20"],
    "flow": ["foreign_net_20", "trust_net_20", "dealer_net_20", "chip_flow_intensity_20"],
    "margin": ["margin_balance_chg_20", "short_balance_chg_20", "margin_short_ratio"],
    "fund": ["fund_revenue_yoy", "fund_revenue_mom", "fund_revenue_trend_3m"],
    "theme": ["theme_hot_score", "theme_return_20", "theme_turnover_ratio"],
}


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

    ret_20 = float(pd.to_numeric(row.get("ret_20"), errors="coerce") or 0.0)
    breakout = float(pd.to_numeric(row.get("breakout_20"), errors="coerce") or 0.0)
    macd = float(pd.to_numeric(row.get("macd_hist"), errors="coerce") or 0.0)
    drawdown = float(pd.to_numeric(row.get("drawdown_60"), errors="coerce") or 0.0)
    z_ret = float(z_map["ret_20"].loc[row.name])

    if ret_20 > 0.06 and breakout > 0 and macd > 0 and drawdown > -0.18:
        signal = 2
    elif ret_20 > 0 and macd >= 0:
        signal = 1
    elif ret_20 < -0.08 and macd < 0 and drawdown < -0.28:
        signal = -2
    elif ret_20 < 0:
        signal = -1
    else:
        signal = 0
    conf = _clip01((abs(z_ret) + (1.0 if breakout > 0 else 0.0) + (1.0 if macd > 0 else 0.0)) / 4.0)
    out["signal"] = signal
    out["confidence"] = conf
    out["reasons"] = [f"ret_20={ret_20:.4f}", f"breakout_20={breakout:.4f}", f"macd_hist={macd:.4f}"]
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

    foreign = float(pd.to_numeric(row.get("foreign_net_20"), errors="coerce") or 0.0)
    intensity = float(pd.to_numeric(row.get("chip_flow_intensity_20"), errors="coerce") or 0.0)
    z_foreign = float(z_map["foreign_net_20"].loc[row.name])
    if foreign > 0 and intensity > 0.002 and z_foreign > 0.5:
        signal = 2
    elif foreign > 0:
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
    out["reasons"] = [f"foreign_net_20={foreign:.2f}", f"chip_flow_intensity_20={intensity:.6f}"]
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

    margin_chg = float(pd.to_numeric(row.get("margin_balance_chg_20"), errors="coerce") or 0.0)
    short_chg = float(pd.to_numeric(row.get("short_balance_chg_20"), errors="coerce") or 0.0)
    ratio = float(pd.to_numeric(row.get("margin_short_ratio"), errors="coerce") or 0.0)

    if margin_chg > 0 and short_chg < 0 and ratio < 0.25:
        signal = 2
    elif margin_chg > 0:
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
        f"margin_balance_chg_20={margin_chg:.4f}",
        f"short_balance_chg_20={short_chg:.4f}",
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

    yoy = float(pd.to_numeric(row.get("fund_revenue_yoy"), errors="coerce") or 0.0)
    mom = float(pd.to_numeric(row.get("fund_revenue_mom"), errors="coerce") or 0.0)
    trend = float(pd.to_numeric(row.get("fund_revenue_trend_3m"), errors="coerce") or 0.0)

    if yoy > 0.15 and trend > 0:
        signal = 2
    elif yoy > 0:
        signal = 1
    elif yoy < -0.1 and trend < 0:
        signal = -2
    elif yoy < 0:
        signal = -1
    else:
        signal = 0
    conf = _clip01(min((abs(yoy) * 2.5 + abs(trend) * 2.0 + abs(mom)), 1.0))
    out["signal"] = signal
    out["confidence"] = conf
    out["reasons"] = [f"fund_revenue_yoy={yoy:.4f}", f"fund_revenue_trend_3m={trend:.4f}"]
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

    hot = float(pd.to_numeric(row.get("theme_hot_score"), errors="coerce") or 0.0)
    ret20 = float(pd.to_numeric(row.get("theme_return_20"), errors="coerce") or 0.0)
    turnover_ratio = float(pd.to_numeric(row.get("theme_turnover_ratio"), errors="coerce") or 0.0)

    if hot > 1.0 and ret20 > 0:
        signal = 2
    elif ret20 > 0:
        signal = 1
    elif hot > 1.0 and ret20 < 0:
        signal = -2
    elif ret20 < 0:
        signal = -1
    else:
        signal = 0
    conf = _clip01(min((abs(ret20) * 4 + abs(hot) * 0.3 + abs(turnover_ratio) * 2), 1.0))
    out["signal"] = signal
    out["confidence"] = conf
    out["reasons"] = [f"theme_hot_score={hot:.4f}", f"theme_return_20={ret20:.4f}"]
    out["risk_flags"] = _risk_flags(row)
    return out


def _normalize_weights(weights: Dict[str, float], available_agents: List[str]) -> Dict[str, float]:
    valid = {k: float(weights.get(k, 0.0)) for k in available_agents}
    valid = {k: max(v, 0.0) for k, v in valid.items()}
    total = float(sum(valid.values()))
    if total <= 0:
        eq = 1.0 / len(available_agents) if available_agents else 0.0
        return {k: eq for k in available_agents}
    return {k: float(v / total) for k, v in valid.items()}


def _summarize_agent_dump(outputs: Dict[str, Dict[str, Dict[str, object]]]) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for agent in AGENT_NAMES:
        rows = [stock_outputs[agent] for stock_outputs in outputs.values() if agent in stock_outputs]
        if not rows:
            summary[agent] = {"unavailable_count": 0, "avg_confidence": 0.0, "signal_distribution": {}}
            continue
        unavailable_count = sum(1 for r in rows if bool(r.get("unavailable")))
        confs = [float(r.get("confidence", 0.0)) for r in rows]
        dist: Dict[str, int] = {}
        for r in rows:
            key = str(int(r.get("signal", 0)))
            dist[key] = dist.get(key, 0) + 1
        summary[agent] = {
            "unavailable_count": unavailable_count,
            "avg_confidence": float(np.mean(confs)) if confs else 0.0,
            "signal_distribution": dist,
        }
    return summary


def run_multi_agent_selection(
    feature_df: pd.DataFrame,
    stock_ids: pd.Series,
    pick_date: date,
    topn: int,
    config,
    dq_ctx: dict,
    selection_meta: dict,
) -> Tuple[pd.DataFrame, dict]:
    if feature_df.empty or stock_ids.empty:
        return pd.DataFrame(columns=["stock_id", "score", "reason_json"]), {"outputs": {}, "summary": {}}

    df = feature_df.copy().reset_index(drop=True)
    df["stock_id"] = stock_ids.astype(str).reset_index(drop=True)
    z_map = {
        "ret_20": _zscore(df.get("ret_20", pd.Series(0.0, index=df.index))),
        "foreign_net_20": _zscore(df.get("foreign_net_20", pd.Series(0.0, index=df.index))),
    }
    ratio_median = float(pd.to_numeric(df.get("margin_short_ratio"), errors="coerce").median(skipna=True) or 0.2)

    requested_weights = dict(getattr(config, "multi_agent_weights", {}) or {})
    for a in AGENT_NAMES:
        requested_weights.setdefault(a, 0.0)

    outputs: Dict[str, Dict[str, Dict[str, object]]] = {}
    rows: List[Dict[str, object]] = []

    for idx, row in df.iterrows():
        ticker = str(row["stock_id"])
        tech = _tech_agent(row, z_map)
        flow = _flow_agent(row, z_map, dq_ctx)
        margin = _margin_agent(row, dq_ctx, ratio_median)
        fund = _fund_agent(row, dq_ctx)
        theme = _theme_agent(row, dq_ctx)
        agent_map = {"tech": tech, "flow": flow, "margin": margin, "fund": fund, "theme": theme}
        outputs[ticker] = agent_map

        available = [a for a, o in agent_map.items() if not bool(o.get("unavailable"))]
        weights_used = _normalize_weights(requested_weights, available)
        final_score = 0.0
        risk_flags = set()
        for name, out in agent_map.items():
            signal = int(out.get("signal", 0))
            conf = float(out.get("confidence", 0.0))
            risk_flags.update(out.get("risk_flags", []))
            if name not in weights_used:
                continue
            final_score += float(weights_used[name]) * float(SIGNAL_TO_SCORE.get(signal, 0.0) * conf)
        if "high_drawdown" in risk_flags:
            final_score *= 0.9
        if "liquidity_risk" in risk_flags:
            final_score *= 0.9

        meta = dict(selection_meta or {})
        meta.update(
            {
                "pick_date": pick_date.isoformat(),
                "selection_mode": "multi_agent",
                "dq_ctx": dict(dq_ctx or {}),
                "weights_used": weights_used,
                "weights_requested": requested_weights,
            }
        )
        rows.append(
            {
                "stock_id": ticker,
                "score": float(final_score),
                "reason_json": {
                    "_selection_meta": meta,
                    "agents": agent_map,
                    "explain": f"deterministic multi-agent score={final_score:.6f}",
                },
            }
        )

    picks_df = pd.DataFrame(rows).sort_values(["score", "stock_id"], ascending=[False, True]).head(topn).reset_index(drop=True)
    return picks_df, {"outputs": outputs, "summary": _summarize_agent_dump(outputs)}
