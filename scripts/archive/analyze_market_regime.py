#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from app.models import Feature, RawPrice
from scripts.evaluate_experiment import (
    ARTIFACTS_DIR,
    _load_adjust_factors,
    _load_price_frame,
    _merge_adjusted_close,
    _monthly_rebalance_dates,
    _period_return,
    _to_config,
    run_experiment,
)


def _load_round4() -> Dict[str, object]:
    return json.loads((ARTIFACTS_DIR / "round4_summary.json").read_text(encoding="utf-8"))


def _build_market_regime(start_date: date, end_date: date, rb_dates: List[date]) -> pd.DataFrame:
    with get_session() as session:
        stmt = (
            select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close)
            .where(RawPrice.trading_date.between(start_date, end_date))
            .order_by(RawPrice.stock_id, RawPrice.trading_date)
        )
        df = pd.read_sql(stmt, session.get_bind())
    if df.empty:
        return pd.DataFrame()
    df["stock_id"] = df["stock_id"].astype(str)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values(["stock_id", "trading_date"])
    df["ret1"] = df.groupby("stock_id")["close"].pct_change()
    daily = df.groupby("trading_date")["ret1"].median().reset_index().sort_values("trading_date")
    daily["ret1"] = daily["ret1"].fillna(0.0)
    daily["mkt_index"] = (1.0 + daily["ret1"]).cumprod()
    daily["ma20"] = daily["mkt_index"].rolling(20, min_periods=5).mean()
    daily["ma60"] = daily["mkt_index"].rolling(60, min_periods=20).mean()
    daily["slope20"] = daily["mkt_index"] / daily["mkt_index"].shift(20) - 1.0
    daily["vol20"] = daily["ret1"].rolling(20, min_periods=10).std() * np.sqrt(252.0)
    daily["dd60"] = daily["mkt_index"] / daily["mkt_index"].rolling(60, min_periods=20).max() - 1.0

    reg = daily[daily["trading_date"].isin(rb_dates)].copy()
    vol_threshold = float(reg["vol20"].median(skipna=True)) if not reg.empty else 0.0
    reg["trend_regime"] = np.where(reg["slope20"].abs() >= 0.03, "趨勢盤", "震盪盤")
    reg["strength_regime"] = np.where((reg["mkt_index"] >= reg["ma60"]) & (reg["dd60"] > -0.08), "強勢盤", "弱勢盤")
    reg["vol_regime"] = np.where(reg["vol20"] >= vol_threshold, "高波動", "低波動")
    reg = reg[["trading_date", "trend_regime", "strength_regime", "vol_regime", "slope20", "vol20", "dd60"]]
    return reg


def _load_picks_map(experiment_id: str) -> Dict[str, List[Dict[str, object]]]:
    rows = json.loads((ARTIFACTS_DIR / f"experiment_picks_{experiment_id}.json").read_text(encoding="utf-8"))
    out: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        out.setdefault(str(r["date"]), []).append(r)
    return out


def _regime_perf_table(per_period: List[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    df = pd.DataFrame(per_period)
    if df.empty:
        return []
    out = []
    for reg, g in df.groupby(key):
        m = pd.to_numeric(g["model_return"], errors="coerce")
        a = pd.to_numeric(g["multi_agent_return"], errors="coerce")
        out.append(
            {
                "regime": reg,
                "periods": int(len(g)),
                "model_avg_return": float(m.mean()),
                "multi_agent_avg_return": float(a.mean()),
                "multi_agent_minus_model": float(a.mean() - m.mean()),
                "model_win_rate": float((m > 0).mean()),
                "multi_agent_win_rate": float((a > 0).mean()),
            }
        )
    return sorted(out, key=lambda x: x["regime"])


def _agent_impact_by_regime(picks_map: Dict[str, List[Dict[str, object]]], regime_map: Dict[str, Dict[str, str]], key: str) -> List[Dict[str, object]]:
    rows = []
    for d, rows_d in picks_map.items():
        if d not in regime_map:
            continue
        reg = regime_map[d][key]
        for r in rows_d:
            agents = ((r.get("reason_json") or {}).get("agents") or {})
            meta = ((r.get("reason_json") or {}).get("_selection_meta") or {})
            wu = meta.get("weights_used", {}) if isinstance(meta, dict) else {}
            for a in ["tech", "flow", "margin", "fund"]:
                ao = agents.get(a, {}) or {}
                sig = float(ao.get("signal", 0.0))
                conf = float(ao.get("confidence", 0.0))
                contrib = float(wu.get(a, 0.0)) * sig * conf
                rows.append({"regime": reg, "agent": a, "abs_contrib": abs(contrib), "signed_contrib": contrib})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out = []
    for (reg, a), g in df.groupby(["regime", "agent"]):
        out.append(
            {
                "regime": reg,
                "agent": a,
                "avg_abs_contrib": float(g["abs_contrib"].mean()),
                "avg_signed_contrib": float(g["signed_contrib"].mean()),
            }
        )
    return sorted(out, key=lambda x: (x["regime"], -x["avg_abs_contrib"]))


def analyze_market_regime() -> Dict[str, object]:
    r4 = _load_round4()
    s = r4["split"]
    start_date = date.fromisoformat(s["development_start"])
    end_date = date.fromisoformat(s["validation_end"])
    cfg_base = load_config()
    weights = dict(r4["validation_model_vs_best_multi_agent"]["best_multi_agent_by_sharpe"]["weights_requested"])
    model_cfg = _to_config(cfg_base, {"selection_mode": "model", "data_quality_mode": "research", "topn": cfg_base.topn}, cfg_base.topn)
    ma_cfg = _to_config(
        cfg_base,
        {"selection_mode": "multi_agent", "data_quality_mode": "research", "topn": cfg_base.topn, "multi_agent_weights": weights},
        cfg_base.topn,
    )
    model_id = "regime_model_round5"
    ma_id = "regime_ma_round5"
    run_experiment(model_id, start_date, end_date, model_cfg, resume=False)
    run_experiment(ma_id, start_date, end_date, ma_cfg, resume=False)

    with get_session() as session:
        rows = (
            session.query(Feature.trading_date)
            .filter(Feature.trading_date.between(start_date, end_date))
            .distinct()
            .order_by(Feature.trading_date)
            .all()
        )
        rb_dates = _monthly_rebalance_dates([r[0] for r in rows])
        px_df = _merge_adjusted_close(
            _load_price_frame(session, start_date, end_date),
            _load_adjust_factors(session, start_date, end_date),
            bool(getattr(cfg_base, "use_adjusted_price", True)),
        )

    regime_df = _build_market_regime(start_date, end_date, rb_dates)
    regime_map = {
        str(r["trading_date"]): {
            "trend_regime": r["trend_regime"],
            "strength_regime": r["strength_regime"],
            "vol_regime": r["vol_regime"],
        }
        for _, r in regime_df.iterrows()
    }

    model_picks = _load_picks_map(model_id)
    ma_picks = _load_picks_map(ma_id)
    per_period = []
    for i in range(len(rb_dates) - 1):
        d1, d2 = rb_dates[i], rb_dates[i + 1]
        k1 = d1.isoformat()
        if k1 not in regime_map:
            continue
        mret, _ = _period_return([str(x["stock_id"]) for x in model_picks.get(k1, [])], d1, d2, px_df, float(cfg_base.transaction_cost_pct))
        aret, _ = _period_return([str(x["stock_id"]) for x in ma_picks.get(k1, [])], d1, d2, px_df, float(cfg_base.transaction_cost_pct))
        per_period.append(
            {
                "entry_date": k1,
                "exit_date": d2.isoformat(),
                "trend_regime": regime_map[k1]["trend_regime"],
                "strength_regime": regime_map[k1]["strength_regime"],
                "vol_regime": regime_map[k1]["vol_regime"],
                "model_return": mret,
                "multi_agent_return": aret,
                "multi_agent_minus_model": (aret or 0.0) - (mret or 0.0),
            }
        )

    trend_perf = _regime_perf_table(per_period, "trend_regime")
    strength_perf = _regime_perf_table(per_period, "strength_regime")
    vol_perf = _regime_perf_table(per_period, "vol_regime")
    best_regimes = sorted(
        [*trend_perf, *strength_perf, *vol_perf],
        key=lambda x: x["multi_agent_minus_model"],
        reverse=True,
    )[:3]

    payload = {
        "period": {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
        "multi_agent_weights": weights,
        "per_period": per_period,
        "performance_by_regime": {
            "trend_regime": trend_perf,
            "strength_regime": strength_perf,
            "vol_regime": vol_perf,
        },
        "regimes_multi_agent_relatively_strong": best_regimes,
        "agent_impact_by_regime": {
            "trend_regime": _agent_impact_by_regime(ma_picks, regime_map, "trend_regime"),
            "strength_regime": _agent_impact_by_regime(ma_picks, regime_map, "strength_regime"),
            "vol_regime": _agent_impact_by_regime(ma_picks, regime_map, "vol_regime"),
        },
    }

    (ARTIFACTS_DIR / "market_regime_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    lines = [
        "# Market Regime Analysis",
        "",
        f"- period: `{start_date.isoformat()} ~ {end_date.isoformat()}`",
        f"- multi_agent_weights(theme=0): `{weights}`",
        "",
        "## Trend Regime",
        "| regime | periods | model_avg_return | multi_agent_avg_return | ma_minus_model |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in trend_perf:
        lines.append(f"| {r['regime']} | {r['periods']} | {r['model_avg_return']} | {r['multi_agent_avg_return']} | {r['multi_agent_minus_model']} |")
    lines.extend(["", "## Strength Regime", "| regime | periods | model_avg_return | multi_agent_avg_return | ma_minus_model |", "|---|---:|---:|---:|---:|"])
    for r in strength_perf:
        lines.append(f"| {r['regime']} | {r['periods']} | {r['model_avg_return']} | {r['multi_agent_avg_return']} | {r['multi_agent_minus_model']} |")
    lines.extend(["", "## Volatility Regime", "| regime | periods | model_avg_return | multi_agent_avg_return | ma_minus_model |", "|---|---:|---:|---:|---:|"])
    for r in vol_perf:
        lines.append(f"| {r['regime']} | {r['periods']} | {r['model_avg_return']} | {r['multi_agent_avg_return']} | {r['multi_agent_minus_model']} |")
    lines.extend(["", "## Multi-Agent 相對較強 Regime"])
    for r in best_regimes:
        lines.append(f"- {r['regime']}: ma_minus_model={r['multi_agent_minus_model']:.4f}")
    lines.extend(["", "## Agent Impact by Regime（avg_abs_contrib）"])
    for key, rows in payload["agent_impact_by_regime"].items():
        lines.append(f"- {key}: `{rows[:8]}`")
    (ARTIFACTS_DIR / "market_regime_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    payload = analyze_market_regime()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
