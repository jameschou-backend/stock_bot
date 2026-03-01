#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from app.models import Feature
from scripts.evaluate_experiment import (
    ARTIFACTS_DIR,
    _fund_snapshot_asof,
    _monthly_rebalance_dates,
    _to_config,
    run_experiment,
)
from skills.daily_pick import _parse_features


ROUND4_MATRIX = PROJECT_ROOT / "experiments" / "multi_agent_round4.yaml"
FUND_COLS = ["fund_revenue_mom", "fund_revenue_yoy", "fund_revenue_trend_3m"]
KEY_METRICS = ["total_return", "sharpe", "max_drawdown", "turnover", "picks_stability"]


def _load_matrix(path: Path) -> Dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _numeric(v):
    return float(v) if isinstance(v, (int, float)) and np.isfinite(float(v)) else None


def _compute_split(start_date: date, end_date: date) -> Dict[str, object]:
    with get_session() as session:
        rows = (
            session.query(Feature.trading_date)
            .filter(Feature.trading_date.between(start_date, end_date))
            .distinct()
            .order_by(Feature.trading_date)
            .all()
        )
    rb_dates = _monthly_rebalance_dates([r[0] for r in rows])
    if len(rb_dates) < 4:
        raise ValueError("Not enough rebalance dates for round4 70/30 split")
    split_idx = max(1, min(int(len(rb_dates) * 0.7), len(rb_dates) - 2))
    return {
        "rebalance_dates_total": len(rb_dates),
        "development_start": start_date.isoformat(),
        "development_end": (rb_dates[split_idx] - timedelta(days=1)).isoformat(),
        "validation_start": rb_dates[split_idx].isoformat(),
        "validation_end": end_date.isoformat(),
        "development_rebalance_dates": [d.isoformat() for d in rb_dates[:split_idx]],
        "validation_rebalance_dates": [d.isoformat() for d in rb_dates[split_idx:]],
    }


def _fund_alignment_debug(start_date: date, end_date: date) -> Dict[str, object]:
    rows_out = []
    with get_session() as session:
        rows = (
            session.query(Feature.trading_date)
            .filter(Feature.trading_date.between(start_date, end_date))
            .distinct()
            .order_by(Feature.trading_date)
            .all()
        )
        rb_dates = _monthly_rebalance_dates([r[0] for r in rows])
        for rb in rb_dates:
            feat_stmt = select(Feature.stock_id, Feature.features_json).where(Feature.trading_date == rb)
            feat_df = pd.read_sql(feat_stmt, session.get_bind())
            if feat_df.empty:
                continue
            feat_df["stock_id"] = feat_df["stock_id"].astype(str)
            parsed = _parse_features(feat_df["features_json"])
            before = {}
            for c in FUND_COLS:
                s = pd.to_numeric(parsed[c], errors="coerce") if c in parsed.columns else pd.Series(np.nan, index=parsed.index)
                before[c] = float(s.notna().mean()) if len(s) else 0.0
            snap = _fund_snapshot_asof(session, rb, feat_df["stock_id"].tolist())
            after = {}
            if snap.empty:
                for c in FUND_COLS:
                    after[c] = 0.0
            else:
                sm = snap.set_index("stock_id")
                for c in FUND_COLS:
                    s = pd.to_numeric(feat_df["stock_id"].map(sm[c]), errors="coerce")
                    after[c] = float(s.notna().mean()) if len(s) else 0.0
            rows_out.append(
                {
                    "rebalance_date": rb.isoformat(),
                    "stocks": int(len(feat_df)),
                    "before_non_null": before,
                    "after_non_null": after,
                }
            )

    payload = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "rows": rows_out,
    }
    jp = ARTIFACTS_DIR / "debug_fund_alignment_round4.json"
    mp = ARTIFACTS_DIR / "debug_fund_alignment_round4.md"
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Debug Fund Alignment Round4",
        "",
        "| rebalance_date | stocks | before_yoy | before_mom | before_trend | after_yoy | after_mom | after_trend |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_out:
        b = r["before_non_null"]
        a = r["after_non_null"]
        lines.append(
            f"| {r['rebalance_date']} | {r['stocks']} | {b['fund_revenue_yoy']:.2%} | {b['fund_revenue_mom']:.2%} | "
            f"{b['fund_revenue_trend_3m']:.2%} | {a['fund_revenue_yoy']:.2%} | {a['fund_revenue_mom']:.2%} | {a['fund_revenue_trend_3m']:.2%} |"
        )
    low_after_core = [
        r
        for r in rows_out
        if min(r["after_non_null"]["fund_revenue_yoy"], r["after_non_null"]["fund_revenue_mom"]) < 0.5
    ]
    trend_missing_dates = [r for r in rows_out if r["after_non_null"]["fund_revenue_trend_3m"] < 0.5]
    lines.extend(
        [
            "",
            f"- 低覆蓋日期數（after yoy/mom < 50%）: `{len(low_after_core)}`",
            f"- trend_3m 仍大量缺失日期數（<50%）: `{len(trend_missing_dates)}`",
            "- 對齊規則：`as-of snapshot (<= rebalance date)`，不使用未來資料（無 lookahead）。",
        ]
    )
    mp.write_text("\n".join(lines), encoding="utf-8")
    return payload


def run_round4() -> Dict[str, object]:
    matrix = _load_matrix(ROUND4_MATRIX)
    base = load_config()
    start_date = date.fromisoformat(str(matrix["start_date"]))
    end_date = date.fromisoformat(str(matrix["end_date"]))
    topn = int(matrix.get("topn", 20))
    split = _compute_split(start_date, end_date)
    dev_start = date.fromisoformat(split["development_start"])
    dev_end = date.fromisoformat(split["development_end"])
    val_start = date.fromisoformat(split["validation_start"])
    val_end = date.fromisoformat(split["validation_end"])

    align_debug = _fund_alignment_debug(start_date, end_date)

    development = []
    validation = []
    for spec in matrix.get("experiments", []):
        cfg = _to_config(base, spec, topn=topn)
        dev_id = f"{spec['name']}_round4_dev"
        val_id = f"{spec['name']}_round4_val"
        dev = run_experiment(dev_id, dev_start, dev_end, cfg, resume=False)
        val = run_experiment(val_id, val_start, val_end, cfg, resume=False)
        development.append(
            {
                "name": spec["name"],
                "selection_mode": cfg.selection_mode,
                "weights_requested": cfg.multi_agent_weights if cfg.selection_mode == "multi_agent" else None,
                "metrics": dev.get("metrics", {}),
            }
        )
        validation.append(
            {
                "name": spec["name"],
                "selection_mode": cfg.selection_mode,
                "weights_requested": cfg.multi_agent_weights if cfg.selection_mode == "multi_agent" else None,
                "metrics": val.get("metrics", {}),
            }
        )

    def pick(rows: List[Dict[str, object]], name: str) -> Dict[str, object]:
        return next(r for r in rows if r["name"] == name)

    dev_3 = pick(development, "ma_3agent_tfm")
    dev_4 = pick(development, "ma_4agent_tfmf")
    val_3 = pick(validation, "ma_3agent_tfm")
    val_4 = pick(validation, "ma_4agent_tfmf")
    val_model = pick(validation, "model_baseline")

    fund_gain = {
        "development_delta_4agent_vs_3agent": {
            k: (_numeric(dev_4["metrics"].get(k)) or 0.0) - (_numeric(dev_3["metrics"].get(k)) or 0.0)
            for k in KEY_METRICS
        },
        "validation_delta_4agent_vs_3agent": {
            k: (_numeric(val_4["metrics"].get(k)) or 0.0) - (_numeric(val_3["metrics"].get(k)) or 0.0)
            for k in KEY_METRICS
        },
    }
    fund_effective_gain = bool(
        fund_gain["validation_delta_4agent_vs_3agent"]["total_return"] > 0
        and fund_gain["validation_delta_4agent_vs_3agent"]["sharpe"] > 0
    )

    payload = {
        "matrix_file": str(ROUND4_MATRIX),
        "split": split,
        "fund_alignment_debug": {
            "file": str(ARTIFACTS_DIR / "debug_fund_alignment_round4.md"),
            "low_coverage_dates_yoy_mom": len(
                [
                    r
                    for r in align_debug["rows"]
                    if min(r["after_non_null"]["fund_revenue_yoy"], r["after_non_null"]["fund_revenue_mom"]) < 0.5
                ]
            ),
            "trend_missing_dates": len(
                [r for r in align_debug["rows"] if r["after_non_null"]["fund_revenue_trend_3m"] < 0.5]
            ),
        },
        "development": development,
        "validation": validation,
        "fund_gain_assessment": {
            **fund_gain,
            "fund_effective_gain": fund_effective_gain,
        },
        "validation_model_vs_best_multi_agent": {
            "model_baseline": val_model["metrics"],
            "best_multi_agent_by_sharpe": max(
                [r for r in validation if r["selection_mode"] == "multi_agent"],
                key=lambda r: _numeric(r["metrics"].get("sharpe")) or -1e9,
            ),
        },
    }

    (ARTIFACTS_DIR / "round4_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    lines = [
        "# Round4 Summary",
        "",
        "## Development",
        "| name | mode | total_return | sharpe | max_drawdown | turnover | picks_stability |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in development:
        m = r["metrics"]
        lines.append(
            f"| {r['name']} | {r['selection_mode']} | {m.get('total_return')} | {m.get('sharpe')} | "
            f"{m.get('max_drawdown')} | {m.get('turnover')} | {m.get('picks_stability')} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "| name | mode | total_return | sharpe | max_drawdown | turnover | picks_stability |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in validation:
        m = r["metrics"]
        lines.append(
            f"| {r['name']} | {r['selection_mode']} | {m.get('total_return')} | {m.get('sharpe')} | "
            f"{m.get('max_drawdown')} | {m.get('turnover')} | {m.get('picks_stability')} |"
        )
    dv = fund_gain["development_delta_4agent_vs_3agent"]
    vv = fund_gain["validation_delta_4agent_vs_3agent"]
    lines.extend(
        [
            "",
            "## 4-agent vs 3-agent（fund 增益）",
            f"- development delta: return={dv['total_return']}, sharpe={dv['sharpe']}, max_dd={dv['max_drawdown']}",
            f"- validation delta: return={vv['total_return']}, sharpe={vv['sharpe']}, max_dd={vv['max_drawdown']}",
            f"- fund_effective_gain: `{fund_effective_gain}`",
            "",
            "## Fund Alignment",
            f"- debug report: `{ARTIFACTS_DIR / 'debug_fund_alignment_round4.md'}`",
        ]
    )
    (ARTIFACTS_DIR / "round4_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    payload = run_round4()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
