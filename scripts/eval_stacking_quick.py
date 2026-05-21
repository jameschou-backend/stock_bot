"""Stage 6.1 quick eval：stacking ensemble vs LightGBM-only 對 production 資料的效益。

對既有 features parquet（已 enrich 含 fracdiff + PER）跑：
  1. 訓三個 base model（LightGBM + XGBoost + CatBoost）
  2. 計算 base model 之間 prediction correlation
  3. 在 held-out（後 20%）跑 cross-sectional IC：
     - 三個 base model 各自 IC
     - rank-averaged ensemble IC
  4. 報告 ensemble IC lift 多少

判定：
  - 若 ensemble IC > max(base IC) by >= 5% → 有 diversity gain，值得整合進 backtest
  - 否則 stacking 帶不來 alpha，跳過

不跑完整 backtest（太久）。如果 IC 顯示 promising 才下一步做 backtest 對照。

用法：
    python scripts/eval_stacking_quick.py
    python scripts/eval_stacking_quick.py --start 2020-01-01
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from sqlalchemy import select

from app.db import get_session
from app.models import Label
from skills.build_features import PRUNED_FEATURE_COLS
from skills.feature_store import FeatureStore
from skills.stacking import (
    StackingEnsemble,
    _rank_to_percentile,
    base_model_correlation,
    train_stacking_ensemble,
)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--val-frac", type=float, default=0.2)
    return p.parse_args()


def _cross_sectional_ic(df: pd.DataFrame, score_col: str, label_col: str,
                         min_stocks: int = 30) -> dict:
    """Per-trading_date Spearman rank IC + aggregate stats."""
    sub = df.dropna(subset=[score_col, label_col])
    ics = []
    for d, g in sub.groupby("trading_date", sort=True):
        if len(g) < min_stocks:
            continue
        ic, _ = stats.spearmanr(g[score_col].to_numpy(), g[label_col].to_numpy())
        if not np.isnan(ic):
            ics.append(float(ic))
    if len(ics) < 5:
        return {"n_days": len(ics), "ic_mean": None, "icir": None}
    s = pd.Series(ics)
    mean = s.mean()
    std = s.std(ddof=1)
    icir = (mean / std * math.sqrt(len(ics))) if std > 1e-9 else 0
    return {
        "n_days": len(ics),
        "ic_mean": float(mean),
        "ic_std": float(std),
        "icir": float(icir),
        "positive_rate": float((s > 0).mean()),
    }


def main() -> int:
    args = _parse_args()
    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date() if args.end else date.today()

    print(f"=== Stage 6.1 stacking quick eval ===")
    print(f"  期間: {start} ~ {end}")
    print(f"  val_frac: {args.val_frac}")
    print()

    print("  載入 features + labels ...", end="", flush=True)
    fs = FeatureStore()
    feat = fs.read(start, end)
    feat["trading_date"] = pd.to_datetime(feat["trading_date"]).dt.date

    with get_session() as s:
        q = (select(Label.stock_id, Label.trading_date, Label.future_ret_h)
             .where(Label.trading_date >= start)
             .where(Label.trading_date <= end))
        lab = pd.read_sql(q, s.get_bind())
    lab["trading_date"] = pd.to_datetime(lab["trading_date"]).dt.date
    lab["future_ret_h"] = pd.to_numeric(lab["future_ret_h"], errors="coerce")

    df = feat.merge(lab, on=["stock_id", "trading_date"], how="inner")
    df = df.dropna(subset=["future_ret_h"])
    df = df.sort_values("trading_date").reset_index(drop=True)
    print(f" {len(df):,} rows")

    # 只取 PRUNED_FEATURE_COLS（與生產一致）
    feature_cols = [c for c in PRUNED_FEATURE_COLS if c in df.columns]
    missing = set(PRUNED_FEATURE_COLS) - set(feature_cols)
    if missing:
        print(f"  ⚠ PRUNED 但 features 內沒有: {sorted(missing)}")
    print(f"  使用 features: {len(feature_cols)} 個")

    # NaN 處理（fillna 0，跟 backtest 一致）
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["future_ret_h"].to_numpy()

    # 時序切分
    n = len(df)
    val_n = int(n * args.val_frac)
    train_X = X.iloc[:-val_n].copy()
    train_y = y[:-val_n]
    val_X = X.iloc[-val_n:].copy()
    val_y = y[-val_n:]
    val_meta = df.iloc[-val_n:][["stock_id", "trading_date", "future_ret_h"]].reset_index(drop=True)

    print(f"  train: {len(train_X):,} samples ({train_X.shape[1]} features)")
    print(f"  val:   {len(val_X):,} samples ({val_meta['trading_date'].min()} ~ {val_meta['trading_date'].max()})")
    print()

    # ── 訓練 stacking ensemble ──
    print("─── 訓練 3 個 base models ───")
    t0 = time.monotonic()
    ens = train_stacking_ensemble(train_X, train_y, val_X, val_y, seed=42)
    train_secs = time.monotonic() - t0
    print(f"  訓練完成 {train_secs:.1f}s")
    print(f"  engines: {ens.engines_used}")
    print()

    # ── Base model 相關性 ──
    print("─── Base model prediction correlation ───")
    corr = base_model_correlation(ens, val_X)
    print(corr.round(4).to_string())
    print()

    # ── 各 base model + ensemble IC ──
    print("─── IC on validation set (per-trading_date cross-sectional Spearman) ───")
    Xf = val_X[ens.feature_names]
    val_meta = val_meta.copy()
    ic_summary = []
    base_ics = []
    for name, model in ens.base_models.items():
        raw = model.predict(Xf)
        val_meta[f"score_{name}"] = raw
        stats_d = _cross_sectional_ic(val_meta, f"score_{name}", "future_ret_h")
        ic_summary.append({"model": name, **stats_d})
        if stats_d.get("ic_mean") is not None:
            base_ics.append(stats_d["ic_mean"])

    # Ensemble: rank average
    pcts = []
    for name in ens.base_models:
        raw = val_meta[f"score_{name}"].to_numpy()
        p = _rank_to_percentile(raw, by_group=val_meta["trading_date"].to_numpy())
        pcts.append(p)
    ensemble_score = np.mean(pcts, axis=0)
    val_meta["score_ensemble"] = ensemble_score
    ens_stats = _cross_sectional_ic(val_meta, "score_ensemble", "future_ret_h")
    ic_summary.append({"model": "ENSEMBLE", **ens_stats})

    print(f"\n  {'model':<14} {'n_days':<8} {'IC mean':<12} {'ICIR':<10} {'pos_rate'}")
    print("  " + "-" * 56)
    for r in ic_summary:
        ic = r.get("ic_mean")
        icir = r.get("icir")
        pr = r.get("positive_rate")
        ic_s = f"{ic:+.4f}" if ic is not None else "N/A"
        icir_s = f"{icir:+.3f}" if icir is not None else "N/A"
        pr_s = f"{pr:.1%}" if pr is not None else "N/A"
        marker = " ⭐" if r["model"] == "ENSEMBLE" else ""
        print(f"  {r['model']:<14} {r['n_days']:<8} {ic_s:<12} {icir_s:<10} {pr_s}{marker}")
    print()

    # ── 判定 ──
    if base_ics and ens_stats.get("ic_mean") is not None:
        best_base = max(base_ics, key=abs)
        ens_ic = ens_stats["ic_mean"]
        # 我們知道台股是 negative IC（mean-reversion），用 abs 比
        lift = (abs(ens_ic) - abs(best_base)) / max(abs(best_base), 1e-9)
        print(f"=== 判定 ===")
        print(f"  Best base IC: {best_base:+.4f}")
        print(f"  Ensemble IC : {ens_ic:+.4f}")
        print(f"  Lift: {lift:+.1%}")
        if lift >= 0.05:
            print(f"  ✅ ensemble 顯著優於 base（>= 5% lift），建議整合進 backtest 跑 60mo 對照")
        elif lift >= 0.01:
            print(f"  △ ensemble 微幅優於 base（1-5%），效益邊緣")
        elif lift > -0.05:
            print(f"  ⚠️ ensemble 與 base 接近（在 noise 範圍）")
        else:
            print(f"  ❌ ensemble 反而劣於 base，不採用 stacking")
    return 0


if __name__ == "__main__":
    sys.exit(main())
