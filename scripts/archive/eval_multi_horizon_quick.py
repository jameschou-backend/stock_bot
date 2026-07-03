"""Stage 6.2 quick eval：Multi-Horizon ensemble vs 單 horizon=20 LightGBM。

對既有 features parquet 跑：
  1. 從 raw_prices 計算 5/10/20/40 日 forward return
  2. 訓 4 個 LightGBM（各 horizon 一個）
  3. 在 val set 計算：
     - 各 horizon model 預測 20-day return 的 IC（不論訓練 horizon 用 20d label 評估）
     - Multi-Horizon Ensemble（rank average）IC
  4. 對照 baseline=horizon=20 single LightGBM IC

評估 metric：對 future_ret_20（既有生產 label）做 cross-sectional IC，因為這
是 backtest 的真實目標。

用法：
    python scripts/eval_multi_horizon_quick.py --start 2020-01-01
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
from app.models import Label, RawPrice
from skills.build_features import PRUNED_FEATURE_COLS
from skills.feature_store import FeatureStore
from skills.multi_horizon import (
    DEFAULT_HORIZONS,
    compute_multi_horizon_labels,
    horizon_model_correlation,
    train_multi_horizon_ensemble,
)
from skills.stacking import _rank_to_percentile


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2020-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--horizons", nargs="*", type=int, default=list(DEFAULT_HORIZONS))
    return p.parse_args()


def _cross_sectional_ic(df, score_col, label_col, min_stocks=30):
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
    return {
        "n_days": len(ics),
        "ic_mean": float(mean),
        "ic_std": float(std),
        "icir": (float(mean / std * math.sqrt(len(ics))) if std > 1e-9 else 0),
        "positive_rate": float((s > 0).mean()),
    }


def main() -> int:
    args = _parse_args()
    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date() if args.end else date.today()

    print(f"=== Stage 6.2 multi-horizon quick eval ===")
    print(f"  期間: {start} ~ {end}, val_frac: {args.val_frac}")
    print(f"  horizons: {args.horizons}")
    print()

    # ── 1. 載 features + 既有 future_ret_20 label ──
    print("  載入 features ...", end="", flush=True)
    fs = FeatureStore()
    feat = fs.read(start, end)
    feat["trading_date"] = pd.to_datetime(feat["trading_date"]).dt.date
    print(f" {len(feat):,} rows")

    print("  載入 raw_prices (for multi-horizon labels) ...", end="", flush=True)
    with get_session() as s:
        q = (select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close)
             .where(RawPrice.trading_date >= start)
             .where(RawPrice.trading_date <= end))
        prices = pd.read_sql(q, s.get_bind())
    prices["trading_date"] = pd.to_datetime(prices["trading_date"]).dt.date
    print(f" {len(prices):,} rows")

    print(f"  計算 multi-horizon labels {args.horizons} ...", end="", flush=True)
    t0 = time.monotonic()
    ml_labels = compute_multi_horizon_labels(prices, horizons=args.horizons)
    print(f" {len(ml_labels):,} rows ({time.monotonic()-t0:.1f}s)")

    # ── 2. Merge: features + multi-horizon labels ──
    print("  merge features + labels ...", end="", flush=True)
    df = feat.merge(ml_labels, on=["stock_id", "trading_date"], how="inner")
    df = df.sort_values("trading_date").reset_index(drop=True)
    # 至少有一個 horizon label 非 NaN 才保留
    label_cols = [f"future_ret_{h}" for h in args.horizons]
    df = df.dropna(subset=label_cols, how="all")
    print(f" {len(df):,} rows")

    # ── 3. 準備 features ──
    feature_cols = [c for c in PRUNED_FEATURE_COLS if c in df.columns]
    print(f"  使用 features: {len(feature_cols)} 個")
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    n = len(df)
    val_n = int(n * args.val_frac)
    train_X = X.iloc[:-val_n].copy()
    train_lp = df.iloc[:-val_n][label_cols].reset_index(drop=True)
    val_X = X.iloc[-val_n:].copy()
    val_lp = df.iloc[-val_n:][label_cols].reset_index(drop=True)
    val_meta = df.iloc[-val_n:][["stock_id", "trading_date"] + label_cols].reset_index(drop=True)
    print(f"  train: {len(train_X):,}, val: {len(val_X):,} ({val_meta['trading_date'].min()} ~ {val_meta['trading_date'].max()})")
    print()

    # ── 4. 訓 ensemble ──
    print(f"─── 訓練 {len(args.horizons)} 個 horizon models ───")
    ens = train_multi_horizon_ensemble(train_X, train_lp, val_X, val_lp,
                                        horizons=args.horizons)
    print(f"  訓練完成 {ens.train_secs:.1f}s")
    print(f"  active horizons: {ens.horizons}")
    print()

    # ── 5. Horizon model 相關性 ──
    print("─── Horizon model prediction correlation ───")
    corr = horizon_model_correlation(ens, val_X)
    print(corr.round(4).to_string())
    print()

    # ── 6. 各 horizon model + ensemble IC（against future_ret_20）──
    print("─── IC on val set (target = future_ret_20) ───")
    Xf = val_X[ens.feature_names]
    val_meta = val_meta.copy()
    pcts = []
    for h, model in ens.horizon_models.items():
        raw = model.predict(Xf)
        val_meta[f"score_h{h}"] = raw
        ic_stats = _cross_sectional_ic(val_meta, f"score_h{h}", "future_ret_20")
        ic = ic_stats.get("ic_mean")
        icir = ic_stats.get("icir")
        nd = ic_stats.get("n_days")
        ic_s = f"{ic:+.4f}" if ic is not None else "N/A"
        icir_s = f"{icir:+.3f}" if icir is not None else "N/A"
        marker = " ← 與 target horizon 一致" if h == 20 else ""
        print(f"  h={h:<3} IC={ic_s} ICIR={icir_s} n_days={nd}{marker}")
        # ensemble component
        pct = _rank_to_percentile(raw, by_group=val_meta["trading_date"].to_numpy())
        pcts.append(pct)

    # Multi-horizon ensemble
    val_meta["score_ensemble"] = np.mean(pcts, axis=0)
    ens_stats = _cross_sectional_ic(val_meta, "score_ensemble", "future_ret_20")
    print(f"  ENSEMBLE IC={ens_stats['ic_mean']:+.4f} ICIR={ens_stats['icir']:+.3f} ⭐")

    print()
    print("=== 判定 ===")
    base_h20 = ens.horizon_models.get(20)
    if base_h20 is None:
        print("  沒有 h=20 base model，無法比對")
        return 1
    base_ic = val_meta.apply(
        lambda r: r,
        axis=1,
    )
    # 直接從前面 print 的 score_h20 拿
    base_ic_stats = _cross_sectional_ic(val_meta, "score_h20", "future_ret_20")
    base_ic_mean = base_ic_stats.get("ic_mean") or 0
    ens_ic_mean = ens_stats.get("ic_mean") or 0
    lift = (abs(ens_ic_mean) - abs(base_ic_mean)) / max(abs(base_ic_mean), 1e-9)
    print(f"  h=20 IC:  {base_ic_mean:+.4f}")
    print(f"  Ensemble: {ens_ic_mean:+.4f}")
    print(f"  Lift: {lift:+.1%}")
    if lift >= 0.05:
        print(f"  ✅ multi-horizon ensemble 顯著優於 h=20 single（>= 5%）")
    elif lift >= 0.01:
        print(f"  △ 微幅優於（1-5%）")
    elif lift > -0.05:
        print(f"  ⚠️ 接近（noise）")
    else:
        print(f"  ❌ 反劣於 single h=20")
    return 0


if __name__ == "__main__":
    sys.exit(main())
