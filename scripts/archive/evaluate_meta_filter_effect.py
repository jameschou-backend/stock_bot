"""Stage 4.2 效益評估：模擬「Strategy D pick top-N」前後 Meta filter 對比。

在 validation set 上模擬：
  Mode A (baseline): 每日依 primary_score 取 top-N（同 Strategy D pos=4）
  Mode B (meta filter): 先過 p_meta >= threshold，再依 primary_score 取 top-N

報告兩者的 mean tb_return / std / annualized Sharpe-like。

⚠️ Limitations：
  - 這是 quick eval，**不是完整 walk-forward**：
    - validation set 是訓練時 chronological 後 20%，意味著 primary model 訓練看過
      val 之前的資料 → 對 val 的 primary_score 算是 OOS（OK）
    - meta model 訓練也是 chronological 後 20% 留做 val → 也算 OOS（OK）
    - 但 primary 用整 dataset 訓練後 in-sample predict 給 meta → 訓練時有 leakage
      因此這個 eval 結果**偏樂觀**
  - tb_return 是 5-day forward return（path-dependent at PT/SL/time barriers），
    與 Strategy D 實際 trailing-stop 出場行為不完全相同
  - 因此把這個結果當作「上界估計」，正式測量還是要 wire 進 backtest_rotation.py

用法：
  python scripts/evaluate_meta_filter_effect.py \\
      --model artifacts/models/meta_label_d_5_3_only_positive.joblib \\
      --tb-labels artifacts/labels/tb_d_5_3.parquet \\
      --top-n 4
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from sqlalchemy import select

from app.db import get_session
from app.models import Label
from skills.feature_store import FeatureStore


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="meta_label_*.joblib")
    p.add_argument("--tb-labels", type=Path, required=True, help="tb labels parquet (matches model's TB config)")
    p.add_argument("--top-n", type=int, default=4, help="每日選股數（Strategy D 預設 4）")
    p.add_argument("--val-frac", type=float, default=0.2, help="後 N% 當 val set（同訓練設定）")
    p.add_argument("--periods-per-year", type=int, default=52,
                   help="年化倍數（5-day horizon 用 ~52，10-day 用 ~26，20-day 用 12）")
    p.add_argument("--thresholds", nargs="*", type=float,
                   default=[0.0, 0.50, 0.55, 0.60, 0.65, 0.70],
                   help="thresholds to sweep（0.0 = baseline 不 filter）")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print(f"=== Stage 4.2 效益評估（quick eval on val set）===")
    print(f"  model: {args.model}")
    print(f"  TB labels: {args.tb_labels}")
    print(f"  top_n: {args.top_n}, val_frac: {args.val_frac}, periods_per_year: {args.periods_per_year}")
    print()

    bundle = joblib.load(args.model)
    meta = bundle["meta_model"]
    primary = bundle["primary_model"]
    primary_features = bundle["primary_features"]

    print("  讀 features + labels ...", end="", flush=True)
    fs = FeatureStore()
    feat_df = fs.read(date(2016, 1, 1), fs.get_max_date())
    with get_session() as s:
        label_df = pd.read_sql(
            select(Label.stock_id, Label.trading_date, Label.future_ret_h),
            s.get_bind(),
        )
    feat_df["trading_date"] = pd.to_datetime(feat_df["trading_date"]).dt.date
    label_df["trading_date"] = pd.to_datetime(label_df["trading_date"]).dt.date
    feat_df = feat_df.merge(label_df, on=["stock_id", "trading_date"], how="inner")
    feat_df = feat_df.dropna(subset=["future_ret_h"])
    print(f" {len(feat_df):,} rows")

    print("  primary_score in-sample predict ...", end="", flush=True)
    primary_score = primary.predict(feat_df[primary_features].fillna(0))
    feat_df = feat_df.assign(primary_score=primary_score)
    print(" OK")

    print("  讀 TB labels ...", end="", flush=True)
    tb_df = pd.read_parquet(args.tb_labels)
    tb_df["trading_date"] = pd.to_datetime(tb_df["trading_date"]).dt.date
    print(f" {len(tb_df):,} rows")

    print("  merge + filter primary_score > 0 ...", end="", flush=True)
    joined = feat_df.merge(
        tb_df[["stock_id", "trading_date", "tb_return", "tb_label"]],
        on=["stock_id", "trading_date"], how="inner",
    )
    joined = joined[joined["primary_score"] > 0]
    joined = joined.sort_values("trading_date").reset_index(drop=True)
    print(f" {len(joined):,} rows")

    print("  meta predict_proba ...", end="", flush=True)
    X = joined[primary_features].copy()
    X["__primary_score"] = joined["primary_score"].to_numpy()
    p_meta = meta.estimator.predict_proba(X)[:, 1]
    joined["p_meta"] = p_meta
    print(" OK")

    # Validation set = 後 val_frac %
    n = len(joined)
    val_n = int(n * args.val_frac)
    val = joined.iloc[-val_n:].copy()
    print(f"  val set: {len(val):,} rows ({val['trading_date'].min()} ~ {val['trading_date'].max()})")
    print()

    rows = []
    for thr in args.thresholds:
        sub = val if thr <= 0 else val[val["p_meta"] >= thr]
        if sub.empty:
            rows.append({
                "threshold": thr,
                "n_picks": 0,
                "n_dates": 0,
                "pre_filter_rate": (sub.shape[0] / val.shape[0]) if val.shape[0] else 0,
                "mean_pick_ret_pct": np.nan,
                "std_pick_ret_pct": np.nan,
                "sharpe_naive": np.nan,
                "ann_return_pct": np.nan,
            })
            continue
        # 每日選 top-N
        picks = (
            sub.sort_values(["trading_date", "primary_score"], ascending=[True, False])
            .groupby("trading_date").head(args.top_n)
        )
        daily_mean = picks.groupby("trading_date")["tb_return"].mean()
        n_dates = daily_mean.shape[0]
        mean = float(daily_mean.mean())
        std = float(daily_mean.std(ddof=1)) if n_dates > 1 else 0.0
        sharpe = (mean / std * np.sqrt(args.periods_per_year)) if std > 0 else 0.0
        # Annualized return: (1+mean)^periods - 1
        ann_ret = ((1 + mean) ** args.periods_per_year - 1) if mean > -1 else np.nan
        rows.append({
            "threshold": thr,
            "n_picks": int(len(picks)),
            "n_dates": int(n_dates),
            "pre_filter_rate": float(sub.shape[0] / val.shape[0]),
            "mean_pick_ret_pct": mean * 100,
            "std_pick_ret_pct": std * 100,
            "sharpe_naive": sharpe,
            "ann_return_pct": ann_ret * 100 if not np.isnan(ann_ret) else np.nan,
        })

    df = pd.DataFrame(rows)
    print("=== Sweep 結果（top-N=%d，每日選股的平均 tb_return）===" % args.top_n)
    print()
    cols_fmt = {
        "threshold": "{:.2f}".format,
        "n_picks": "{:>8,}".format,
        "n_dates": "{:>6,}".format,
        "pre_filter_rate": "{:.1%}".format,
        "mean_pick_ret_pct": "{:>+7.3f}%".format,
        "std_pick_ret_pct": "{:>6.3f}%".format,
        "sharpe_naive": "{:>+7.3f}".format,
        "ann_return_pct": "{:>+9.1f}%".format,
    }
    out = df.copy()
    for c, f in cols_fmt.items():
        out[c] = out[c].apply(lambda v: "—" if pd.isna(v) else f(v))
    print(out.to_string(index=False))

    # 對比 baseline (thr=0.0)
    base = df[df["threshold"] == 0.0].iloc[0]
    print()
    print("=== 相對 baseline（threshold=0）改善 ===")
    for _, r in df.iterrows():
        if r["threshold"] == 0.0:
            continue
        sharpe_delta = r["sharpe_naive"] - base["sharpe_naive"]
        mean_delta = r["mean_pick_ret_pct"] - base["mean_pick_ret_pct"]
        trade_drop = base["n_picks"] - r["n_picks"]
        print(f"  thr={r['threshold']:.2f}: "
              f"Sharpe Δ {sharpe_delta:+.3f}, "
              f"mean ret Δ {mean_delta:+.3f}%, "
              f"picks 從 {base['n_picks']:,} 降到 {r['n_picks']:,} (-{trade_drop:,})")

    print()
    print("⚠️ 提醒：這是 in-sample（primary 訓練時看過 val）的快速估算，")
    print("   實際 Strategy D backtest 整合測量會略低。正式驗證請 wire 進 backtest_rotation.py。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
