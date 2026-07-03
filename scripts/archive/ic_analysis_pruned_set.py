#!/usr/bin/env python
"""Stage 8.1: 重新評估 _PRUNE_SET 內所有特徵的 IC/ICIR 與資料覆蓋率。

當初 SHAP 剪枝是在資料量較少時做的（特別是 sponsor 籌碼資料每日累積），
本腳本在最近 36/60 個月窗口重新跑 cross-sectional Spearman IC，找出值得
回收到 PRUNED_FEATURE_COLS 的特徵。

判定門檻：
  - 樣本覆蓋率 ≥ 50%（有效樣本數 / 總樣本數）
  - |ICIR| ≥ 0.30（IC mean / IC std）
  - 樣本日數 ≥ 200 個交易日

用法：
    python scripts/ic_analysis_pruned_set.py                 # 預設 36mo
    python scripts/ic_analysis_pruned_set.py --months 60     # 60mo
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import select

from app.config import load_config  # noqa: F401
from app.db import get_session
from app.models import Feature, Label
from skills.build_features import _PRUNE_SET, _SPONSOR_FEATURES


def fetch_features(session, start_date, end_date, only_cols=None):
    stmt = (
        select(Feature.stock_id, Feature.trading_date, Feature.features_json)
        .where(Feature.trading_date.between(start_date, end_date))
        .order_by(Feature.trading_date, Feature.stock_id)
    )
    rows = session.execute(stmt).fetchall()
    if not rows:
        return pd.DataFrame()
    records = []
    for stock_id, trading_date, fjson in rows:
        d = fjson if isinstance(fjson, dict) else json.loads(fjson)
        if only_cols:
            d = {k: d.get(k) for k in only_cols}
        d["stock_id"] = stock_id
        d["trading_date"] = trading_date
        records.append(d)
    return pd.DataFrame(records)


def fetch_labels(session, start_date, end_date):
    stmt = (
        select(Label.stock_id, Label.trading_date, Label.future_ret_h)
        .where(Label.trading_date.between(start_date, end_date))
        .order_by(Label.trading_date, Label.stock_id)
    )
    rows = session.execute(stmt).fetchall()
    df = pd.DataFrame(rows, columns=["stock_id", "trading_date", "future_ret_h"])
    df["future_ret_h"] = pd.to_numeric(df["future_ret_h"], errors="coerce")
    return df


def compute_ic_stats(feat_df, label_df, feature):
    if feature not in feat_df.columns:
        return None
    sub = feat_df[["stock_id", "trading_date", feature]].copy()
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
    n_total = len(sub)
    n_valid = sub[feature].notna().sum()
    coverage = n_valid / n_total if n_total > 0 else 0.0

    merged = sub.merge(label_df, on=["stock_id", "trading_date"], how="inner")
    merged = merged.dropna(subset=[feature, "future_ret_h"])
    if merged.empty:
        return {"feature": feature, "coverage": coverage, "n_dates": 0,
                "ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan, "pos_rate": np.nan}

    ic_by_date = {}
    for td, grp in merged.groupby("trading_date"):
        if len(grp) < 10:
            continue
        rho, _ = stats.spearmanr(grp[feature], grp["future_ret_h"])
        if not np.isnan(rho):
            ic_by_date[td] = rho
    if not ic_by_date:
        return {"feature": feature, "coverage": coverage, "n_dates": 0,
                "ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan, "pos_rate": np.nan}

    s = pd.Series(ic_by_date)
    ic_mean = float(s.mean())
    ic_std = float(s.std())
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    return {
        "feature": feature,
        "coverage": float(coverage),
        "n_dates": int(len(s)),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": float(icir),
        "pos_rate": float((s > 0).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=36)
    p.add_argument("--ic-threshold", type=float, default=0.30, help="|ICIR| 門檻")
    p.add_argument("--coverage-threshold", type=float, default=0.50)
    p.add_argument("--min-dates", type=int, default=200)
    args = p.parse_args()

    end = date.today()
    start = end - timedelta(days=args.months * 30)

    print(f"\n{'='*78}")
    print(f"Stage 8.1：_PRUNE_SET 特徵 IC/ICIR 再評估")
    print(f"{'='*78}")
    print(f"期間：{start} ~ {end}（{args.months} 個月）")
    print(f"門檻：|ICIR|≥{args.ic_threshold}, coverage≥{args.coverage_threshold:.0%}, n_dates≥{args.min_dates}")

    only_cols = list(_PRUNE_SET)
    print(f"分析特徵數：{len(only_cols)}")

    with get_session() as session:
        print("\n[1/3] 載入特徵...")
        feat_df = fetch_features(session, start, end, only_cols=only_cols)
        if feat_df.empty:
            print("  ❌ 無資料")
            return
        print(f"  ✓ {feat_df['trading_date'].nunique()} 個交易日, {feat_df['stock_id'].nunique()} 股")

        print("\n[2/3] 載入標籤...")
        label_df = fetch_labels(session, start, end)
        print(f"  ✓ {label_df['trading_date'].nunique()} 個標籤日")

    print("\n[3/3] 計算 IC/ICIR...")
    print(f"\n{'特徵':<32} {'cov':>6} {'n_dt':>6} {'IC_mean':>10} {'IC_std':>10} {'ICIR':>8} {'pos%':>7} {'類型':<8} {'判定':<12}")
    print("-" * 110)

    results = []
    for feat in sorted(only_cols):
        r = compute_ic_stats(feat_df, label_df, feat)
        if r is None:
            print(f"{feat:<32} {'N/A':>6} {'N/A':>6} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>7} {'-':<8} (DB 缺欄位)")
            continue
        results.append(r)

    # 排序：依 |ICIR| 由大到小
    results.sort(key=lambda x: -abs(x["icir"]) if not np.isnan(x["icir"]) else 0)

    recoverable = []
    for r in results:
        is_sponsor = r["feature"] in _SPONSOR_FEATURES
        kind = "sponsor" if is_sponsor else "shap-prune"

        ok_ic = abs(r["icir"]) >= args.ic_threshold if not np.isnan(r["icir"]) else False
        ok_cov = r["coverage"] >= args.coverage_threshold
        ok_dt = r["n_dates"] >= args.min_dates
        verdict = "✅ 可回收" if (ok_ic and ok_cov and ok_dt) else (
            "⚠️ cov 不足" if not ok_cov else
            "⚠️ ICIR 弱" if not ok_ic else
            "⚠️ n_dt 少"
        )
        if verdict == "✅ 可回收":
            recoverable.append(r)

        print(f"{r['feature']:<32} {r['coverage']:>6.1%} {r['n_dates']:>6d} "
              f"{r['ic_mean']:>10.4f} {r['ic_std']:>10.4f} {r['icir']:>8.4f} "
              f"{r['pos_rate']:>7.1%} {kind:<8} {verdict}")

    print("\n" + "=" * 78)
    print(f"建議回收（{len(recoverable)} 個特徵）：")
    if recoverable:
        for r in recoverable:
            print(f"  • {r['feature']:<28}  ICIR={r['icir']:+.4f}  cov={r['coverage']:.1%}")
        print("\n下一步：移出 _PRUNE_SET → 10y backtest 驗證")
    else:
        print("  無特徵滿足回收門檻。考慮：(a) 放寬門檻; (b) 等更多資料累積; (c) Stage 8 改走 LLM/NLP")
    print("=" * 78)


if __name__ == "__main__":
    main()
