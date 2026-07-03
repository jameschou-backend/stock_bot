#!/usr/bin/env python
"""Stage 8.1b：訊號組合 / 走勢一致性特徵 IC 評估（standalone，不污染 build_features）。

候選 5 個新特徵（用 raw_prices + raw_institutional + raw_margin_short 計算）：

  inst_consensus_5d       三大法人方向一致性 5d MA，值域 [-1, +1]
                          ＝ avg over 5d of (sign(foreign)+sign(trust)+sign(dealer))/3

  foreign_net_vol_20      外資 daily net 近 20 日波動率 / 20 日 amt
                          ＝ 衡量外資籌碼震盪強度

  intraday_strength_5d    日內 (close-open)/(high-low) 近 5 日平均
                          ＝ 開高走高趨勢

  opening_gap_5d          近 5 日跳空 (open - prev_close)/prev_close 加總
                          ＝ 連續跳空強度

  margin_share_zscore_20  個股融資餘額佔全市場比例的 20d z-score
                          ＝ 散戶資金集中度異常

判定門檻：|ICIR| ≥ 0.30 + cov ≥ 50% → 候選加入主特徵池
"""
from __future__ import annotations

import argparse
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
from app.models import Label, RawInstitutional, RawMarginShort, RawPrice


def load_prices(session, start, end):
    stmt = (
        select(
            RawPrice.stock_id, RawPrice.trading_date,
            RawPrice.open, RawPrice.high, RawPrice.low, RawPrice.close, RawPrice.volume,
        )
        .where(RawPrice.trading_date.between(start, end))
        .order_by(RawPrice.trading_date, RawPrice.stock_id)
    )
    df = pd.read_sql(stmt, session.get_bind())
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_institutional(session, start, end):
    stmt = (
        select(
            RawInstitutional.stock_id, RawInstitutional.trading_date,
            RawInstitutional.foreign_net, RawInstitutional.trust_net, RawInstitutional.dealer_net,
        )
        .where(RawInstitutional.trading_date.between(start, end))
    )
    df = pd.read_sql(stmt, session.get_bind())
    for c in ("foreign_net", "trust_net", "dealer_net"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_margin(session, start, end):
    stmt = (
        select(
            RawMarginShort.stock_id, RawMarginShort.trading_date,
            RawMarginShort.margin_purchase_balance,
        )
        .where(RawMarginShort.trading_date.between(start, end))
    )
    df = pd.read_sql(stmt, session.get_bind())
    df["margin_purchase_balance"] = pd.to_numeric(df["margin_purchase_balance"], errors="coerce")
    return df


def load_labels(session, start, end):
    stmt = (
        select(Label.stock_id, Label.trading_date, Label.future_ret_h)
        .where(Label.trading_date.between(start, end))
    )
    df = pd.read_sql(stmt, session.get_bind())
    df["future_ret_h"] = pd.to_numeric(df["future_ret_h"], errors="coerce")
    return df


def compute_features(price_df, inst_df, margin_df):
    """逐股 groupby 計算 5 個組合特徵。回傳 long-format DataFrame。"""
    # ── 1. intraday_strength_5d、opening_gap_5d（純價格）──
    price_df = price_df.sort_values(["stock_id", "trading_date"]).reset_index(drop=True)
    rng = (price_df["high"] - price_df["low"]).replace(0, np.nan)
    price_df["intraday_str"] = (price_df["close"] - price_df["open"]) / rng

    def _per_stock_price(g):
        g = g.copy()
        g["intraday_strength_5d"] = g["intraday_str"].rolling(5, min_periods=3).mean()
        g["prev_close"] = g["close"].shift(1)
        g["gap"] = (g["open"] - g["prev_close"]) / g["prev_close"].replace(0, np.nan)
        g["opening_gap_5d"] = g["gap"].rolling(5, min_periods=3).sum()
        return g[["stock_id", "trading_date", "intraday_strength_5d", "opening_gap_5d"]]

    feat_price = price_df.groupby("stock_id", group_keys=False).apply(_per_stock_price)

    # ── 2. inst_consensus_5d ──
    inst_df = inst_df.sort_values(["stock_id", "trading_date"]).reset_index(drop=True)
    inst_df["f_sgn"] = np.sign(inst_df["foreign_net"])
    inst_df["t_sgn"] = np.sign(inst_df["trust_net"])
    inst_df["d_sgn"] = np.sign(inst_df["dealer_net"])
    inst_df["consensus"] = (inst_df["f_sgn"] + inst_df["t_sgn"] + inst_df["d_sgn"]) / 3.0

    def _per_stock_inst(g):
        g = g.copy()
        g["inst_consensus_5d"] = g["consensus"].rolling(5, min_periods=3).mean()
        # 外資 net 近 20 日波動率 / 20 日均量
        g["foreign_net_vol_20"] = (
            g["foreign_net"].rolling(20, min_periods=10).std()
        )
        return g[["stock_id", "trading_date", "inst_consensus_5d", "foreign_net_vol_20"]]

    feat_inst = inst_df.groupby("stock_id", group_keys=False).apply(_per_stock_inst)

    # foreign_net_vol_20 需要除以 20 日均量歸一化 — 用 price.volume
    vol20 = (
        price_df.sort_values(["stock_id", "trading_date"])
        .groupby("stock_id", group_keys=False)
        .apply(lambda g: g.assign(vol20=g["volume"].rolling(20, min_periods=10).mean()))
        [["stock_id", "trading_date", "vol20"]]
    )
    feat_inst = feat_inst.merge(vol20, on=["stock_id", "trading_date"], how="left")
    feat_inst["foreign_net_vol_20"] = (
        feat_inst["foreign_net_vol_20"] / feat_inst["vol20"].replace(0, np.nan)
    )
    feat_inst = feat_inst.drop(columns=["vol20"])

    # ── 3. margin_share_zscore_20 ──
    # 個股融資餘額 / 全市場當日融資總額，再對個股做 20d z-score
    margin_df = margin_df.sort_values(["trading_date", "stock_id"]).reset_index(drop=True)
    daily_total = margin_df.groupby("trading_date")["margin_purchase_balance"].sum().rename("total")
    margin_df = margin_df.merge(daily_total, on="trading_date", how="left")
    margin_df["margin_share"] = margin_df["margin_purchase_balance"] / margin_df["total"].replace(0, np.nan)

    def _per_stock_margin(g):
        g = g.sort_values("trading_date").copy()
        m = g["margin_share"].rolling(20, min_periods=10).mean()
        s = g["margin_share"].rolling(20, min_periods=10).std()
        g["margin_share_zscore_20"] = (g["margin_share"] - m) / s.replace(0, np.nan)
        return g[["stock_id", "trading_date", "margin_share_zscore_20"]]

    feat_margin = margin_df.groupby("stock_id", group_keys=False).apply(_per_stock_margin)

    # ── merge 全部 ──
    out = feat_price.merge(feat_inst, on=["stock_id", "trading_date"], how="outer")
    out = out.merge(feat_margin, on=["stock_id", "trading_date"], how="outer")
    return out


def cross_section_ic(feat_df, label_df, feature):
    sub = feat_df[["stock_id", "trading_date", feature]].copy()
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
    n_total = len(sub)
    n_valid = sub[feature].notna().sum()
    cov = n_valid / n_total if n_total > 0 else 0.0

    merged = sub.merge(label_df, on=["stock_id", "trading_date"], how="inner").dropna()
    ic_by_date = {}
    for td, grp in merged.groupby("trading_date"):
        if len(grp) < 10:
            continue
        rho, _ = stats.spearmanr(grp[feature], grp["future_ret_h"])
        if not np.isnan(rho):
            ic_by_date[td] = rho
    if not ic_by_date:
        return {"feature": feature, "coverage": cov, "n_dates": 0, "ic_mean": np.nan,
                "ic_std": np.nan, "icir": np.nan, "pos_rate": np.nan}
    s = pd.Series(ic_by_date)
    ic_m = float(s.mean()); ic_s = float(s.std())
    return {
        "feature": feature, "coverage": float(cov), "n_dates": int(len(s)),
        "ic_mean": ic_m, "ic_std": ic_s,
        "icir": ic_m / ic_s if ic_s > 0 else 0.0,
        "pos_rate": float((s > 0).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=36)
    args = p.parse_args()

    end = date.today()
    start = end - timedelta(days=args.months * 30 + 60)  # 多 60 日預熱

    print(f"\n{'='*78}")
    print(f"Stage 8.1b：訊號組合特徵 IC 評估")
    print(f"{'='*78}")
    print(f"期間：{start} ~ {end} ({args.months}mo + 60d warmup)")

    with get_session() as session:
        print("\n[1/4] 載入 prices...")
        price_df = load_prices(session, start, end)
        print(f"  ✓ {len(price_df):,} rows, {price_df['stock_id'].nunique()} stocks")

        print("[2/4] 載入 institutional...")
        inst_df = load_institutional(session, start, end)
        print(f"  ✓ {len(inst_df):,} rows")

        print("[3/4] 載入 margin_short...")
        margin_df = load_margin(session, start, end)
        print(f"  ✓ {len(margin_df):,} rows")

        print("[4/4] 載入 labels...")
        label_df = load_labels(session, start, end)
        print(f"  ✓ {len(label_df):,} rows")

    print("\n[5/5] 計算特徵...")
    feat_df = compute_features(price_df, inst_df, margin_df)
    # 排除預熱期
    eval_start = end - timedelta(days=args.months * 30)
    feat_df = feat_df[feat_df["trading_date"] >= eval_start]
    label_df = label_df[label_df["trading_date"] >= eval_start]

    features = [
        "inst_consensus_5d", "foreign_net_vol_20",
        "intraday_strength_5d", "opening_gap_5d",
        "margin_share_zscore_20",
    ]

    print(f"\n{'特徵':<28} {'cov':>7} {'n_dt':>6} {'IC_mean':>10} {'IC_std':>10} {'ICIR':>8} {'pos%':>7} {'判定'}")
    print("-" * 98)
    recoverable = []
    for f in features:
        r = cross_section_ic(feat_df, label_df, f)
        ok = abs(r["icir"]) >= 0.30 and r["coverage"] >= 0.50 and r["n_dates"] >= 200
        verdict = "✅ 候選" if ok else (
            "⚠️ ICIR 弱" if not abs(r["icir"]) >= 0.30 else
            "⚠️ cov 不足" if r["coverage"] < 0.50 else "⚠️ n_dt 少"
        )
        if ok:
            recoverable.append(r)
        print(f"{f:<28} {r['coverage']:>7.1%} {r['n_dates']:>6d} "
              f"{r['ic_mean']:>10.4f} {r['ic_std']:>10.4f} {r['icir']:>+8.4f} "
              f"{r['pos_rate']:>7.1%}  {verdict}")

    print("\n" + "=" * 78)
    if recoverable:
        print(f"候選加入主特徵池（{len(recoverable)} 個）：")
        for r in recoverable:
            print(f"  • {r['feature']:<28}  ICIR={r['icir']:+.4f}  cov={r['coverage']:.1%}")
    else:
        print("⚠️ 5 個候選都未達標。考慮：(a) 設計更非線性的訊號 (b) 結合多訊號互動")
    print("=" * 78)


if __name__ == "__main__":
    main()
