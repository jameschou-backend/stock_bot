"""
IC 衰減分析腳本
比較各特徵在不同時間段的預測力（Spearman IC / ICIR），找出近 2 年已衰減的特徵。

用法：
    python scripts/ic_decay_analysis.py
    python scripts/ic_decay_analysis.py --recent-years 2 --output artifacts/ic_decay.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db import get_session
from app.models import Label
from sqlalchemy import select


# ── 載入資料 ──────────────────────────────────────────────────────────────────

def _load_labels_df() -> pd.DataFrame:
    print("載入 labels ...", flush=True)
    with get_session() as s:
        rows = s.execute(
            select(Label.stock_id, Label.trading_date, Label.future_ret_h)
        ).fetchall()
    df = pd.DataFrame(rows, columns=["stock_id", "trading_date", "future_ret_h"])
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    df["future_ret_h"] = pd.to_numeric(df["future_ret_h"], errors="coerce")
    return df.dropna(subset=["future_ret_h"])


def _load_features_df() -> pd.DataFrame:
    """從 Parquet 讀取（快），不存在時從 DB JSON 讀取（慢）。"""
    parquet_files = sorted((ROOT / "artifacts" / "features").glob("features_*.parquet"))
    if parquet_files:
        print(f"從 Parquet 載入特徵（{len(parquet_files)} 個年份）...", flush=True)
        parts = []
        for fp in parquet_files:
            try:
                parts.append(pd.read_parquet(fp))
            except Exception as e:
                print(f"  跳過 {fp.name}：{e}")
        df = pd.concat(parts, ignore_index=True)
    else:
        print("從 DB 載入特徵（較慢）...", flush=True)
        from app.models import Feature
        from skills.feature_utils import parse_features_json
        with get_session() as s:
            rows = s.execute(
                select(Feature.stock_id, Feature.trading_date, Feature.features_json)
            ).fetchall()
        records = []
        for sid, td, fj in rows:
            feat = parse_features_json(pd.Series([fj]))[0] if fj else {}
            feat["stock_id"] = str(sid)
            feat["trading_date"] = pd.to_datetime(td)
            records.append(feat)
        df = pd.DataFrame(records)

    df["trading_date"] = pd.to_datetime(df["trading_date"])
    return df


# ── IC 計算 ──────────────────────────────────────────────────────────────────

def _compute_monthly_ic(feat_df: pd.DataFrame, label_df: pd.DataFrame,
                         feat_cols: list[str]) -> pd.DataFrame:
    """每月計算各特徵的 Spearman IC。"""
    merged = feat_df[["stock_id", "trading_date"] + feat_cols].merge(
        label_df[["stock_id", "trading_date", "future_ret_h"]],
        on=["stock_id", "trading_date"], how="inner",
    )
    merged["ym"] = merged["trading_date"].dt.to_period("M")

    records = []
    for ym, grp in merged.groupby("ym"):
        y = grp["future_ret_h"].values
        row = {"ym": str(ym), "n": len(grp)}
        for col in feat_cols:
            x = grp[col].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 30:
                row[col] = np.nan
            else:
                ic, _ = spearmanr(x[mask], y[mask])
                row[col] = float(ic) if np.isfinite(ic) else np.nan
        records.append(row)
        if len(records) % 12 == 0:
            print(f"  已計算 {len(records)} 個月...", flush=True)

    return pd.DataFrame(records).set_index("ym")


def _summarize(monthly_ic: pd.DataFrame, feat_cols: list[str],
               recent_cutoff: str) -> pd.DataFrame:
    """彙整全期 vs 近期 IC / ICIR。"""
    ic_data = monthly_ic[feat_cols]
    recent = ic_data[ic_data.index >= recent_cutoff]
    history = ic_data[ic_data.index < recent_cutoff]

    rows = []
    for col in feat_cols:
        full_ic   = ic_data[col].dropna()
        hist_ic   = history[col].dropna()
        rec_ic    = recent[col].dropna()

        def stats(s):
            if len(s) == 0:
                return np.nan, np.nan
            mu = s.mean()
            se = s.std() / (len(s) ** 0.5 + 1e-9)
            return float(mu), float(mu / (s.std() + 1e-9))

        full_mean, full_icir = stats(full_ic)
        hist_mean, hist_icir = stats(hist_ic)
        rec_mean,  rec_icir  = stats(rec_ic)

        decay_pct = ((rec_mean - hist_mean) / (abs(hist_mean) + 1e-6)) * 100

        if abs(rec_mean) < 0.005 and abs(hist_mean) >= 0.01:
            status = "❌ 失效"
        elif abs(rec_mean) < abs(hist_mean) * 0.5:
            status = "⚠️  衰減"
        elif abs(rec_mean) > abs(hist_mean) * 1.2:
            status = "🔺 增強"
        else:
            status = "✅ 穩定"

        rows.append({
            "特徵": col,
            "全期IC": round(full_mean, 4),
            "全期ICIR": round(full_icir, 3),
            "歷史IC": round(hist_mean, 4),
            "近期IC": round(rec_mean, 4),
            "近期ICIR": round(rec_icir, 3),
            "衰減%": round(decay_pct, 1),
            "狀態": status,
        })

    df = pd.DataFrame(rows).sort_values("衰減%")
    return df


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IC 衰減分析")
    parser.add_argument("--recent-years", type=int, default=2,
                        help="「近期」定義：最近 N 年（預設 2）")
    parser.add_argument("--output", default="artifacts/ic_decay.csv",
                        help="輸出 CSV 路徑")
    args = parser.parse_args()

    cutoff_date = date.today() - timedelta(days=args.recent_years * 365)
    recent_cutoff = cutoff_date.strftime("%Y-%m")
    print(f"分析配置：近期 = {recent_cutoff} 以後（最近 {args.recent_years} 年）\n")

    label_df   = _load_labels_df()
    feature_df = _load_features_df()

    # 找出可用特徵欄位
    from skills.build_features import FEATURE_COLUMNS
    feat_cols = [c for c in FEATURE_COLUMNS if c in feature_df.columns]
    print(f"分析 {len(feat_cols)} 個特徵，{len(label_df)} 筆 label\n")

    print("計算月度 IC（這需要幾分鐘）...", flush=True)
    monthly_ic = _compute_monthly_ic(feature_df, label_df, feat_cols)

    summary = _summarize(monthly_ic, feat_cols, recent_cutoff)

    # ── 輸出報告 ──
    print("\n" + "=" * 80)
    print(f"IC 衰減分析報告（近期基準：{recent_cutoff}）")
    print("=" * 80)

    # 失效特徵
    failed = summary[summary["狀態"].str.startswith("❌")]
    decay  = summary[summary["狀態"].str.startswith("⚠️")]
    boost  = summary[summary["狀態"].str.startswith("🔺")]
    stable = summary[summary["狀態"].str.startswith("✅")]

    print(f"\n❌ 失效（{len(failed)} 個）：預測力幾乎歸零")
    print(failed.to_string(index=False))

    print(f"\n⚠️  衰減（{len(decay)} 個）：近期 IC < 歷史 50%")
    print(decay.to_string(index=False))

    print(f"\n🔺 增強（{len(boost)} 個）：近期 IC > 歷史 120%")
    print(boost.to_string(index=False))

    print(f"\n✅ 穩定（{len(stable)} 個）")
    print(stable.to_string(index=False))

    # 儲存 CSV
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n已儲存：{out_path}")

    # 也儲存月度 IC
    ic_out = out_path.with_name("ic_monthly.csv")
    monthly_ic.to_csv(ic_out, encoding="utf-8-sig")
    print(f"月度 IC：{ic_out}")

    return summary


if __name__ == "__main__":
    main()
