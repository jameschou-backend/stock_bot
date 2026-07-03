"""Stage 5 IC 分析：PER value factors + FracDiff features vs 既有 ma_5/20/60。

每個候選 feature：
  - 每個 trading_date 跨股算 cross-sectional Spearman rank corr(feature, future_ret_h)
  - 統計 IC mean / std / ICIR / positive_rate over time
  - ICIR = mean / std × sqrt(n_days)（n_days 是有效日數）

判定門檻（López de Prado）：
  - |IC mean| >= 0.02 + |ICIR| >= 0.5  → 有效因子
  - |IC mean| < 0.01                    → 雜訊
  - 中間 → 邊緣

用法：
    python scripts/ic_analysis_stage5.py                    # 全特徵分析
    python scripts/ic_analysis_stage5.py --start 2024-01-01 # 限縮期間
    python scripts/ic_analysis_stage5.py --output artifacts/ic_stage5.csv
"""
from __future__ import annotations

import argparse
import math
import sys
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
from app.models import Label, RawPER
from skills.feature_store import FeatureStore


# 要分析的特徵組（grouped by source for output）
PER_FEATURES = ["per", "pbr", "dividend_yield"]
FRACDIFF_FEATURES = ["close_fracdiff_0_30", "close_fracdiff_0_40", "close_fracdiff_0_50"]
EXISTING_FEATURES = ["ma_5", "ma_20", "ma_60", "ret_5", "ret_20", "vol_ratio_5"]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2018-01-01", help="分析起始日（YYYY-MM-DD）")
    p.add_argument("--end", type=str, default=None, help="分析結束日（YYYY-MM-DD；預設今天）")
    p.add_argument("--min-stocks-per-day", type=int, default=30,
                   help="某日股票數低於此值則跳過該日（避免 noise）")
    p.add_argument("--output", type=Path, default=Path("artifacts/ic_stage5.csv"))
    p.add_argument("--fracdiff-parquet", type=Path,
                   default=Path("artifacts/labels/fracdiff_features.parquet"))
    return p.parse_args()


def _ic_stats(ic_series: pd.Series) -> dict:
    """IC mean / std / ICIR / positive_rate。"""
    s = ic_series.dropna()
    n = len(s)
    if n < 5:
        return {"n_days": n, "ic_mean": None, "ic_std": None, "icir": None,
                "positive_rate": None, "t_stat": None}
    mean = float(s.mean())
    std = float(s.std(ddof=1))
    icir = (mean / std * math.sqrt(n)) if std > 1e-9 else 0.0
    t_stat = (mean / (std / math.sqrt(n))) if std > 1e-9 else 0.0
    return {
        "n_days": n,
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "positive_rate": float((s > 0).mean()),
        "t_stat": t_stat,
    }


def _compute_cross_sectional_ic(
    df: pd.DataFrame,
    feature_col: str,
    label_col: str,
    min_stocks: int,
) -> pd.Series:
    """每個 trading_date 跨股 Spearman rank IC。"""
    sub = df.dropna(subset=[feature_col, label_col])
    out = []
    for d, g in sub.groupby("trading_date", sort=True):
        if len(g) < min_stocks:
            continue
        try:
            ic, _ = stats.spearmanr(g[feature_col].to_numpy(), g[label_col].to_numpy())
            if not np.isnan(ic):
                out.append((d, float(ic)))
        except Exception:
            continue
    if not out:
        return pd.Series(dtype=float)
    s = pd.DataFrame(out, columns=["trading_date", "ic"]).set_index("trading_date")["ic"]
    return s


def _verdict(stats_d: dict) -> str:
    """根據 IC mean / ICIR 給判定。"""
    ic = stats_d.get("ic_mean")
    icir = stats_d.get("icir")
    if ic is None:
        return "no data"
    abs_ic = abs(ic)
    abs_icir = abs(icir) if icir is not None else 0
    if abs_ic >= 0.02 and abs_icir >= 0.5:
        return "✅ 有效"
    elif abs_ic >= 0.01 and abs_icir >= 0.3:
        return "△ 邊緣"
    elif abs_ic < 0.005:
        return "❌ 雜訊"
    else:
        return "△ 弱"


def main() -> int:
    args = _parse_args()
    start_date = pd.to_datetime(args.start).date()
    end_date = pd.to_datetime(args.end).date() if args.end else date.today()

    print(f"=== Stage 5 IC 分析 ===")
    print(f"  期間: {start_date} ~ {end_date}")
    print(f"  min stocks/day: {args.min_stocks_per_day}")
    print()

    # ── 1. Features parquet（含 ma_*、per_*、其他 existing）────────────
    print("  載入 features parquet ...", end="", flush=True)
    fs = FeatureStore()
    feat_df = fs.read(start_date, end_date)
    print(f" {len(feat_df):,} rows, {len([c for c in feat_df.columns if c not in ('stock_id','trading_date')])} cols")

    # ── 2. Labels ──────────────────────────────────────────────────
    print("  載入 labels ...", end="", flush=True)
    with get_session() as s:
        q = (select(Label.stock_id, Label.trading_date, Label.future_ret_h)
             .where(Label.trading_date >= start_date)
             .where(Label.trading_date <= end_date))
        label_df = pd.read_sql(q, s.get_bind())
    label_df["trading_date"] = pd.to_datetime(label_df["trading_date"]).dt.date
    label_df["future_ret_h"] = pd.to_numeric(label_df["future_ret_h"], errors="coerce")
    print(f" {len(label_df):,} rows")

    # ── 3. PER table（單獨表）─────────────────────────────────────
    print("  載入 raw_per ...", end="", flush=True)
    with get_session() as s:
        q = (select(RawPER.stock_id, RawPER.trading_date, RawPER.per, RawPER.pbr, RawPER.dividend_yield)
             .where(RawPER.trading_date >= start_date)
             .where(RawPER.trading_date <= end_date))
        per_df = pd.read_sql(q, s.get_bind())
    per_df["trading_date"] = pd.to_datetime(per_df["trading_date"]).dt.date
    for c in ["per", "pbr", "dividend_yield"]:
        per_df[c] = pd.to_numeric(per_df[c], errors="coerce")
    print(f" {len(per_df):,} rows")

    # ── 4. FracDiff parquet ───────────────────────────────────────
    if args.fracdiff_parquet.exists():
        print(f"  載入 fracdiff {args.fracdiff_parquet} ...", end="", flush=True)
        fd_df = pd.read_parquet(args.fracdiff_parquet)
        fd_df["trading_date"] = pd.to_datetime(fd_df["trading_date"]).dt.date
        fd_df = fd_df[(fd_df["trading_date"] >= start_date) & (fd_df["trading_date"] <= end_date)]
        print(f" {len(fd_df):,} rows")
    else:
        print(f"  ⚠️ fracdiff parquet 不存在: {args.fracdiff_parquet}")
        fd_df = None

    # ── 合併到主表 ─────────────────────────────────────────────
    feat_df["trading_date"] = pd.to_datetime(feat_df["trading_date"]).dt.date
    main = feat_df.merge(label_df, on=["stock_id", "trading_date"], how="inner")
    main = main.merge(per_df, on=["stock_id", "trading_date"], how="left", suffixes=("", "_per"))
    if fd_df is not None:
        main = main.merge(fd_df, on=["stock_id", "trading_date"], how="left")
    print(f"  合併後 main: {len(main):,} rows, {main['trading_date'].nunique()} dates")

    # ── 對每個 feature 跑 IC ──────────────────────────────────
    all_feats = []
    # PER：來自 raw_per merge，欄名是 per/pbr/dividend_yield（可能與 features 重名）
    all_feats.extend([(f, "PER value factors") for f in PER_FEATURES if f in main.columns])
    # FracDiff
    if fd_df is not None:
        all_feats.extend([(f, "FracDiff (Stage 4.3)") for f in FRACDIFF_FEATURES if f in main.columns])
    # 既有特徵對照
    all_feats.extend([(f, "Existing baseline") for f in EXISTING_FEATURES if f in main.columns])

    if not all_feats:
        print("❌ 沒有任何待分析特徵在 main 中")
        return 1

    rows = []
    print()
    print(f"=== IC results ===")
    print(f"  {'group':<25} {'feature':<25} {'n_days':<8} {'IC mean':<10} {'ICIR':<10} {'pos_rate':<10} {'verdict'}")
    print("  " + "-" * 95)
    for col, group in all_feats:
        ic_series = _compute_cross_sectional_ic(
            main, col, "future_ret_h", min_stocks=args.min_stocks_per_day,
        )
        s = _ic_stats(ic_series)
        verdict = _verdict(s)
        row = {"group": group, "feature": col, **s, "verdict": verdict}
        rows.append(row)
        ic_s = f"{s['ic_mean']:+.4f}" if s["ic_mean"] is not None else "N/A"
        icir_s = f"{s['icir']:+.3f}" if s["icir"] is not None else "N/A"
        pr_s = f"{s['positive_rate']:.1%}" if s["positive_rate"] is not None else "N/A"
        print(f"  {group:<25} {col:<25} {s['n_days']:<8} {ic_s:<10} {icir_s:<10} {pr_s:<10} {verdict}")

    # 寫 CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print()
    print(f"✅ 結果存檔: {args.output}")

    print()
    print("=== 解讀建議 ===")
    valid = [r for r in rows if r["verdict"].startswith("✅")]
    if valid:
        print(f"  ✅ {len(valid)} 個有效因子 → 考慮從 _PRUNE_SET 移出加入 production model:")
        for r in valid:
            print(f"     - {r['feature']:<25} (IC={r['ic_mean']:+.4f}, ICIR={r['icir']:+.2f})")
    else:
        print("  ❌ 沒有達到 IC>=0.02 + ICIR>=0.5 標準的特徵")
        print("     可嘗試：(1) 拉長分析期間 --start 2016-01-01")
        print("            (2) 換 horizon（目前 future_ret_h 是 20 日 forward return）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
