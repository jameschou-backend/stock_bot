from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import func, select

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from app.models import Feature, Label


def _parse_features(series: pd.Series) -> pd.DataFrame:
    parsed = []
    for value in series:
        if value is None:
            parsed.append({})
        elif isinstance(value, dict):
            parsed.append(value)
        elif isinstance(value, str):
            parsed.append(json.loads(value))
        else:
            parsed.append({})
    return pd.json_normalize(parsed)


def _calc_monthly_ic(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    df = df.copy()
    df["month"] = pd.to_datetime(df["trading_date"]).dt.to_period("M").astype(str)
    for feat in feature_cols:
        sub = df[["month", feat, "future_ret_h"]].dropna()
        if sub.empty:
            continue
        for month, mdf in sub.groupby("month"):
            if len(mdf) < 30:
                continue
            # 常數序列無法計算 Spearman，直接跳過避免噪音 warning。
            if mdf[feat].nunique(dropna=True) <= 1 or mdf["future_ret_h"].nunique(dropna=True) <= 1:
                continue
            ic, _ = spearmanr(mdf[feat], mdf["future_ret_h"])
            if np.isnan(ic):
                continue
            rows.append({"feature": feat, "month": month, "ic": float(ic)})
    return pd.DataFrame(rows)


def _calc_quantile_returns(df: pd.DataFrame, feature_cols: list[str], q: int = 10) -> pd.DataFrame:
    rows = []
    work_base = df.copy()
    work_base["month"] = pd.to_datetime(work_base["trading_date"]).dt.to_period("M").astype(str)
    for feat in feature_cols:
        work = work_base[["month", feat, "future_ret_h"]].dropna().copy()
        if work.empty:
            continue
        for month, tdf in work.groupby("month"):
            if len(tdf) < q * 3:
                continue
            try:
                tdf["bucket"] = pd.qcut(tdf[feat], q=q, labels=False, duplicates="drop") + 1
            except ValueError:
                continue
            agg = tdf.groupby("bucket", as_index=False)["future_ret_h"].mean()
            for _, row in agg.iterrows():
                rows.append(
                    {
                        "feature": feat,
                        "month": month,
                        "quantile": int(row["bucket"]),
                        "avg_future_ret": float(row["future_ret_h"]),
                    }
                )
    return pd.DataFrame(rows)


def _calc_factor_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    if ic_df.empty:
        return pd.DataFrame(columns=["feature", "ic_mean", "ic_std", "positive_month_ratio", "sample_months"])
    summary = (
        ic_df.groupby("feature")["ic"]
        .agg(
            ic_mean="mean",
            ic_std="std",
            positive_month_ratio=lambda s: float((s > 0).mean()),
            sample_months="count",
        )
        .reset_index()
        .sort_values("ic_mean", ascending=False)
    )
    return summary


def _table_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "無資料"
    # 避免 pandas.to_markdown 需要額外 tabulate 依賴
    return "```\n" + df.to_string(index=False) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description="10 年因子研究")
    parser.add_argument("--max-features", type=int, default=50, help="最多研究幾個因子（依缺失率排序）")
    args = parser.parse_args()

    config = load_config()
    output_dir = PROJECT_ROOT / "artifacts" / "ai_answers"
    output_dir.mkdir(parents=True, exist_ok=True)

    with get_session() as session:
        max_label_date = session.query(func.max(Label.trading_date)).scalar()
        if max_label_date is None:
            raise RuntimeError("labels 表無資料，請先跑 pipeline-build")
        start_date = max_label_date - timedelta(days=365 * 10)
        feat_stmt = (
            select(Feature.stock_id, Feature.trading_date, Feature.features_json)
            .where(Feature.trading_date.between(start_date, max_label_date))
            .order_by(Feature.trading_date, Feature.stock_id)
        )
        label_stmt = (
            select(Label.stock_id, Label.trading_date, Label.future_ret_h)
            .where(Label.trading_date.between(start_date, max_label_date))
            .order_by(Label.trading_date, Label.stock_id)
        )
        feat_df = pd.read_sql(feat_stmt, session.get_bind())
        label_df = pd.read_sql(label_stmt, session.get_bind())

    if feat_df.empty or label_df.empty:
        raise RuntimeError("features 或 labels 無資料")

    fmat = _parse_features(feat_df["features_json"])
    feat_df = feat_df.drop(columns=["features_json"]).reset_index(drop=True)
    merged = pd.concat([feat_df, fmat], axis=1).merge(label_df, on=["stock_id", "trading_date"], how="inner")
    merged = merged[merged["stock_id"].astype(str).str.fullmatch(r"\d{4}")].copy()
    merged["future_ret_h"] = pd.to_numeric(merged["future_ret_h"], errors="coerce")

    feature_cols = [
        c
        for c in merged.columns
        if c not in {"stock_id", "trading_date", "future_ret_h"}
        and pd.api.types.is_numeric_dtype(merged[c])
    ]
    # 先保留缺失率較低的因子，避免研究時間過長
    miss_rate = merged[feature_cols].isna().mean().sort_values()
    feature_cols = miss_rate.index.tolist()[: args.max_features]

    ic_monthly = _calc_monthly_ic(merged, feature_cols)
    quantile_df = _calc_quantile_returns(merged, feature_cols, q=10)
    ic_summary = _calc_factor_summary(ic_monthly)

    ic_path = output_dir / "factor_ic_report.csv"
    quantile_path = output_dir / "factor_quantile_report.csv"
    summary_path = output_dir / "factor_summary.csv"
    coverage_path = output_dir / "research_universe_and_coverage.csv"
    md_path = output_dir / "strategy_research_summary.md"

    ic_monthly.to_csv(ic_path, index=False)
    quantile_df.to_csv(quantile_path, index=False)
    ic_summary.to_csv(summary_path, index=False)
    coverage = (
        merged.groupby("trading_date", as_index=False)
        .agg(stocks=("stock_id", "nunique"), rows=("stock_id", "count"))
        .sort_values("trading_date")
    )
    coverage.to_csv(coverage_path, index=False)

    top = ic_summary.head(10)
    weak = ic_summary.sort_values("ic_mean").head(10)
    md = [
        "# 10年因子研究摘要",
        "",
        f"- 研究區間：{start_date} ~ {max_label_date}",
        f"- 樣本筆數：{len(merged):,}",
        f"- 因子數：{len(feature_cols)}",
        "",
        "## Top 10 因子（依 ic_mean）",
        _table_text(top),
        "",
        "## 待淘汰因子（ic_mean 最低）",
        _table_text(weak),
        "",
        "## 輸出檔案",
        f"- {ic_path}",
        f"- {quantile_path}",
        f"- {summary_path}",
        f"- {coverage_path}",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"[factor-research] wrote: {ic_path}")
    print(f"[factor-research] wrote: {quantile_path}")
    print(f"[factor-research] wrote: {summary_path}")
    print(f"[factor-research] wrote: {coverage_path}")
    print(f"[factor-research] wrote: {md_path}")


if __name__ == "__main__":
    main()
