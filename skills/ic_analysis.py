"""IC 分析工具：從 DB 讀取 features 與 forward return，計算每個特徵的 IC / ICIR / 勝率，
並輸出 artifacts/ic_report.csv 與 artifacts/ic_report.md。
特別標注 CLEANUP_REPORT.md 中標記 LOW_IC? 的特徵，並給出保留 / 降級 / 移除建議。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import Feature, Label, RawPrice

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# 標注 CLEANUP_REPORT.md 中 LOW_IC? 的特徵
LOW_IC_CANDIDATES = {
    "vol_20",
    "dealer_net_5",
    "dealer_net_20",
    "drawdown_60",
    "fund_revenue_trend_3m",
    "theme_hot_score",
}

# 判斷標準
# |IC| < 0.02 且 ICIR < 0.3 → 建議移除
# |IC| 0.02~0.05 且 ICIR < 0.5 → 建議降級到 EXTENDED
# |IC| > 0.05 → 保留
def _recommend(ic_mean: Optional[float], icir: Optional[float]) -> str:
    if ic_mean is None or icir is None or np.isnan(ic_mean) or np.isnan(icir):
        return "資料不足"
    abs_ic = abs(ic_mean)
    if abs_ic < 0.02 and icir < 0.3:
        return "建議移除"
    if abs_ic < 0.05 and icir < 0.5:
        return "建議降級→EXTENDED"
    return "保留"


def _compute_forward_returns(session: Session, horizons: List[int]) -> pd.DataFrame:
    """從 raw_prices 計算多個 horizon 的 forward return。"""
    stmt = select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close).order_by(
        RawPrice.stock_id, RawPrice.trading_date
    )
    prices = pd.read_sql(stmt, session.get_bind())
    prices["trading_date"] = pd.to_datetime(prices["trading_date"])
    prices["close"] = prices["close"].astype(float)

    result = prices[["stock_id", "trading_date"]].copy()
    for h in horizons:
        col = f"future_ret_{h}d"
        prices[col] = prices.groupby("stock_id")["close"].transform(
            lambda s: s.shift(-h) / s - 1
        )
        result[col] = prices[col].values
    return result


def _load_features(session: Session) -> pd.DataFrame:
    """從 DB 讀取 features_json，展開為寬表。"""
    stmt = select(Feature.stock_id, Feature.trading_date, Feature.features_json).order_by(
        Feature.trading_date
    )
    raw = pd.read_sql(stmt, session.get_bind())
    if raw.empty:
        return pd.DataFrame()

    raw["trading_date"] = pd.to_datetime(raw["trading_date"])

    def _parse(val):
        if val is None:
            return {}
        if isinstance(val, dict):
            return val
        return json.loads(val)

    parsed = [_parse(v) for v in raw["features_json"]]
    feat_df = pd.json_normalize(parsed)
    feat_df.insert(0, "stock_id", raw["stock_id"].values)
    feat_df.insert(1, "trading_date", raw["trading_date"].values)
    return feat_df


def _compute_ic_table(
    feat_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    horizons: List[int],
) -> pd.DataFrame:
    """對每個特徵、每個 horizon，計算截面 IC（Spearman），再彙總 ICIR 與勝率。"""
    merged = feat_df.merge(returns_df, on=["stock_id", "trading_date"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    feature_cols = [
        c for c in feat_df.columns if c not in ("stock_id", "trading_date")
    ]
    ret_cols = [f"future_ret_{h}d" for h in horizons]

    rows: List[Dict] = []
    for feat in feature_cols:
        for h, ret_col in zip(horizons, ret_cols):
            sub = merged[["trading_date", feat, ret_col]].dropna()
            if sub.empty or sub[feat].std() == 0:
                rows.append({
                    "feature": feat,
                    "horizon": f"{h}d",
                    "ic_mean": None,
                    "ic_std": None,
                    "icir": None,
                    "win_rate": None,
                    "n_dates": 0,
                })
                continue

            # 截面 IC：每個交易日計算一次 Spearman，再對日期取均值
            daily_ic = []
            for _, grp in sub.groupby("trading_date"):
                if len(grp) < 5:
                    continue
                ic_val = spearmanr(grp[feat], grp[ret_col]).correlation
                if not np.isnan(ic_val):
                    daily_ic.append(ic_val)

            if not daily_ic:
                rows.append({
                    "feature": feat,
                    "horizon": f"{h}d",
                    "ic_mean": None,
                    "ic_std": None,
                    "icir": None,
                    "win_rate": None,
                    "n_dates": 0,
                })
                continue

            ic_arr = np.array(daily_ic)
            ic_mean = float(np.mean(ic_arr))
            ic_std = float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else float("nan")
            icir = ic_mean / ic_std if ic_std > 0 else float("nan")
            win_rate = float((ic_arr > 0).mean())

            rows.append({
                "feature": feat,
                "horizon": f"{h}d",
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": icir,
                "win_rate": win_rate,
                "n_dates": len(ic_arr),
            })

    return pd.DataFrame(rows)


def _build_summary(ic_table: pd.DataFrame) -> pd.DataFrame:
    """對每個特徵取所有 horizon 的平均 IC / ICIR，給出建議。"""
    if ic_table.empty:
        return pd.DataFrame()

    summary_rows = []
    for feat, grp in ic_table.groupby("feature"):
        valid = grp.dropna(subset=["ic_mean", "icir"])
        if valid.empty:
            ic_mean = None
            icir = None
        else:
            ic_mean = float(valid["ic_mean"].mean())
            icir = float(valid["icir"].mean())

        rec = _recommend(ic_mean, icir)
        summary_rows.append({
            "feature": feat,
            "is_low_ic_candidate": feat in LOW_IC_CANDIDATES,
            "ic_mean_avg": round(ic_mean, 4) if ic_mean is not None else None,
            "icir_avg": round(icir, 4) if icir is not None else None,
            "recommendation": rec,
        })

    df = pd.DataFrame(summary_rows)
    # 先排序：LOW_IC 候選優先，再按 |ic_mean| 升序
    df["_abs_ic"] = df["ic_mean_avg"].abs().fillna(0)
    df = df.sort_values(["is_low_ic_candidate", "_abs_ic"], ascending=[False, True])
    df = df.drop(columns=["_abs_ic"])
    return df.reset_index(drop=True)


def _write_markdown(ic_table: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# IC 分析報告",
        "",
        f"> 產出時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 一、特徵建議摘要",
        "",
        "| 特徵 | LOW_IC? | IC均值 | ICIR均值 | 建議 |",
        "|------|---------|--------|----------|------|",
    ]
    for _, row in summary.iterrows():
        flag = "⚠️" if row["is_low_ic_candidate"] else ""
        ic = f"{row['ic_mean_avg']:.4f}" if row["ic_mean_avg"] is not None else "N/A"
        icir = f"{row['icir_avg']:.4f}" if row["icir_avg"] is not None else "N/A"
        lines.append(f"| `{row['feature']}` | {flag} | {ic} | {icir} | {row['recommendation']} |")

    lines += [
        "",
        "---",
        "",
        "## 二、各 Horizon IC 明細",
        "",
        "| 特徵 | Horizon | IC均值 | IC標準差 | ICIR | 勝率 | 有效日數 |",
        "|------|---------|--------|---------|------|------|---------|",
    ]
    for _, row in ic_table.sort_values(["feature", "horizon"]).iterrows():
        ic = f"{row['ic_mean']:.4f}" if row["ic_mean"] is not None else "N/A"
        ic_std = f"{row['ic_std']:.4f}" if row["ic_std"] is not None else "N/A"
        icir = f"{row['icir']:.4f}" if row["icir"] is not None else "N/A"
        wr = f"{row['win_rate']:.2%}" if row["win_rate"] is not None else "N/A"
        lines.append(
            f"| `{row['feature']}` | {row['horizon']} | {ic} | {ic_std} | {icir} | {wr} | {int(row['n_dates'])} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 三、判斷標準",
        "",
        "| 條件 | 建議 |",
        "|------|------|",
        "| \\|IC\\| < 0.02 且 ICIR < 0.3 | 建議移除 |",
        "| \\|IC\\| 0.02~0.05 且 ICIR < 0.5 | 建議降級→EXTENDED |",
        "| \\|IC\\| > 0.05 | 保留 |",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(config=None, db_session: Session = None, horizons: List[int] = None, **kwargs) -> Dict:
    """執行 IC 分析，輸出 CSV 與 Markdown 報告。

    Args:
        config: AppConfig（可為 None，僅用於介面一致性）
        db_session: SQLAlchemy Session。若為 None，自動從 get_session() 取得。
        horizons: forward return 的天期列表，預設 [5, 10, 20]。
    Returns:
        {"csv": str, "md": str, "features_analyzed": int}
    """
    if horizons is None:
        horizons = [5, 10, 20]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACT_DIR / "ic_report.csv"
    md_path = ARTIFACT_DIR / "ic_report.md"

    def _run(session: Session) -> Dict:
        print("[ic_analysis] 讀取 features...")
        feat_df = _load_features(session)
        if feat_df.empty:
            print("[ic_analysis] features 表為空，中止。")
            return {"csv": str(csv_path), "md": str(md_path), "features_analyzed": 0}

        print(f"[ic_analysis] 讀取 raw_prices 計算 {horizons} 天期 forward return...")
        returns_df = _compute_forward_returns(session, horizons)

        print("[ic_analysis] 計算截面 IC...")
        ic_table = _compute_ic_table(feat_df, returns_df, horizons)
        if ic_table.empty:
            print("[ic_analysis] IC 計算結果為空，中止。")
            return {"csv": str(csv_path), "md": str(md_path), "features_analyzed": 0}

        summary = _build_summary(ic_table)

        # 輸出 CSV（詳細明細）
        ic_table.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 輸出 Markdown（含摘要與明細）
        _write_markdown(ic_table, summary, md_path)

        n_feat = ic_table["feature"].nunique()
        print(f"[ic_analysis] 完成。分析 {n_feat} 個特徵，輸出至 {md_path}")
        return {"csv": str(csv_path), "md": str(md_path), "features_analyzed": n_feat}

    if db_session is not None:
        return _run(db_session)

    with get_session() as session:
        return _run(session)


if __name__ == "__main__":
    result = run()
    print(result)
