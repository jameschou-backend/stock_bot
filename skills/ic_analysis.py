"""IC 分析工具：從 DB 讀取 features 與 price 資料，用 alphalens-reloaded 計算。
產出完整 tear sheet 到 artifacts/alphalens_report/，並保留原本的 ic_report.csv 格式。
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import Feature, RawPrice

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ALPHALENS_DIR = ARTIFACT_DIR / "alphalens_report"

# 標注 CLEANUP_REPORT.md 中 LOW_IC? 的特徵
LOW_IC_CANDIDATES = {
    "vol_20",
    "dealer_net_5",
    "dealer_net_20",
    "drawdown_60",
    "fund_revenue_trend_3m",
    "theme_hot_score",
}


def _recommend(ic_mean: Optional[float], icir: Optional[float]) -> str:
    if ic_mean is None or icir is None or np.isnan(ic_mean) or np.isnan(icir):
        return "資料不足"
    abs_ic = abs(ic_mean)
    if abs_ic < 0.02 and icir < 0.3:
        return "建議移除"
    if abs_ic < 0.05 and icir < 0.5:
        return "建議降級→EXTENDED"
    return "保留"


def _load_prices(session: Session) -> pd.DataFrame:
    """從 DB 讀取收盤價，回傳 MultiIndex(date, asset) Series（alphalens 格式）。"""
    stmt = select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close).order_by(
        RawPrice.stock_id, RawPrice.trading_date
    )
    prices = pd.read_sql(stmt, session.get_bind())
    prices["trading_date"] = pd.to_datetime(prices["trading_date"])
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["close"])

    # alphalens 需要 columns=asset, index=date
    price_pivot = prices.pivot_table(
        index="trading_date", columns="stock_id", values="close"
    )
    price_pivot.index = pd.DatetimeIndex(price_pivot.index)
    price_pivot.index.name = "date"
    return price_pivot


def _load_factors(session: Session) -> pd.DataFrame:
    """從 DB 讀取 features_json，展開為長格式 MultiIndex(date, asset) DataFrame。"""
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
    feat_df.insert(1, "date", raw["trading_date"].values)
    return feat_df.set_index(["date", "stock_id"])


def _run_alphalens_tearsheet(
    factor_series: pd.Series,
    price_pivot: pd.DataFrame,
    factor_name: str,
    output_dir: Path,
    periods: tuple = (5, 10, 20),
) -> Optional[pd.DataFrame]:
    """用 alphalens 計算單一因子的 IC tear sheet，輸出圖表 PNG。"""
    try:
        import alphalens
        from alphalens.utils import get_clean_factor_and_forward_returns
        from alphalens.performance import mean_information_coefficient
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            factor_data = get_clean_factor_and_forward_returns(
                factor_series,
                price_pivot,
                periods=periods,
                quantiles=5,
                max_loss=0.35,
            )

        if factor_data is None or factor_data.empty:
            return None

        # IC tear sheet
        ic = mean_information_coefficient(factor_data)
        factor_dir = output_dir / factor_name
        factor_dir.mkdir(parents=True, exist_ok=True)

        # --- 快速 tear sheet（不需互動模式）---
        try:
            from alphalens.tears import create_information_tear_sheet
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                create_information_tear_sheet(factor_data, long_short=True, group_neutral=False)
            plt.suptitle(factor_name, y=1.02)
            plt.tight_layout()
            plt.savefig(factor_dir / "ic_tearsheet.png", dpi=80, bbox_inches="tight")
            plt.close("all")
        except Exception:
            pass

        return ic

    except Exception as exc:
        print(f"  [alphalens] {factor_name} 失敗：{exc}")
        return None


def _compute_ic_table_scipy(
    feat_df: pd.DataFrame,
    price_pivot: pd.DataFrame,
    horizons: List[int],
) -> pd.DataFrame:
    """對每個特徵、每個 horizon，用 scipy 計算截面 IC（不依賴 alphalens）。"""
    # 計算 forward return
    returns_long: Dict[str, pd.DataFrame] = {}
    for h in horizons:
        ret = price_pivot.pct_change(h).shift(-h)  # t+h / t - 1
        ret_long = ret.stack(dropna=True).reset_index()
        ret_long.columns = ["trading_date", "stock_id", f"future_ret_{h}d"]
        returns_long[h] = ret_long

    # 展開 features
    feat_reset = feat_df.reset_index()
    feat_reset = feat_reset.rename(columns={"date": "trading_date"})
    feature_cols = [c for c in feat_reset.columns if c not in ("trading_date", "stock_id")]

    rows: List[Dict] = []
    for feat in feature_cols:
        for h in horizons:
            ret_col = f"future_ret_{h}d"
            sub = feat_reset[["trading_date", "stock_id", feat]].dropna()
            merged = sub.merge(returns_long[h], on=["trading_date", "stock_id"], how="inner").dropna()
            if merged.empty or merged[feat].std() == 0:
                rows.append({"feature": feat, "horizon": f"{h}d",
                              "ic_mean": None, "ic_std": None,
                              "icir": None, "win_rate": None, "n_dates": 0})
                continue

            daily_ic = []
            for _, grp in merged.groupby("trading_date"):
                if len(grp) < 5:
                    continue
                ic_val = spearmanr(grp[feat], grp[ret_col]).correlation
                if not np.isnan(ic_val):
                    daily_ic.append(ic_val)

            if not daily_ic:
                rows.append({"feature": feat, "horizon": f"{h}d",
                              "ic_mean": None, "ic_std": None,
                              "icir": None, "win_rate": None, "n_dates": 0})
                continue

            ic_arr = np.array(daily_ic)
            ic_mean = float(np.mean(ic_arr))
            ic_std = float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else float("nan")
            icir = ic_mean / ic_std if ic_std > 0 else float("nan")
            win_rate = float((ic_arr > 0).mean())
            rows.append({"feature": feat, "horizon": f"{h}d",
                          "ic_mean": ic_mean, "ic_std": ic_std,
                          "icir": icir, "win_rate": win_rate, "n_dates": len(ic_arr)})

    return pd.DataFrame(rows)


def _build_summary(ic_table: pd.DataFrame) -> pd.DataFrame:
    if ic_table.empty:
        return pd.DataFrame()

    summary_rows = []
    for feat, grp in ic_table.groupby("feature"):
        valid = grp.dropna(subset=["ic_mean", "icir"])
        ic_mean = float(valid["ic_mean"].mean()) if not valid.empty else None
        icir = float(valid["icir"].mean()) if not valid.empty else None
        rec = _recommend(ic_mean, icir)
        summary_rows.append({
            "feature": feat,
            "is_low_ic_candidate": feat in LOW_IC_CANDIDATES,
            "ic_mean_avg": round(ic_mean, 4) if ic_mean is not None else None,
            "icir_avg": round(icir, 4) if icir is not None else None,
            "recommendation": rec,
        })

    df = pd.DataFrame(summary_rows)
    df["_abs_ic"] = df["ic_mean_avg"].abs().fillna(0)
    df = df.sort_values(["is_low_ic_candidate", "_abs_ic"], ascending=[False, True])
    return df.drop(columns=["_abs_ic"]).reset_index(drop=True)


def _write_markdown(ic_table: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# IC 分析報告（alphalens-reloaded）",
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
        "## 三、Alphalens Tear Sheet",
        "",
        f"> 完整圖表輸出至 `artifacts/alphalens_report/<feature_name>/ic_tearsheet.png`",
        "",
        "## 四、判斷標準",
        "",
        "| 條件 | 建議 |",
        "|------|------|",
        "| \\|IC\\| < 0.02 且 ICIR < 0.3 | 建議移除 |",
        "| \\|IC\\| 0.02~0.05 且 ICIR < 0.5 | 建議降級→EXTENDED |",
        "| \\|IC\\| > 0.05 | 保留 |",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(config=None, db_session: Session = None, horizons: List[int] = None,
        run_alphalens_tearsheet: bool = True,
        max_factors_tearsheet: int = 10,
        **kwargs) -> Dict:
    """執行 IC 分析，輸出 CSV、Markdown 與 alphalens tear sheet。

    Args:
        config: AppConfig（可為 None）
        db_session: SQLAlchemy Session
        horizons: forward return 天期列表，預設 [5, 10, 20]
        run_alphalens_tearsheet: 是否產出 alphalens 圖表（依 |IC| 排序取前 N 個因子）
        max_factors_tearsheet: 最多產出幾個因子的 tear sheet（避免太慢）
    Returns:
        {"csv": str, "md": str, "features_analyzed": int, "alphalens_dir": str}
    """
    if horizons is None:
        horizons = [5, 10, 20]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ALPHALENS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACT_DIR / "ic_report.csv"
    md_path = ARTIFACT_DIR / "ic_report.md"

    def _run(session: Session) -> Dict:
        print("[ic_analysis] 讀取 raw_prices ...")
        price_pivot = _load_prices(session)
        if price_pivot.empty:
            print("[ic_analysis] prices 表為空，中止。")
            return {"csv": str(csv_path), "md": str(md_path),
                    "features_analyzed": 0, "alphalens_dir": str(ALPHALENS_DIR)}

        print("[ic_analysis] 讀取 features ...")
        factor_df = _load_factors(session)
        if factor_df.empty:
            print("[ic_analysis] features 表為空，中止。")
            return {"csv": str(csv_path), "md": str(md_path),
                    "features_analyzed": 0, "alphalens_dir": str(ALPHALENS_DIR)}

        print(f"[ic_analysis] 計算截面 IC（horizons={horizons}）...")
        ic_table = _compute_ic_table_scipy(factor_df, price_pivot, horizons)
        if ic_table.empty:
            print("[ic_analysis] IC 計算結果為空，中止。")
            return {"csv": str(csv_path), "md": str(md_path),
                    "features_analyzed": 0, "alphalens_dir": str(ALPHALENS_DIR)}

        summary = _build_summary(ic_table)

        # 輸出 CSV（向下相容格式）
        ic_table.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 輸出 Markdown
        _write_markdown(ic_table, summary, md_path)

        n_feat = ic_table["feature"].nunique()

        # ── alphalens tear sheet（對 IC 絕對值最高的 N 個因子）──
        if run_alphalens_tearsheet:
            # 選出 |ic_mean| 最大的因子（5d horizon）
            best_features = (
                ic_table[ic_table["horizon"] == "5d"]
                .dropna(subset=["ic_mean"])
                .assign(abs_ic=lambda df: df["ic_mean"].abs())
                .sort_values("abs_ic", ascending=False)
                .head(max_factors_tearsheet)["feature"]
                .tolist()
            )
            print(f"[ic_analysis] 產出 alphalens tear sheet（{len(best_features)} 個因子）...")
            al_periods = tuple(horizons)
            for feat_name in best_features:
                if feat_name not in factor_df.columns:
                    continue
                factor_series = factor_df[feat_name].dropna()
                _run_alphalens_tearsheet(
                    factor_series, price_pivot, feat_name, ALPHALENS_DIR, al_periods
                )

        print(f"[ic_analysis] 完成。分析 {n_feat} 個特徵，輸出至 {md_path}")
        return {
            "csv": str(csv_path),
            "md": str(md_path),
            "features_analyzed": n_feat,
            "alphalens_dir": str(ALPHALENS_DIR),
        }

    if db_session is not None:
        return _run(db_session)

    with get_session() as session:
        return _run(session)


if __name__ == "__main__":
    result = run()
    print(result)
