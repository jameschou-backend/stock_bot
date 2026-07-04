#!/usr/bin/env python
"""容量分析：你的資金量 vs 流動性門檻（2026-07-04，基準 v2 後續）。

基準 v2 顯示 alpha 大半集中在 0.5 億門檻外的微型股（歸因臂 +970% vs 主臂 +431%）。
但 0.5 億門檻是「機構級」保守假設——個人資金量若遠小於標的日成交值，
真實可執行的 universe 比門檻臂大得多。本工具回答：

  給定資金量 C、持股數 N、單日參與率上限 p（成交金額佔比），
  每檔部位 C/N 需要 20 日均成交值 ≥ C/(N×p) 才能在一天內進出不砸盤。
  → 你的「個人等效門檻」是多少？比 0.5 億寬多少？能吃回多少微型股 alpha？

用法：
    python scripts/capacity_analysis.py --capital 1000000            # 100 萬 TWD
    python scripts/capacity_analysis.py --capital 5000000 --topn 30 --participation 0.10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import text

from app.db import get_session


def main() -> int:
    ap = argparse.ArgumentParser(description="資金容量 vs 流動性門檻分析")
    ap.add_argument("--capital", type=float, required=True, help="總資金（TWD）")
    ap.add_argument("--topn", type=int, default=30, help="持股數（預設 30，等權）")
    ap.add_argument("--participation", type=float, default=0.10,
                    help="單日成交參與率上限（預設 0.10 = 部位不超過日成交金額的 10%%）")
    args = ap.parse_args()

    position = args.capital / args.topn
    required_amt20 = position / args.participation

    with get_session() as s:
        latest = s.execute(text("SELECT max(trading_date) FROM raw_prices")).scalar()
        # 近 20 日均成交金額（close × volume），全 universe（四碼）
        df = pd.read_sql(text(
            "SELECT stock_id, AVG(close * volume) AS amt_20 FROM raw_prices "
            "WHERE trading_date > (SELECT DATE_SUB(max(trading_date), INTERVAL 40 DAY) FROM raw_prices) "
            "AND stock_id REGEXP '^[0-9]{4}$' GROUP BY stock_id"
        ), s.connection())

    df["amt_20"] = pd.to_numeric(df["amt_20"], errors="coerce")
    df = df.dropna()
    n_total = len(df)
    n_my = int((df["amt_20"] >= required_amt20).sum())
    n_05e = int((df["amt_20"] >= 0.5e8).sum())

    print(f"資料日：{latest}，universe：{n_total} 檔（四碼）")
    print(f"資金 {args.capital:,.0f} / topN {args.topn} = 每檔部位 {position:,.0f} TWD")
    print(f"參與率 ≤{args.participation:.0%} → 個人等效門檻：20 日均成交值 ≥ {required_amt20:,.0f} TWD"
          f"（= {required_amt20/1e8:.4f} 億）")
    print()
    print(f"  你的等效門檻可交易檔數：{n_my}（{n_my/n_total:.0%}）")
    print(f"  現行 0.5 億門檻可交易檔數：{n_05e}（{n_05e/n_total:.0%}）")
    if required_amt20 < 0.5e8:
        gap = n_my - n_05e
        print(f"\n→ 你的資金量下 0.5 億門檻**過嚴**：多出 {gap} 檔可交易"
              f"（正是歸因臂 alpha 集中區）。")
        print(f"  建議實驗：--production-baseline 但 --min-avg-turnover {required_amt20/1e8:.2f}"
              f"（個人口徑臂），與主臂/歸因臂三方比較。")
    else:
        print(f"\n→ 你的資金量已達機構級門檻（≥0.5 億），現行主臂即個人口徑。")

    # 門檻敏感度表
    print("\n門檻敏感度（可交易檔數）：")
    for thr in (0.01e8, 0.05e8, 0.1e8, 0.3e8, 0.5e8, 1e8, 3e8):
        n = int((df["amt_20"] >= thr).sum())
        marker = " ← 你的等效門檻附近" if abs(thr - required_amt20) <= thr * 0.5 else ""
        print(f"  ≥{thr/1e8:>5.2f} 億：{n:4d} 檔（{n/n_total:.0%}）{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
