"""
eval_signal_perf.py — 歷史訊號績效評估

每日推薦訊號（top_candidates）與實際後續報酬的對照分析：

指標：
  - IC（Spearman）：分數與實際 N 日報酬的相關係數（模型有無預測力）
  - 命中率：推薦股票中漲的比例（>0）
  - 超額報酬：推薦 Top-N vs 全市場當日平均
  - 持倉勝率：實際持倉（holdings）的損益分佈

用法：
  python scripts/eval_signal_perf.py              # Strategy C，分析 1d/3d/5d
  python scripts/eval_signal_perf.py --strategy d
  python scripts/eval_signal_perf.py --topn 5     # 只看前 5 名
  python scripts/eval_signal_perf.py --horizon 3  # 只看 3d 報酬
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config

SIGNAL_DIR = ROOT / "artifacts" / "daily_signal"


# ──────────────────────────────────────────────────────────
# 資料載入
# ──────────────────────────────────────────────────────────

def load_signal_files(strategy: str = "c") -> List[Dict]:
    files = sorted(SIGNAL_DIR.glob(f"strategy_{strategy}_20*.json"))
    signals = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("top_candidates"):
                signals.append(d)
        except Exception as e:
            print(f"[warn] {f.name} 解析失敗：{e}")
    return signals


def fetch_prices_batch(stock_ids: List[str], start: date, end: date) -> pd.DataFrame:
    """從 DB 批次查詢收盤價，回傳 DataFrame (trading_date × stock_id)。"""
    from app.db import get_session
    from app.models import RawPrice
    from sqlalchemy import select

    with get_session() as session:
        rows = session.execute(
            select(RawPrice.trading_date, RawPrice.stock_id, RawPrice.close)
            .where(
                RawPrice.stock_id.in_(stock_ids),
                RawPrice.trading_date >= start,
                RawPrice.trading_date <= end,
            )
        ).fetchall()

    df = pd.DataFrame(rows, columns=["trading_date", "stock_id", "close"])
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    pivot = df.pivot_table(index="trading_date", columns="stock_id", values="close")
    pivot.sort_index(inplace=True)
    return pivot


def get_trading_dates_from_db(start: date, end: date) -> List[date]:
    """從 DB raw_prices 撈有資料的交易日。"""
    from app.db import get_session
    from app.models import RawPrice
    from sqlalchemy import select, func, distinct

    with get_session() as session:
        rows = session.execute(
            select(distinct(RawPrice.trading_date))
            .where(
                RawPrice.trading_date >= start,
                RawPrice.trading_date <= end,
            )
            .order_by(RawPrice.trading_date)
        ).fetchall()
    return sorted([r[0] for r in rows])


def next_n_trading_date(
    ref_date: date, n: int, trading_dates: List[date]
) -> Optional[date]:
    """取 ref_date 之後第 n 個交易日。"""
    future = [d for d in trading_dates if d > ref_date]
    if len(future) < n:
        return None
    return future[n - 1]


# ──────────────────────────────────────────────────────────
# 計算指標
# ──────────────────────────────────────────────────────────

def spearman_ic(scores: List[float], returns: List[float]) -> float:
    """Spearman IC（-1 ~ 1），NaN 時回傳 NaN。"""
    if len(scores) < 3:
        return float("nan")
    s = pd.Series(scores)
    r = pd.Series(returns)
    mask = r.notna() & s.notna()
    if mask.sum() < 3:
        return float("nan")
    return float(s[mask].corr(r[mask], method="spearman"))


def compute_day_metrics(
    sig: Dict,
    price_pivot: pd.DataFrame,
    trading_dates: List[date],
    horizons: List[int],
    topn: int,
) -> Dict:
    """計算單一訊號日的績效指標。"""
    sig_date = date.fromisoformat(sig["date"])
    candidates = sig.get("top_candidates", [])[:topn]
    if not candidates:
        return {}

    result = {"date": sig_date}

    # 當日股票全市場報酬（用於計算超額）
    # 取該日在 price_pivot 上有值的所有股票
    if sig_date not in price_pivot.index:
        return {}  # 當日無價格

    close_t0 = price_pivot.loc[sig_date]

    for h in horizons:
        future_date = next_n_trading_date(sig_date, h, trading_dates)
        if future_date is None or future_date not in price_pivot.index:
            result[f"ic_{h}d"]      = float("nan")
            result[f"hit_{h}d"]     = float("nan")
            result[f"excess_{h}d"]  = float("nan")
            result[f"top_ret_{h}d"] = float("nan")
            result[f"mkt_ret_{h}d"] = float("nan")
            continue

        close_th = price_pivot.loc[future_date]

        # 推薦股票報酬
        scores, rets = [], []
        for c in candidates:
            sid = str(c["stock_id"])
            if sid in close_t0 and sid in close_th:
                p0, ph = float(close_t0[sid]), float(close_th[sid])
                if p0 > 0 and not np.isnan(p0) and not np.isnan(ph):
                    scores.append(float(c["score_today"]))
                    rets.append(ph / p0 - 1)

        if not rets:
            result[f"ic_{h}d"]      = float("nan")
            result[f"hit_{h}d"]     = float("nan")
            result[f"excess_{h}d"]  = float("nan")
            result[f"top_ret_{h}d"] = float("nan")
            result[f"mkt_ret_{h}d"] = float("nan")
            continue

        # 全市場當日報酬（計算超額基準）
        all_p0  = close_t0.dropna()
        all_ph  = close_th.dropna()
        common  = all_p0.index.intersection(all_ph.index)
        mkt_ret = ((all_ph[common] / all_p0[common] - 1)
                   .replace([np.inf, -np.inf], np.nan)
                   .dropna())
        mkt_mean = float(mkt_ret.mean()) if len(mkt_ret) > 0 else float("nan")

        avg_top_ret = float(np.mean(rets))

        result[f"ic_{h}d"]      = spearman_ic(scores, rets)
        result[f"hit_{h}d"]     = float(np.mean([r > 0 for r in rets]))
        result[f"excess_{h}d"]  = avg_top_ret - mkt_mean
        result[f"top_ret_{h}d"] = avg_top_ret
        result[f"mkt_ret_{h}d"] = mkt_mean
        result[f"n_{h}d"]       = len(rets)

    return result


# ──────────────────────────────────────────────────────────
# 持倉損益分析
# ──────────────────────────────────────────────────────────

def analyze_holdings(
    signals: List[Dict],
    price_pivot: pd.DataFrame,
) -> pd.DataFrame:
    """分析 holdings 中每個實際持倉的損益。"""
    rows = []
    for sig in signals:
        sig_date = date.fromisoformat(sig["date"])
        for h in sig.get("holdings", []):
            action = h.get("action", "")
            if action not in ("hold", "new"):
                continue
            sid        = str(h["stock_id"])
            entry_date = h.get("entry_date")
            days_held  = h.get("days_held", 0)
            score      = h.get("score_today", float("nan"))

            if not entry_date or sig_date not in price_pivot.index:
                continue

            entry_d = date.fromisoformat(entry_date)
            if entry_d not in price_pivot.index or sid not in price_pivot.columns:
                continue

            p_entry   = float(price_pivot.loc[entry_d, sid]) if entry_d in price_pivot.index else float("nan")
            p_current = float(price_pivot.loc[sig_date, sid]) if sig_date in price_pivot.index else float("nan")

            if p_entry > 0 and not np.isnan(p_entry) and not np.isnan(p_current):
                pnl = p_current / p_entry - 1
            else:
                pnl = float("nan")

            rows.append({
                "sig_date":  sig_date,
                "stock_id":  sid,
                "name":      h.get("name", sid),
                "entry_date": entry_d,
                "days_held": days_held,
                "action":    action,
                "score":     score,
                "pnl_pct":   pnl,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy 歷史訊號績效評估")
    parser.add_argument("--strategy", default="c", choices=["c", "d"], help="策略 c/d")
    parser.add_argument("--topn",     type=int,   default=10,  help="只評估前 N 名推薦（預設 10）")
    parser.add_argument("--horizon",  type=int,   default=None, help="只看此 N 日報酬（預設 1/3/5）")
    args = parser.parse_args()

    load_config()

    horizons = [args.horizon] if args.horizon else [1, 3, 5]

    print(f"\n{'='*60}")
    print(f"  Strategy {args.strategy.upper()} 訊號績效評估（Top-{args.topn}）")
    print(f"{'='*60}\n")

    # 載入訊號
    signals = load_signal_files(args.strategy)
    if not signals:
        print(f"❌ 找不到 Strategy {args.strategy.upper()} 訊號檔案，請先執行 make daily-{args.strategy}")
        sys.exit(1)

    print(f"✅ 載入 {len(signals)} 個訊號日（{signals[0]['date']} ~ {signals[-1]['date']}）")

    # 收集所有涉及的股票與日期範圍
    all_stocks: set = set()
    for sig in signals:
        for c in sig.get("top_candidates", []):
            all_stocks.add(str(c["stock_id"]))
        for h in sig.get("holdings", []):
            all_stocks.add(str(h["stock_id"]))

    # 日期範圍：訊號最早日 到 最晚日 + 10 交易日
    start_d = date.fromisoformat(signals[0]["date"])
    end_d   = date.fromisoformat(signals[-1]["date"]) + timedelta(days=20)
    end_d   = min(end_d, date.today())

    print(f"⏳ 查詢 DB 價格（{start_d} ~ {end_d}，{len(all_stocks)} 支股票）...")
    # 全市場也需要，先用 universe 的方式
    # 先抓全部推薦股票 + 取全市場用以算超額，分兩步
    try:
        price_pivot = fetch_prices_batch(list(all_stocks), start_d, end_d)
    except Exception as e:
        print(f"❌ 價格查詢失敗：{e}")
        sys.exit(1)

    # 全市場日期序列
    print("⏳ 查詢交易日曆...")
    try:
        trading_dates = get_trading_dates_from_db(start_d, end_d)
    except Exception as e:
        print(f"❌ 交易日曆查詢失敗：{e}")
        sys.exit(1)

    # ── 逐日計算超額報酬需要全市場價格，但我們先用 top_candidates 做 IC 分析
    # 超額報酬則用同日推薦 vs price_pivot 現有股票的全市場均值（近似）
    print("⏳ 計算逐日指標...")

    # 為了計算全市場超額，需要全市場價格
    # 用更大的 stock set：price_pivot 已有的股票
    # 我們加載全市場以便計算市場均值（讀取 top_candidates 的 above_threshold_stocks 已足夠近似）
    # 此處使用 DB 全市場查詢
    all_mkt_stocks: set = set()
    for sig in signals:
        all_mkt_stocks.update(str(s) for s in sig.get("above_threshold_stocks", []))

    if all_mkt_stocks - all_stocks:
        extra = list(all_mkt_stocks - all_stocks)[:500]  # 最多 500
        try:
            extra_pivot = fetch_prices_batch(extra, start_d, end_d)
            price_pivot = pd.concat([price_pivot, extra_pivot], axis=1)
            price_pivot = price_pivot.loc[:, ~price_pivot.columns.duplicated()]
        except Exception:
            pass

    daily_rows = []
    for sig in signals:
        row = compute_day_metrics(sig, price_pivot, trading_dates, horizons, args.topn)
        if row:
            daily_rows.append(row)

    if not daily_rows:
        print("❌ 無法計算任何指標（可能缺乏後續價格資料）")
        sys.exit(1)

    df = pd.DataFrame(daily_rows).set_index("date").sort_index()

    # ── 顯示逐日 IC 表 ──
    print(f"\n{'─'*60}")
    print(f"  逐日 IC / 命中率 / 超額報酬")
    print(f"{'─'*60}")

    for h in horizons:
        ic_col     = f"ic_{h}d"
        hit_col    = f"hit_{h}d"
        excess_col = f"excess_{h}d"
        top_col    = f"top_ret_{h}d"
        mkt_col    = f"mkt_ret_{h}d"

        if ic_col not in df.columns:
            continue

        print(f"\n【{h}日後報酬】")
        sub = df[[ic_col, hit_col, excess_col, top_col, mkt_col]].dropna()
        sub.columns = ["IC", "命中率", "超額", "Top推薦均值", "市場均值"]

        for idx, row in sub.iterrows():
            ic_str  = f"{row['IC']:+.3f}"
            hit_str = f"{row['命中率']*100:.0f}%"
            ex_str  = f"{row['超額']*100:+.2f}%"
            top_str = f"{row['Top推薦均值']*100:+.2f}%"
            mkt_str = f"{row['市場均值']*100:+.2f}%"
            print(f"  {idx}  IC={ic_str}  命中={hit_str}  超額={ex_str}  推薦={top_str}  市場={mkt_str}")

    # ── 彙總統計 ──
    print(f"\n{'─'*60}")
    print(f"  彙總統計")
    print(f"{'─'*60}")

    summary_rows = []
    for h in horizons:
        ic_col     = f"ic_{h}d"
        hit_col    = f"hit_{h}d"
        excess_col = f"excess_{h}d"
        top_col    = f"top_ret_{h}d"

        if ic_col not in df.columns:
            continue

        ic_vals  = df[ic_col].dropna()
        hit_vals = df[hit_col].dropna()
        ex_vals  = df[excess_col].dropna()
        top_vals = df[top_col].dropna()

        summary_rows.append({
            "horizon":   f"{h}d",
            "樣本天數":   len(ic_vals),
            "IC 均值":    ic_vals.mean(),
            "IC 標準差":  ic_vals.std(),
            "ICIR":       ic_vals.mean() / ic_vals.std() if ic_vals.std() > 0 else float("nan"),
            "IC>0 天數":  (ic_vals > 0).sum(),
            "命中率均值": hit_vals.mean(),
            "超額均值":   ex_vals.mean(),
            "超額>0 天數": (ex_vals > 0).sum(),
            "累積超額":   ex_vals.sum(),
        })

    if summary_rows:
        sum_df = pd.DataFrame(summary_rows).set_index("horizon")
        for col in ["IC 均值", "IC 標準差", "ICIR", "命中率均值", "超額均值", "累積超額"]:
            if col in sum_df.columns:
                sum_df[col] = sum_df[col].map(lambda x: f"{x:+.4f}" if not np.isnan(x) else "N/A")
        print(sum_df.to_string())

    # ── 持倉損益分析 ──
    print(f"\n{'─'*60}")
    print(f"  持倉損益分析（holdings）")
    print(f"{'─'*60}")

    holdings_df = analyze_holdings(signals, price_pivot)

    if not holdings_df.empty:
        # 按 action 分群
        for act in ["new", "hold"]:
            sub = holdings_df[holdings_df["action"] == act].dropna(subset=["pnl_pct"])
            if sub.empty:
                continue
            label = "新進場" if act == "new" else "持倉中"
            hit = (sub["pnl_pct"] > 0).mean()
            avg_pnl = sub["pnl_pct"].mean()
            print(f"\n{label}（n={len(sub)}）：")
            print(f"  命中率 {hit*100:.1f}%  |  平均損益 {avg_pnl*100:+.2f}%")

            # 分數 vs pnl IC
            ic = spearman_ic(sub["score"].tolist(), sub["pnl_pct"].tolist())
            print(f"  Score-PnL IC（Spearman）= {ic:+.3f}")

        # 最差 / 最佳持倉
        worst = holdings_df.dropna(subset=["pnl_pct"]).nsmallest(5, "pnl_pct")[
            ["sig_date", "stock_id", "name", "days_held", "pnl_pct"]
        ]
        best  = holdings_df.dropna(subset=["pnl_pct"]).nlargest(5, "pnl_pct")[
            ["sig_date", "stock_id", "name", "days_held", "pnl_pct"]
        ]

        print(f"\n📉 虧損最大 5 筆：")
        for _, r in worst.iterrows():
            print(f"  {r['sig_date']} {r['stock_id']} {r['name']}  {r['days_held']}天  {r['pnl_pct']*100:+.2f}%")

        print(f"\n📈 獲利最大 5 筆：")
        for _, r in best.iterrows():
            print(f"  {r['sig_date']} {r['stock_id']} {r['name']}  {r['days_held']}天  {r['pnl_pct']*100:+.2f}%")
    else:
        print("（無持倉記錄）")

    # ── 優化建議 ──
    print(f"\n{'─'*60}")
    print(f"  優化建議")
    print(f"{'─'*60}")

    for h in horizons:
        ic_col  = f"ic_{h}d"
        ex_col  = f"excess_{h}d"
        if ic_col not in df.columns:
            continue
        ic_vals = df[ic_col].dropna()
        ex_vals = df[ex_col].dropna()
        if len(ic_vals) < 3:
            continue

        ic_mean = ic_vals.mean()
        ic_std  = ic_vals.std()
        icir    = ic_mean / ic_std if ic_std > 0 else float("nan")
        ex_mean = ex_vals.mean()
        ex_pos  = (ex_vals > 0).mean()

        print(f"\n[{h}日] ICIR={icir:+.3f}  超額均值={ex_mean*100:+.2f}%  超額勝率={ex_pos*100:.0f}%")

        if ic_mean > 0.03 and icir > 0.3:
            print(f"  ✅ 模型對 {h}d 報酬有明顯預測力，維持現有配置")
        elif ic_mean > 0 and icir > 0:
            print(f"  ⚠️  預測力偏弱（ICIR={icir:.3f}），建議：")
            print(f"     1. 增加訓練樣本多樣性")
            print(f"     2. 縮小 rank_threshold（現 0.20 → 0.15）只取更高分股票")
        else:
            print(f"  ❌ 模型對 {h}d 報酬幾乎無預測力，建議重訓或換特徵")

        if ex_pos < 0.5:
            print(f"  ⚠️  超額勝率不足 50%，市場可能進入不利模式")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
