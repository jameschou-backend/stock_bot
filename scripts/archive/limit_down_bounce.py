#!/usr/bin/env python
"""#4 跌停斷頭隔日反彈 backtest（行為金融 / TWSE 微結構）

理論（Agent 2 提案）：
  - 連兩根跌停 + 高融資餘額 = 強制斷頭風險
  - T 日早盤斷頭瀑布（9:00-9:30 集中爆量下殺）
  - 13:00 前已釋放賣壓 → 短線反彈進場
  - T+1 開盤集合競價賣出

⚠️ Backtest 限制：
  - 系統無 intraday tick data
  - 用 T 日 close 作為「下午尾盤已反彈進場」的近似
  - 用 T+1 open 作為「開盤集合競價賣出」的近似
  - 真實表現可能因盤中跳價而有 ±2~3% 誤差
  - 此 backtest 主要用來估計 alpha 是否存在；實盤需手動盤中執行

進場條件：
  1. T-2 與 T-1 兩日均接近跌停（close vs prev_close ≤ -9%）
  2. T-1 融資餘額占股本比 > 8%（或近 10 日融資增加 > 20%）
  3. T 日 low < open × 0.95（盤中曾深跌）
  4. T 日 volume > 1.5 × ma_20（爆量）
  5. T 日 close > low × 1.02（已從低點反彈，非繼續跌停）
  6. 排除：amt_20 < 1000 萬（流動性過低）

出場條件：
  - T+1 open 賣出（市價單）

用法：
  python scripts/limit_down_bounce.py --months 36
  python scripts/limit_down_bounce.py --scan  # 今日 scan 模式
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.config import load_config
from app.db import get_session


# ──────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────
def load_prices(session, start: date, end: date) -> pd.DataFrame:
    pre = start - timedelta(days=60)
    sql = text("""
        SELECT stock_id, trading_date, open, high, low, close, volume
        FROM raw_prices
        WHERE trading_date >= :s AND trading_date <= :e
          AND stock_id REGEXP '^[0-9]{4}$'
        ORDER BY stock_id, trading_date
    """)
    df = pd.read_sql(sql, session.bind, params={"s": pre, "e": end})
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"])


def load_margin(session, start: date, end: date) -> pd.DataFrame:
    pre = start - timedelta(days=30)
    sql = text("""
        SELECT stock_id, trading_date,
               margin_purchase_balance, margin_purchase_limit
        FROM raw_margin_short
        WHERE trading_date >= :s AND trading_date <= :e
          AND stock_id REGEXP '^[0-9]{4}$'
    """)
    df = pd.read_sql(sql, session.bind, params={"s": pre, "e": end})
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    return df


def compute_signals(price_df: pd.DataFrame, margin_df: pd.DataFrame,
                    hold_days: int = 1) -> pd.DataFrame:
    """For each (stock_id, trading_date)，計算 entry signals。

    hold_days: 持有天數（1=T+1 open 出場、5=T+5 close 出場 etc）
    """
    t0 = time.time()
    parts = []
    for sid, g in price_df.groupby("stock_id"):
        g = g.sort_values("trading_date").reset_index(drop=True).copy()
        g["prev_close"] = g["close"].shift(1)
        g["daily_ret"] = g["close"] / g["prev_close"] - 1
        # 跌停判定：close vs prev_close ≤ -9%（容許 0.5% 誤差）
        g["is_limit_down"] = (g["daily_ret"] <= -0.09).astype(int)
        g["prev_limit_down"] = g["is_limit_down"].shift(1).fillna(0)
        # 連兩跌停
        g["two_limit_down"] = ((g["is_limit_down"] == 1) & (g["prev_limit_down"] == 1)).astype(int)
        # 盤中深跌
        g["low_drop_pct"] = (g["low"] - g["open"]) / g["open"]
        # 從低點反彈
        g["close_bounce"] = (g["close"] - g["low"]) / g["low"]
        # 成交量
        g["vol_ma_20"] = g["volume"].rolling(20, min_periods=10).mean()
        g["vol_surge"] = g["volume"] / g["vol_ma_20"]
        # 20d 平均成交值
        g["amt_20"] = (g["close"] * g["volume"]).rolling(20, min_periods=10).mean() / 1e4  # 萬元
        # 出場價：hold_days=1 → T+1 open；hold_days>=2 → T+N close
        if hold_days <= 1:
            g["exit_price"] = g["open"].shift(-1)
            g["exit_date"] = g["trading_date"].shift(-1)
        else:
            g["exit_price"] = g["close"].shift(-hold_days)
            g["exit_date"] = g["trading_date"].shift(-hold_days)
        # 保留舊欄位 alias
        g["next_open"] = g["exit_price"]
        g["next_date"] = g["exit_date"]
        parts.append(g)
    g = pd.concat(parts, ignore_index=True)
    print(f"  Price signals computed ({time.time()-t0:.1f}s)")

    # Merge margin balance 變化（10 日趨勢）
    t0 = time.time()
    m_parts = []
    for sid, mg in margin_df.groupby("stock_id"):
        mg = mg.sort_values("trading_date").reset_index(drop=True).copy()
        mg["margin_10d_ago"] = mg["margin_purchase_balance"].shift(10)
        mg["margin_change_10d"] = (
            mg["margin_purchase_balance"] / mg["margin_10d_ago"] - 1
        ).fillna(0)
        mg["margin_util"] = mg["margin_purchase_balance"] / mg["margin_purchase_limit"].replace(0, np.nan)
        m_parts.append(mg)
    margin_df2 = pd.concat(m_parts, ignore_index=True) if m_parts else margin_df
    print(f"  Margin signals computed ({time.time()-t0:.1f}s)")

    # Merge
    g = g.merge(
        margin_df2[["stock_id", "trading_date", "margin_purchase_balance",
                    "margin_util", "margin_change_10d"]],
        on=["stock_id", "trading_date"], how="left",
    )
    return g


def scan_signals(g: pd.DataFrame,
                 min_amt_20: float = 1000,
                 min_low_drop: float = -0.04,
                 min_vol_surge: float = 1.3,
                 min_bounce: float = 0.01,
                 min_margin_change_10d: float = 0.15) -> pd.DataFrame:
    """應用 entry 規則。"""
    cond = (
        (g["two_limit_down"] == 1)
        & (g["low_drop_pct"] <= min_low_drop)       # 盤中曾深跌
        & (g["vol_surge"] >= min_vol_surge)          # 爆量
        & (g["close_bounce"] >= min_bounce)          # 從低點已反彈
        & (g["amt_20"] >= min_amt_20)                # 流動性
        & (g["next_open"].notna())                   # 有 T+1 open
        & (g["margin_change_10d"].fillna(-1) >= min_margin_change_10d)  # 融資累積
    )
    return g[cond].copy()


# ──────────────────────────────────────────────────────────────
# Backtest
# ──────────────────────────────────────────────────────────────
def run_backtest(months: int = 36,
                 initial_capital: float = 1_000_000.0,
                 per_trade_pct: float = 0.10,   # 單筆 10% 資金（最多 2 檔）
                 transaction_cost_pct: float = 0.003,
                 hold_days: int = 1,
                 **scan_kwargs) -> Dict:
    print("=" * 60)
    print("#4 跌停斷頭反彈 Backtest")
    print("=" * 60)
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=months * 30)
    print(f"  Period: {start_date} ~ {end_date}")
    print(f"  Hold days: {hold_days}")
    exit_type = "T+1 open" if hold_days == 1 else f"T+{hold_days} close"
    print(f"  Exit: {exit_type}")

    with get_session() as session:
        price_df = load_prices(session, start_date, end_date)
        margin_df = load_margin(session, start_date, end_date)

    print(f"  Prices: {len(price_df):,} rows")
    print(f"  Margin: {len(margin_df):,} rows")

    g = compute_signals(price_df, margin_df, hold_days=hold_days)
    signals = scan_signals(g, **scan_kwargs)
    # 只保留 backtest 範圍
    signals = signals[signals["trading_date"] >= start_date].copy()
    print(f"  Signals: {len(signals)} 筆觸發")

    # ── Capacity-aware backtest: 同時最多 max_concurrent 部位 ──
    # 若 signal 觸發時已滿倉 → skip (etc 隨日推進)
    signals = signals.sort_values("trading_date").reset_index(drop=True)
    trades = []
    capital = initial_capital
    cash = initial_capital
    # open positions: list of dicts with exit_date
    open_pos = []

    def close_position(p, today):
        nonlocal cash
        fee = p["shares"] * p["exit_price"] * transaction_cost_pct
        net_pnl = p["shares"] * (p["exit_price"] - p["entry_price"]) - p["entry_fee"] - fee
        ret = net_pnl / (p["shares"] * p["entry_price"])
        cash += p["shares"] * p["exit_price"] - fee
        trades.append({
            "stock_id": p["stock_id"],
            "entry_date": str(p["entry_date"]),
            "exit_date": str(p["exit_date"]),
            "entry_price": p["entry_price"],
            "exit_price": p["exit_price"],
            "shares": p["shares"],
            "realized_pnl": net_pnl,
            "realized_pnl_pct": ret,
            "daily_ret_T2": p["daily_ret"],
            "low_drop": p["low_drop"],
            "vol_surge": p["vol_surge"],
            "amt_20_wan": p["amt_20"],
            "margin_change_10d": p["margin_chg"],
        })

    max_concurrent = scan_kwargs.pop("max_concurrent", 2)  # 同時部位上限（預設 2）

    for _, sig in signals.iterrows():
        today_d = sig["trading_date"]
        # 1) 先關掉到期 positions
        still_open = []
        for p in open_pos:
            if p["exit_date"] is None or pd.isna(p["exit_date"]):
                continue
            if p["exit_date"] <= today_d:
                close_position(p, today_d)
            else:
                still_open.append(p)
        open_pos = still_open

        # 2) 嘗試開新倉
        if len(open_pos) >= max_concurrent:
            continue
        if pd.isna(sig["next_open"]) or pd.isna(sig["next_date"]):
            continue
        entry_px = float(sig["close"])
        exit_px = float(sig["next_open"])
        size = per_trade_pct * cash
        shares = int(size / entry_px / 1000) * 1000
        if shares < 1000:
            continue
        cost = shares * entry_px
        if cost > cash:
            continue
        fee = cost * transaction_cost_pct
        cash -= cost + fee
        open_pos.append({
            "stock_id": str(sig["stock_id"]),
            "entry_date": today_d,
            "exit_date": sig["next_date"],
            "entry_price": entry_px,
            "exit_price": exit_px,
            "shares": shares,
            "entry_fee": fee,
            "daily_ret": float(sig.get("daily_ret", 0) or 0),
            "low_drop": float(sig.get("low_drop_pct", 0) or 0),
            "vol_surge": float(sig.get("vol_surge", 0) or 0),
            "amt_20": float(sig.get("amt_20", 0) or 0),
            "margin_chg": float(sig.get("margin_change_10d", 0) or 0),
        })

    # 3) 收尾：關所有還沒到期的部位
    for p in open_pos:
        close_position(p, None)
    equity = cash

    # 統計
    if not trades:
        print("  No trades")
        return {"summary": {"total_trades": 0}, "trades_log": []}

    rets = [t["realized_pnl_pct"] for t in trades]
    total_pnl = sum(t["realized_pnl"] for t in trades)
    cum_return = total_pnl / initial_capital
    win_rate = sum(1 for r in rets if r > 0) / len(rets)
    avg_ret = np.mean(rets)
    median_ret = np.median(rets)
    best = max(rets)
    worst = min(rets)
    # Sharpe approx（trade-level，年化）：乘 sqrt(每年交易次數)，而非 sqrt(252/每年交易次數)
    # （原式方向相反，交易越稀疏越高估）。
    _years = max(months / 12.0, 1e-9)
    _trades_per_year = len(rets) / _years
    sharpe_trade = avg_ret / max(np.std(rets), 1e-9) * np.sqrt(max(_trades_per_year, 1e-9))
    # 逐年
    yearly = defaultdict(list)
    for t in trades:
        y = t["entry_date"][:4]
        yearly[y].append(t["realized_pnl_pct"])
    yearly_summary = {y: float(np.sum(rs)) for y, rs in yearly.items()}

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Win Rate:     {win_rate*100:.1f}%")
    print(f"  Avg Return:   {avg_ret*100:+.2f}%  (median {median_ret*100:+.2f}%)")
    print(f"  Best/Worst:   {best*100:+.1f}% / {worst*100:+.1f}%")
    print(f"  Cum P&L:      ${total_pnl:+,.0f} ({cum_return*100:+.1f}%)")
    print(f"  Approx Sharpe: {sharpe_trade:.2f}")
    print(f"\n  Yearly sum of single-trade returns:")
    for y, r in sorted(yearly_summary.items()):
        n = len(yearly[y])
        print(f"    {y}: {r*100:+.2f}% (n={n})")

    summary = {
        "strategy": "limit_down_bounce",
        "total_trades": len(trades),
        "win_rate": win_rate,
        "avg_return_per_trade": float(avg_ret),
        "median_return_per_trade": float(median_ret),
        "best_trade": float(best),
        "worst_trade": float(worst),
        "cumulative_pnl_pct": float(cum_return),
        "yearly_returns": yearly_summary,
        "approx_sharpe": float(sharpe_trade),
        "backtest_start": str(start_date),
        "backtest_end": str(end_date),
        "config": {
            "per_trade_pct": per_trade_pct,
            **scan_kwargs,
        },
    }
    return {"summary": summary, "trades_log": trades}


def scan_today() -> None:
    """今日 scan 模式：列出今天符合條件的標的。"""
    print("=" * 60)
    print("跌停斷頭隔日反彈：今日 scan")
    print("=" * 60)
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    with get_session() as s:
        price_df = load_prices(s, start_date, end_date)
        margin_df = load_margin(s, start_date, end_date)
    g = compute_signals(price_df, margin_df)
    # 今日訊號（用最後一個 trading date）
    last_d = price_df["trading_date"].max()
    today_sig = g[g["trading_date"] == last_d]
    # 不需要 next_open（盤中 scan）
    matches = today_sig[
        (today_sig["two_limit_down"] == 1)
        & (today_sig["low_drop_pct"] <= -0.03)
        & (today_sig["vol_surge"] >= 1.3)
        & (today_sig["amt_20"] >= 1000)
    ].copy()
    print(f"\n{last_d} 候選 {len(matches)} 檔:")
    if len(matches) == 0:
        print("  無")
        return
    for _, r in matches.iterrows():
        print(f"  {r['stock_id']}  close={r['close']:.2f} (-{abs(r['daily_ret'])*100:.1f}%) "
              f"vol={r['vol_surge']:.1f}x  low_drop={r['low_drop_pct']*100:+.1f}% "
              f"margin_chg_10d={r.get('margin_change_10d', 0)*100:+.1f}%  amt_20={r['amt_20']:.0f}萬")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=36)
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--per-trade-pct", type=float, default=0.10)
    p.add_argument("--min-amt-20", type=float, default=1000.0,
                   help="20d 平均成交值門檻（萬元，預設 1000=1千萬/日）")
    p.add_argument("--min-low-drop", type=float, default=-0.04,
                   help="當日盤中最低跌幅 (open→low) 門檻（預設 -0.04 = -4%%）")
    p.add_argument("--min-vol-surge", type=float, default=1.3,
                   help="爆量倍率（vs 20d ma，預設 1.3）")
    p.add_argument("--min-bounce", type=float, default=0.01,
                   help="收盤從最低點反彈幅度（預設 0.01 = +1%%）")
    p.add_argument("--min-margin-chg", type=float, default=0.15,
                   help="融資 10d 累積增加比例（預設 0.15 = +15%%）")
    p.add_argument("--scan", action="store_true",
                   help="今日 scan 模式（不跑 backtest）")
    p.add_argument("--hold-days", type=int, default=1,
                   help="持有天數（1=T+1 open；5=T+5 close）。診斷顯示 T+5 有強 alpha")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    if args.scan:
        scan_today()
        return

    result = run_backtest(
        months=args.months,
        initial_capital=args.capital,
        per_trade_pct=args.per_trade_pct,
        hold_days=args.hold_days,
        min_amt_20=args.min_amt_20,
        min_low_drop=args.min_low_drop,
        min_vol_surge=args.min_vol_surge,
        min_bounce=args.min_bounce,
        min_margin_change_10d=args.min_margin_chg,
    )

    out = args.output or str(
        ROOT / "artifacts" / "d_replay" / f"limit_down_bounce_{args.months}mo.json"
    )
    with open(out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
