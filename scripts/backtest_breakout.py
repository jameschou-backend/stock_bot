#!/usr/bin/env python
"""#10 200d 新高動能 + 不對稱出場 backtest（Taleb 凸性策略）

設計哲學：
  - Event-driven（不是月頻 calendar rebal）
  - 進場：close ≥ 200d_high AND volume > 1.5 × vol_ma60 AND amt_20 ≥ 30M
  - 輸家快砍：5 天內 close < entry×0.97 OR close < 50MA → 立刻砍
  - 贏家慢跑：chandelier exit（max(high, 22) - 3×ATR(22)）
  - Risk-based sizing：每筆 max_loss ≤ 1% 總資金（fat tail 凸性）
  - 同時持倉上限 12 檔

不使用 ML model（純規則）。
不限產業 / 市值（市場告訴你哪裡有 alpha）。

用法：
  python scripts/backtest_breakout.py --months 36
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
# Position
# ──────────────────────────────────────────────────────────────
class Position:
    __slots__ = ("stock_id", "entry_date", "entry_price", "shares",
                 "hard_stop", "peak_high", "days_held", "amt_20")

    def __init__(self, sid, ed, ep, sh, hs, amt):
        self.stock_id = sid
        self.entry_date = ed
        self.entry_price = ep
        self.shares = sh
        self.hard_stop = hs       # 50MA 或 -3% 取高者（快砍底線）
        self.peak_high = ep       # 持倉期間 high
        self.days_held = 0
        self.amt_20 = amt


# ──────────────────────────────────────────────────────────────
# 載入 + 預計算
# ──────────────────────────────────────────────────────────────
def load_prices(session, start: date, end: date) -> pd.DataFrame:
    """OHLCV + 預計算 200d_high / 50MA / ATR22 / vol_ma60 / amt_20。"""
    t0 = time.time()
    # 多撈 250 天提前期（200d rolling 需要）
    pre_start = start - timedelta(days=400)
    sql = text("""
        SELECT stock_id, trading_date, open, high, low, close, volume
        FROM raw_prices
        WHERE trading_date >= :s AND trading_date <= :e
          AND stock_id REGEXP '^[0-9]{4}$'
        ORDER BY stock_id, trading_date
    """)
    df = pd.read_sql(sql, session.bind, params={"s": pre_start, "e": end})
    print(f"  Prices: {len(df):,} rows ({time.time()-t0:.1f}s)")

    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])

    # 計算 features 每股
    t0 = time.time()
    parts = []
    for sid, g in df.groupby("stock_id"):
        g = g.sort_values("trading_date").reset_index(drop=True)
        g["high_200"] = g["high"].rolling(200, min_periods=100).max().shift(1)
        g["ma_50"] = g["close"].rolling(50, min_periods=25).mean()
        g["ma_high_22"] = g["high"].rolling(22, min_periods=10).max()
        # ATR(22)
        tr = pd.concat([
            (g["high"] - g["low"]).abs(),
            (g["high"] - g["close"].shift(1)).abs(),
            (g["low"] - g["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        g["atr_22"] = tr.rolling(22, min_periods=10).mean()
        # Volume MA & amt
        g["vol_ma_60"] = g["volume"].rolling(60, min_periods=30).mean()
        # 20d 平均成交值（萬元）
        g["amt_20"] = (g["close"] * g["volume"]).rolling(20, min_periods=10).mean() / 1e4
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    print(f"  Features computed ({time.time()-t0:.1f}s)")

    return df


# ──────────────────────────────────────────────────────────────
# Backtest 主迴圈
# ──────────────────────────────────────────────────────────────
def run_breakout(
    months: int = 36,
    max_concurrent: int = 12,
    max_loss_pct: float = 0.01,
    initial_capital: float = 1_000_000.0,
    transaction_cost_pct: float = 0.003,
    min_amt_20: float = 3000.0,          # 萬元（3000 萬）
    vol_surge_mult: float = 1.5,
    quick_fail_days: int = 5,
    quick_fail_pct: float = -0.03,
    chandelier_atr_mult: float = 3.0,
) -> Dict:
    """主回測。"""
    print("\n" + "=" * 60)
    print("#10 200d 新高動能 + 不對稱出場（Taleb 凸性）")
    print("=" * 60)
    print(f"  Max concurrent positions: {max_concurrent}")
    print(f"  Per-trade max loss: {max_loss_pct:.1%} of equity")
    print(f"  Initial capital: ${initial_capital:,.0f}")
    print(f"  Quick fail: {quick_fail_days}d / {quick_fail_pct:.0%}")
    print(f"  Chandelier: high(22) - {chandelier_atr_mult}×ATR(22)")
    print(f"  Min amt_20: {min_amt_20:,.0f} 萬")

    config = load_config()
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=months * 30)

    with get_session() as session:
        df = load_prices(session, start_date, end_date)

    # 全市場日報酬（benchmark）：pivot 成 date×stock 後沿「時間軸」pct_change 再跨股取均。
    # 原寫法 groupby(date) 內 pct_change 算的是「同日相鄰股票價差比」（無意義）。
    _piv = df.pivot_table(index="trading_date", columns="stock_id", values="close").sort_index()
    bm = _piv.pct_change().mean(axis=1).fillna(0)
    bm_cum = (1 + bm).cumprod()

    # 主回測迴圈
    all_dates = sorted(df["trading_date"].unique())
    bt_dates = [d for d in all_dates if d >= start_date]
    print(f"  Trading days: {len(bt_dates)}")

    positions: Dict[str, Position] = {}
    equity = initial_capital
    cash = initial_capital
    trades_log = []
    equity_curve = []
    exit_reasons = defaultdict(int)

    t0 = time.time()
    for di, today in enumerate(bt_dates):
        today_df = df[df["trading_date"] == today]
        if today_df.empty:
            equity_curve.append({"date": str(today), "equity": equity})
            continue

        price_map = {str(r["stock_id"]): float(r["close"]) for _, r in today_df.iterrows()}
        high_map = {str(r["stock_id"]): float(r["high"]) for _, r in today_df.iterrows()}
        feat_map = {str(r["stock_id"]): r.to_dict() for _, r in today_df.iterrows()}

        # ── EXIT logic ──
        for sid in list(positions.keys()):
            pos = positions[sid]
            pos.days_held += 1
            close = price_map.get(sid)
            high = high_map.get(sid, close)
            if close is None:
                continue
            pos.peak_high = max(pos.peak_high, high)

            feat = feat_map.get(sid, {})
            ma50 = float(feat.get("ma_50") or close)
            atr22 = float(feat.get("atr_22") or 0)
            ma_high_22 = float(feat.get("ma_high_22") or pos.peak_high)

            exit_reason = None
            # 1) Quick fail（前 5 天）：close < entry × (1+quick_fail_pct) 或 close < 50MA
            if pos.days_held <= quick_fail_days:
                if close < pos.entry_price * (1 + quick_fail_pct):
                    exit_reason = "Quick Fail (-3%)"
                elif ma50 > 0 and close < ma50:
                    exit_reason = "Quick Fail (50MA)"
            # 2) Trail mode（day 6+）：chandelier exit
            else:
                chandelier = ma_high_22 - chandelier_atr_mult * atr22
                if close < chandelier:
                    exit_reason = "Chandelier"
                elif ma50 > 0 and close < ma50:
                    exit_reason = "50MA Break"

            # 3) Hard stop（任何時候）
            if not exit_reason and close < pos.hard_stop:
                exit_reason = "Hard Stop"

            if exit_reason:
                exit_px = close
                cost = transaction_cost_pct  # 進+出共 0.3%×2 算進整體；簡化
                gross = pos.shares * (exit_px - pos.entry_price)
                fee = pos.shares * exit_px * cost
                net_pnl = gross - fee
                ret = net_pnl / (pos.shares * pos.entry_price)
                cash += pos.shares * exit_px - fee
                trades_log.append({
                    "stock_id": sid,
                    "entry_date": str(pos.entry_date),
                    "exit_date": str(today),
                    "entry_price": pos.entry_price,
                    "exit_price": exit_px,
                    "shares": pos.shares,
                    "realized_pnl": net_pnl,
                    "realized_pnl_pct": ret,
                    "days_held": pos.days_held,
                    "exit_reason": exit_reason,
                    "peak_high": pos.peak_high,
                    "amt_20": pos.amt_20,
                })
                exit_reasons[exit_reason] += 1
                del positions[sid]

        # ── ENTRY logic ──
        if len(positions) < max_concurrent:
            # 找今日所有 breakouts
            tdf = today_df.copy()
            tdf["close_num"] = pd.to_numeric(tdf["close"], errors="coerce")
            tdf["high_200_num"] = pd.to_numeric(tdf["high_200"], errors="coerce")
            tdf["vol_ma_60_num"] = pd.to_numeric(tdf["vol_ma_60"], errors="coerce")
            tdf["volume_num"] = pd.to_numeric(tdf["volume"], errors="coerce")
            tdf["amt_20_num"] = pd.to_numeric(tdf["amt_20"], errors="coerce")
            tdf["ma_50_num"] = pd.to_numeric(tdf["ma_50"], errors="coerce")

            breakouts = tdf[
                (tdf["close_num"] >= tdf["high_200_num"])
                & (tdf["volume_num"] >= vol_surge_mult * tdf["vol_ma_60_num"])
                & (tdf["amt_20_num"] >= min_amt_20)
                & (tdf["ma_50_num"] > 0)
                & (tdf["close_num"] > tdf["ma_50_num"])  # close > 50MA（趨勢確認）
            ].copy()

            # 排除已持倉
            breakouts = breakouts[~breakouts["stock_id"].astype(str).isin(positions.keys())]

            # 按 amt_20 排序（流動性大的優先）— 也可改 score 排序若有 model
            breakouts = breakouts.sort_values("amt_20_num", ascending=False)

            slots = max_concurrent - len(positions)
            for _, row in breakouts.head(slots).iterrows():
                sid = str(row["stock_id"])
                entry_price = float(row["close_num"])
                ma50 = float(row["ma_50_num"])
                # Hard stop：max(entry * 0.95, 50MA × 0.98)
                hard_stop = max(entry_price * 0.95, ma50 * 0.98)
                if hard_stop >= entry_price:
                    continue  # MA50 已在 entry 上方，無 stop 空間
                # Risk-based sizing
                risk_per_share = entry_price - hard_stop
                max_loss = equity * max_loss_pct
                shares = int(max_loss / risk_per_share / 1000) * 1000  # 整張
                if shares < 1000:
                    continue
                cost_amount = shares * entry_price
                if cost_amount > cash * 0.30:  # 單筆不超過現金 30%
                    shares = int(cash * 0.30 / entry_price / 1000) * 1000
                    if shares < 1000:
                        continue
                    cost_amount = shares * entry_price
                fee = cost_amount * transaction_cost_pct
                cash -= cost_amount + fee
                positions[sid] = Position(
                    sid, today, entry_price, shares, hard_stop,
                    float(row.get("amt_20_num") or 0),
                )

        # ── 更新 equity（持倉以收盤計）──
        position_value = sum(pos.shares * price_map.get(sid, pos.entry_price)
                             for sid, pos in positions.items())
        equity = cash + position_value
        equity_curve.append({"date": str(today), "equity": equity})

        # 進度
        if (di + 1) % max(1, len(bt_dates) // 20) == 0:
            pct = (di + 1) / len(bt_dates) * 100
            print(f"  [{today}] {pct:.0f}% | eq: {equity:,.0f} | pos: {len(positions)} | trades: {len(trades_log)} | {time.time()-t0:.0f}s",
                  flush=True)

    # ── 計算結果 ──
    eq = pd.DataFrame(equity_curve)
    eq["date"] = pd.to_datetime(eq["date"])
    eq["equity"] = eq["equity"].astype(float)
    eq["peak"] = eq["equity"].cummax()
    eq["dd"] = eq["equity"] / eq["peak"] - 1
    total_return = eq["equity"].iloc[-1] / initial_capital - 1
    mdd = eq["dd"].min()
    days = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days
    annualized = (1 + total_return) ** (365 / max(days, 1)) - 1
    daily_ret = eq["equity"].pct_change().fillna(0)
    sharpe = (daily_ret.mean() / max(daily_ret.std(), 1e-9)) * np.sqrt(252)
    calmar = annualized / abs(mdd) if mdd != 0 else 0
    rets = [t["realized_pnl_pct"] for t in trades_log]
    win_rate = sum(1 for r in rets if r > 0) / max(len(rets), 1)
    profit_factor = (
        sum(r for r in rets if r > 0) / abs(sum(r for r in rets if r < 0))
        if any(r < 0 for r in rets) else float("inf")
    )
    avg_hold = np.mean([t["days_held"] for t in trades_log]) if trades_log else 0

    # 逐年報酬
    eq["year"] = eq["date"].dt.year
    yearly = {}
    for y, ydf in eq.groupby("year"):
        if len(ydf) < 2:
            continue
        yr = ydf["equity"].iloc[-1] / ydf["equity"].iloc[0] - 1
        yearly[str(y)] = float(yr)

    print(f"\n{'='*60}")
    print(f"#10 Breakout Results")
    print(f"{'='*60}")
    print(f"  Period:        {bt_dates[0]} ~ {bt_dates[-1]} ({time.time()-t0:.0f}s)")
    print(f"  Total Return:  {total_return*100:+.2f}%")
    print(f"  Annualized:    {annualized*100:+.2f}%")
    print(f"  MDD:           {mdd*100:+.2f}%")
    print(f"  Sharpe:        {sharpe:.4f}")
    print(f"  Calmar:        {calmar:.4f}")
    print(f"  Win Rate:      {win_rate*100:.2f}%")
    print(f"  Profit Factor: {profit_factor:.3f}")
    print(f"  Total Trades:  {len(trades_log)}")
    print(f"  Avg Hold Days: {avg_hold:.1f}")
    print(f"\n  Exit Reasons:")
    for r, n in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        pct = n / max(len(trades_log), 1) * 100
        avg = np.mean([t["realized_pnl_pct"] for t in trades_log if t["exit_reason"] == r])
        print(f"    {r}: {n} ({pct:.1f}%) avg={avg*100:+.2f}%")
    print(f"\n  Yearly:")
    for y, r in sorted(yearly.items()):
        print(f"    {y}: {r*100:+.2f}%")

    summary = {
        "total_return": total_return,
        "annualized_return": annualized,
        "max_drawdown": mdd,
        "sharpe_ratio": float(sharpe),
        "calmar_ratio": float(calmar),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": len(trades_log),
        "avg_hold_days": float(avg_hold),
        "exit_reasons": dict(exit_reasons),
        "yearly_returns": yearly,
        "backtest_start": str(bt_dates[0]),
        "backtest_end": str(bt_dates[-1]),
        "config": {
            "strategy": "breakout_200d",
            "max_concurrent": max_concurrent,
            "max_loss_pct": max_loss_pct,
            "min_amt_20": min_amt_20,
            "vol_surge_mult": vol_surge_mult,
            "quick_fail_days": quick_fail_days,
            "chandelier_atr_mult": chandelier_atr_mult,
        },
    }
    return {"summary": summary, "trades_log": trades_log, "equity_curve": equity_curve}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=36)
    p.add_argument("--max-concurrent", type=int, default=12)
    p.add_argument("--max-loss-pct", type=float, default=0.01)
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--min-amt-20", type=float, default=3000.0,
                   help="20d 平均成交值門檻（萬元，預設 3000=3千萬/日）")
    p.add_argument("--vol-surge", type=float, default=1.5)
    p.add_argument("--quick-fail-days", type=int, default=5)
    p.add_argument("--quick-fail-pct", type=float, default=-0.03)
    p.add_argument("--chandelier-atr", type=float, default=3.0)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    result = run_breakout(
        months=args.months,
        max_concurrent=args.max_concurrent,
        max_loss_pct=args.max_loss_pct,
        initial_capital=args.capital,
        min_amt_20=args.min_amt_20,
        vol_surge_mult=args.vol_surge,
        quick_fail_days=args.quick_fail_days,
        quick_fail_pct=args.quick_fail_pct,
        chandelier_atr_mult=args.chandelier_atr,
    )

    out = args.output or str(
        ROOT / "artifacts" / "d_replay" / f"breakout_{args.months}mo.json"
    )
    with open(out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
