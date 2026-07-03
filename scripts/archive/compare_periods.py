#!/usr/bin/env python3
"""Compare F+ vs current run period-by-period to find divergence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def load_json(name):
    path = ROOT / "artifacts" / "backtest" / name
    with open(path) as f:
        return json.load(f)

fp = load_json("exp_f_breakthrough.json")
cur = load_json("breakthrough_no_lookback.json")

fp_periods = fp.get("periods", [])
cur_periods = cur.get("periods", [])

print(f"F+  periods: {len(fp_periods)}")
print(f"Cur periods: {len(cur_periods)}")
print()

print(f"{'Date':<12} {'F+_ret':>9} {'Cur_ret':>9} {'FP_eq':>10} {'Cur_eq':>10} {'FP_trades':>10} {'Cur_trades':>10}")
print("-" * 70)

def get_ret(p):
    """Try multiple key names for period return."""
    for k in ("return", "period_return", "ret"):
        v = p.get(k)
        if v is not None:
            return float(v)
    return 0.0

def get_trades(p):
    """Try multiple key names for trade count."""
    for k in ("trades", "num_trades", "n_trades"):
        v = p.get(k)
        if v is not None:
            if isinstance(v, list):
                return len(v)
            return v
    return "?"

fp_equity = 10000.0
cur_equity = 10000.0

for i, fp_p in enumerate(fp_periods):  # All periods
    fd = str(fp_p.get("rebalance_date", fp_p.get("start_date", "?")))
    fr = get_ret(fp_p)
    fp_equity *= (1 + fr)
    fp_trades = get_trades(fp_p)

    # Find matching current period
    cur_p = None
    for cp in cur_periods:
        cd = str(cp.get("rebalance_date", cp.get("start_date", "?")))
        if cd == fd:
            cur_p = cp
            break

    if cur_p:
        cr = get_ret(cur_p)
        cur_equity *= (1 + cr)
        ct = get_trades(cur_p)
        diff = fr - cr
        flag = " ◄◄◄" if abs(diff) > 0.05 else ""
        print(f"{fd:<12} {fr:>9.4f} {cr:>9.4f} {fp_equity:>10.0f} {cur_equity:>10.0f} {str(fp_trades):>10} {str(ct):>10}{flag}")
    else:
        print(f"{fd:<12} {fr:>9.4f}   NOMATCH   {fp_equity:>10.0f}   NOMATCH")

print()
print("Key metrics:")
fp_sum = fp.get("summary", {})
cur_sum = cur.get("summary", {})
for k in ["total_return", "sharpe", "max_drawdown", "calmar", "win_rate"]:
    fv = fp_sum.get(k, "N/A")
    cv = cur_sum.get(k, "N/A")
    print(f"  {k}: F+={fv}, Cur={cv}")

# Check specific period 2022-07-01
print("\n2022-07 period comparison:")
for ps, name in [(fp_periods, "F+"), (cur_periods, "Cur")]:
    for p in ps:
        date_str = str(p.get("rebalance_date", p.get("start_date", "")))
        if "2022-07" in date_str or "2022-06" in date_str:
            pret = get_ret(p)
            trades = p.get("trades_log", p.get("trades", []))
            if isinstance(trades, dict):
                # trades may be a dict {stock_id: ret} in some formats
                print(f"  {name} {date_str}: return={pret:.4f}, trades_dict={list(trades.keys())[:5]}")
            else:
                print(f"  {name} {date_str}: return={pret:.4f}, trades={len(trades) if isinstance(trades, list) else 'N/A'}")
                if isinstance(trades, list):
                    for t in trades[:5]:
                        sid = t.get("stock_id", "?")
                        score = t.get("score", "?")
                        ret = t.get("realized_pnl_pct", t.get("return", t.get("ret", "?")))
                        entry = t.get("entry_date", "?")
                        print(f"    {sid}: score={score}, ret={ret}, entry={entry}")

# Show the stock_returns for 2022-07 if available
print("\n2022-07 stock_returns comparison:")
for ps, name in [(fp_periods, "F+"), (cur_periods, "Cur")]:
    for p in ps:
        date_str = str(p.get("rebalance_date", p.get("start_date", "")))
        if "2022-07" in date_str:
            sr = p.get("stock_returns", {})
            print(f"  {name} {date_str}: stock_returns={dict(list(sr.items())[:10]) if sr else 'N/A'}")
            # Find 2364 specifically
            v2364 = sr.get("2364")
            print(f"  {name} 2364 return: {v2364}")
