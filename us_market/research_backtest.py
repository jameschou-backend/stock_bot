"""Isolated daily-close research: lagged entries, cash accounting and missing fills."""
import math

import numpy as np

from .research_data import protocol
from .research_signals import ranked_indices


def performance(equity, initial, dates):
    values = np.r_[initial, np.asarray(equity, dtype=float)]
    daily = values[1:] / values[:-1] - 1
    years = len(equity) / 252
    volatility = daily.std(ddof=1) if len(daily) > 1 else 0
    return {
        "net_return": float(values[-1] / initial - 1),
        "cagr": float((values[-1] / initial) ** (1 / years) - 1) if years else 0,
        "max_drawdown": float(np.min(values / np.maximum.accumulate(values) - 1)),
        "sharpe_rf0": float(daily.mean() / volatility * math.sqrt(252)) if volatility > 0 else 0,
        "first_session": dates[0], "last_session": dates[-1], "sessions": len(equity),
    }


def simulate(panel, feat, arm, first, last, cost_bps, spec=None, *, cash_actions=None):
    spec = spec or protocol()
    dates, symbols = panel.dates, panel.symbols
    period = np.flatnonzero((dates >= first) & (dates <= last))
    if not len(period):
        raise ValueError("Backtest period has no market sessions")
    opening, close = (panel.fields[k].to_numpy() for k in ("open", "close"))
    signals = feat["signals"][arm].to_numpy()
    relative, adv, lows = (feat[k].to_numpy() for k in ("rs", "adv", "exit_low"))
    rate = cost_bps / 10000
    cash = nav = float(spec["initial_cash_usd"])
    positions, pending = {}, {}
    history, fills, closed, settlements = [], [], [], []
    skipped_missing_open = stale_position_days = 0
    for i in period:
        # Reserve from information and cash known at the preceding close.
        prior_cash, target = cash, nav / spec["slots"]
        for j, event in (cash_actions or {}).get(i, {}).items():
            if j not in positions:
                continue
            position = positions.pop(j)
            amount = position["units"] * event["adjusted_cash_per_unit"]
            cash += amount
            pending.pop(j, None)
            settlements.append({"ticker": symbols[j], "date": dates[i], "cash_usd": amount,
                                "rate": event["rate"], "unit_factor": event["unit_factor"],
                                "event_ids": ",".join(event["ids"])})
            closed.append({"ticker": symbols[j], "entry_date": dates[position["entry_i"]],
                           "exit_date": dates[i], "entry_signal_date": dates[position["signal_i"]],
                           "exit_signal_date": "", "reason": "cash_merger",
                           "entry_price": position["entry_price"], "exit_price": event["adjusted_cash_per_unit"],
                           "net_return": amount / position["cash_paid"] - 1,
                           "net_pnl": amount - position["cash_paid"], "held_sessions": i - position["entry_i"]})
        for j in list(pending):
            price = opening[i, j]
            if not np.isfinite(price):
                continue
            position = positions.pop(j)
            notional = position["units"] * price
            fee = notional * rate
            cash += notional - fee
            trade = {"ticker": symbols[j], "entry_date": dates[position["entry_i"]],
                     "exit_date": dates[i], "entry_signal_date": dates[position["signal_i"]],
                     "exit_signal_date": dates[pending[j][1]], "reason": pending[j][0],
                     "entry_price": position["entry_price"], "exit_price": price,
                     "net_return": (notional - fee) / position["cash_paid"] - 1,
                     "net_pnl": notional - fee - position["cash_paid"], "held_sessions": i - position["entry_i"]}
            closed.append(trade)
            fills.append({"date": dates[i], "ticker": symbols[j], "side": "sell", "notional": notional, "fee": fee})
            del pending[j]
        if i > period[0]:
            candidates = ranked_indices(signals[i - 1], relative[i - 1], adv[i - 1], symbols)
            for j in candidates:
                if len(positions) >= spec["slots"]:
                    break
                if j in positions:
                    continue
                price = opening[i, j]
                if not np.isfinite(price):
                    skipped_missing_open += 1
                    continue
                amount = min(target, prior_cash / (1 + rate), adv[i - 1, j] * spec["max_adv_participation"])
                if amount <= 1e-8:
                    break
                fee = amount * rate
                cash -= amount + fee
                prior_cash -= amount + fee
                positions[j] = {"units": amount / price, "entry_price": price, "entry_i": i,
                                "signal_i": i - 1, "cash_paid": amount + fee, "mark": price, "last_i": i}
                fills.append({"date": dates[i], "ticker": symbols[j], "side": "buy", "notional": amount, "fee": fee})
        market_value, stale_value = 0.0, 0.0
        for j, position in positions.items():
            observed = close[i, j]
            if np.isfinite(observed):
                position["mark"], position["last_i"] = observed, i
            else:
                stale_position_days += 1
                stale_value += position["units"] * position["mark"]
            market_value += position["units"] * position["mark"]
            if j in pending:
                continue
            reason = None
            if np.isfinite(observed) and observed < position["entry_price"] * (1 - spec["close_stop_fraction"]):
                reason = "close_stop"
            elif np.isfinite(observed) and observed < lows[i, j]:
                reason = "prior_low_break"
            elif i - position["entry_i"] + 1 >= spec["exit_horizon_sessions"]:
                reason = "time_exit"
            if reason:
                pending[j] = (reason, i)
        nav = cash + market_value
        if cash < -1e-6 or not np.isfinite(nav) or nav <= 0:
            raise ValueError("Invalid portfolio accounting")
        history.append({"date": dates[i], "equity": nav, "cash": cash, "exposure": market_value / nav,
                        "positions": len(positions), "stale_value": stale_value, "pending_exits": len(pending)})
    metrics = performance([r["equity"] for r in history], spec["initial_cash_usd"], dates[period])
    metrics.update({
        "arm": arm, "cost_bps_each_side": cost_bps, "closed_trades": len(closed),
        "closed_trade_win_rate": float(np.mean([r["net_pnl"] > 0 for r in closed])) if closed else None,
        "buy_fills": sum(r["side"] == "buy" for r in fills), "fees_usd": sum(r["fee"] for r in fills),
        "annual_one_way_turnover": sum(r["notional"] for r in fills) / 2 / np.mean([r["equity"] for r in history]) / (len(period) / 252),
        "mean_exposure": float(np.mean([r["exposure"] for r in history])),
        "skipped_missing_open": skipped_missing_open, "stale_position_days": stale_position_days,
        "final_pending_exits": len(pending), "final_open_positions": len(positions),
        "final_stale_value": history[-1]["stale_value"],
        "cash_merger_settlements": len(settlements),
        "final_return_stale_marked_zero": (nav - history[-1]["stale_value"]) / spec["initial_cash_usd"] - 1,
        "live_qualified": False,
    })
    holdings = [{"ticker": symbols[j], "last_price_date": dates[p["last_i"]], "entry_date": dates[p["entry_i"]],
                 "value": p["units"] * p["mark"], "pending_exit": j in pending} for j, p in positions.items()]
    return {"metrics": metrics, "equity": history, "fills": fills, "trades": closed, "holdings": holdings, "settlements": settlements}


def benchmark(panel, first, last, cost_bps, spec=None):
    spec = spec or protocol()
    period = np.flatnonzero((panel.dates >= first) & (panel.dates <= last))
    prices = panel.fields["close"][spec["benchmark"]].iloc[period]
    entry = panel.fields["open"][spec["benchmark"]].iloc[period[0]]
    equity = spec["initial_cash_usd"] / (1 + cost_bps / 10000) / entry * prices
    return performance(equity, spec["initial_cash_usd"], panel.dates[period])


def blackhorse_labels(panel, feat, first, last, spec=None):
    """Future values are only consumed here; never used by features or simulation."""
    spec = spec or protocol()
    opening, close = (panel.fields[k].to_numpy() for k in ("open", "close"))
    valid = np.isfinite(close)
    gaps = np.vstack([np.zeros((1, len(panel.symbols)), dtype=int), np.cumsum(~valid, axis=0)])
    b = panel.symbols.index(spec["benchmark"])
    rs, adv = (feat[k].to_numpy() for k in ("rs", "adv"))
    masks = {"eligible_pool": feat["eligible"].to_numpy(), **{k: v.to_numpy() for k, v in feat["signals"].items()}}
    stats = {name: {"candidates": 0, "evaluated": 0, "hits": 0, "unknown": 0, "immature": 0} for name in masks}
    rate = spec["base_cost_bps_each_side"] / 10000
    for i, day in enumerate(panel.dates):
        if not first <= day <= last:
            continue
        entry_i, end_i = i + 1, i + spec["blackhorse_horizon_sessions"]
        for arm, mask in masks.items():
            indices = ranked_indices(mask[i], rs[i], adv[i], panel.symbols)
            if arm != "eligible_pool":
                indices = indices[:spec["slots"]]
            row = stats[arm]
            row["candidates"] += len(indices)
            if end_i >= len(panel.dates) or panel.dates[end_i] > last:
                row["immature"] += len(indices)
                continue
            base = close[end_i, b] / opening[entry_i, b] * (1 - rate) / (1 + rate) - 1
            for j in indices:
                if not np.isfinite(opening[entry_i, j]) or gaps[end_i + 1, j] != gaps[entry_i, j]:
                    row["unknown"] += 1
                    continue
                result = close[end_i, j] / opening[entry_i, j] * (1 - rate) / (1 + rate) - 1
                row["evaluated"] += 1
                row["hits"] += int(result >= spec["blackhorse_net_return"] and result - base >= spec["blackhorse_excess_return"])
    for row in stats.values():
        row["hit_rate"] = row["hits"] / row["evaluated"] if row["evaluated"] else None
    return stats
