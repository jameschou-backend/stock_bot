"""Causal replay of fixed, retrospective event baskets; never a live signal source."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

POLICIES = {'hold': '持有半年', 'risk_exit': '半年內可提前退出'}
COMMISSION = .001425


def replay_event(close, can_trade, members, source_date, *, policy='hold',
                 horizon=126, slippage=.003):
    """Date-only releases enter strictly later; close signals fill next session.

    0050 is an explicit ETF benchmark exception to ordinary-stock membership.
    Fractional adjusted units are research accounting units, not executable shares.
    """
    if (policy not in POLICIES or horizon < 1 or not 0 <= slippage < 1
            or not members or len(set(members)) != len(members)
            or any(not re.fullmatch(r'[0-9]{4}', s) for s in members)):
        raise ValueError('Invalid replay policy, horizon, costs or members')
    if (not close.index.is_monotonic_increasing or close.index.has_duplicates
            or not close.index.equals(can_trade.index)
            or not close.columns.equals(can_trade.columns)):
        raise ValueError('Replay matrices must align on unique ordered sessions')
    prices = close[members]
    if ((prices <= 0) | np.isinf(prices)).any().any():
        raise ValueError('Invalid adjusted price')
    first = int(close.index.searchsorted(pd.Timestamp(source_date).normalize(), side='right'))
    last = first + horizon
    if first < 1 or last >= len(close):
        raise ValueError('Insufficient history for full fixed event window')
    days = close.index
    weight = 1.0 / len(members)
    units = {sid: 0.0 for sid in members}
    cash_by_stock = {sid: weight for sid in members}
    entry, peak, mark, pending = {}, {}, {}, {}
    trades, signals, blocked, anomalies = [], [], [], []
    missing = 0
    costs = turnover = 0.0
    curve = [{'date': str(days[first-1].date()), 'equity': 1.0, 'cash': 1.0}]

    for i in range(first, last+1):
        day = str(days[i].date())
        for sid in members:
            price = prices.iloc[i][sid]
            observed = bool(pd.notna(price))
            tradable = bool(observed and pd.notna(can_trade.iloc[i][sid]) and can_trade.iloc[i][sid])
            if units[sid] > 0:
                if not observed:
                    missing += 1
                previous = prices.iloc[i-1][sid]
                if observed and pd.notna(previous) and abs(price / previous - 1) > .5:
                    anomalies.append({'date': day, 'stock_id': sid,
                                      'adjusted_return': float(price / previous - 1)})
            if observed:
                mark[sid] = float(price)
            if i == first:
                if not tradable:
                    blocked.append({'date': day, 'stock_id': sid, 'side': 'buy', 'reason': 'entry_untradable'})
                    continue
                fill = float(price) * (1+slippage)
                qty = weight / (fill * (1+COMMISSION))
                fee = qty * fill * COMMISSION
                slip_cost = qty * float(price) * slippage
                units[sid], cash_by_stock[sid] = qty, 0.0
                entry[sid] = peak[sid] = float(price)
                costs += fee + slip_cost
                turnover += qty * float(price)
                trades.append({'date': day, 'stock_id': sid, 'side': 'buy',
                               'signal_date': str(pd.Timestamp(source_date).date()), 'reason': 'event',
                               'units': qty, 'reference_price': float(price), 'fill_price': fill,
                               'fee': fee, 'tax': 0.0, 'slippage_cost': slip_cost})
            if units[sid] <= 0:
                continue
            # A pending exit persists after a blocked fill, even if price recovers.
            if sid in pending or i == last:
                order = pending.get(sid, {'signal_date': str(days[first-1].date()), 'reason': 'scheduled_horizon'})
                if not tradable:
                    blocked.append({'date': day, 'stock_id': sid, 'side': 'sell', 'reason': order['reason']})
                    continue
                qty = units[sid]
                fill = float(price) * (1-slippage)
                fee = qty * fill * COMMISSION
                tax = qty * fill * (.001 if sid == '0050' else .003)
                slip_cost = qty * float(price) * slippage
                cash_by_stock[sid] += qty * fill - fee - tax
                costs += fee + tax + slip_cost
                turnover += qty * float(price)
                trades.append({'date': day, 'stock_id': sid, 'side': 'sell', **order,
                               'units': qty, 'reference_price': float(price), 'fill_price': fill,
                               'fee': fee, 'tax': tax, 'slippage_cost': slip_cost})
                units[sid] = 0.0
                pending.pop(sid, None)
                continue
            if policy == 'risk_exit' and observed:
                peak[sid] = max(peak[sid], float(price))
                reason = ('entry_stop' if price <= .85*entry[sid]
                          else 'trailing_stop' if price <= .8*peak[sid] else None)
                if reason:
                    pending[sid] = {'signal_date': day, 'reason': reason}
                    signals.append({'stock_id': sid, **pending[sid]})
        cash = sum(cash_by_stock.values())
        equity = cash + sum(q * mark[s] for s, q in units.items() if q > 0)
        curve.append({'date': day, 'equity': equity, 'cash': cash})

    nav = np.array([r['equity'] for r in curve])
    stock_results = []
    for sid in members:
        value = cash_by_stock[sid] + (units[sid]*mark[sid] if units[sid] > 0 else 0)
        stock_results.append({'stock_id': sid, 'initial_weight': weight, 'final_value': value,
                              'pnl_contribution': value-weight, 'allocated_return': value/weight-1,
                              'entered': sid in entry, 'unliquidated': units[sid] > 0})
    summary = {'start': str(days[first].date()), 'end': str(days[last].date()),
               'trading_intervals': horizon, 'total_return': float(nav[-1]-1),
               'max_drawdown': float((nav/np.maximum.accumulate(nav)-1).min()),
               'average_cash_fraction': float(np.mean([r['cash']/r['equity'] for r in curve[1:]])),
               'cost_per_initial_capital': costs, 'two_way_turnover': turnover,
               'blocked_entries': sum(r['side'] == 'buy' for r in blocked),
               'blocked_exit_days': sum(r['side'] == 'sell' for r in blocked),
               'held_missing_price_days': missing,
               'unliquidated_positions': sum(q > 0 for q in units.values())}
    return {'summary': summary, 'curve': curve, 'trades': trades, 'signals': signals,
            'blocked': blocked, 'large_move_exposures': anomalies, 'per_stock': stock_results}
