"""Offline, fully funded 0050 replacement experiments using adjusted-price units.

This module does not select signals, infer announcement times, fetch prices, or
place orders. The caller supplies the first permitted execution date. Adjusted
price units are accounting units, not executable share or lot quantities.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype


COMMISSION = .001425
SELL_TAX = {'0050': .001, '2330': .003}


def simulate_core(close: pd.DataFrame, flags: pd.DataFrame, entries: list[dict], *,
                  start='2023-01-03', end='2026-06-23', allocation=.30,
                  horizon=63, slippage=.003, mode='signals') -> dict:
    """Compare event-driven replacement with 0050 and a static two-asset mix.

    ``horizon`` counts trading-session intervals: entry at index i is due at
    i + horizon. A blocked conversion waits for both instruments to trade;
    overlapping events never extend a position. Entries on an unavailable date
    are rejected, not shifted to another session. Due exits run before entries.

    Each replacement removes at most ``allocation * pre_trade_nav`` of 0050
    market value. Its net sale proceeds fund the stock purchase including fees.
    Thus the stock's weight is slightly below the nominal allocation. All cash
    movements use additive commission, slippage and sell tax on mark notional.
    Slippage is charged as an explicit cost, not hidden in the adjusted price.

    The requested interval uses the supplied sessions within [start, end]. Both
    actual endpoints are returned. Every held instrument must have a valid,
    tradable final-session price; otherwise the function raises instead of
    inventing a liquidation. Missing intermediate prices use the last valid
    positive mark for valuation only, never for execution.
    """
    if mode not in {'signals', 'benchmark', 'static_mix'}:
        raise ValueError('mode must be signals, benchmark, or static_mix')
    if not math.isfinite(allocation) or not 0 <= allocation <= 1:
        raise ValueError('allocation must be finite and between zero and one')
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError('horizon must be a positive integer number of sessions')
    if not math.isfinite(slippage) or not 0 <= slippage < 1 - COMMISSION - max(SELL_TAX.values()):
        raise ValueError('slippage must leave positive net sale proceeds')
    if (not isinstance(close.index, pd.DatetimeIndex) or close.index.tz is not None
            or close.index.hasnans or not close.index.is_unique
            or not close.index.is_monotonic_increasing
            or not close.index.equals(close.index.normalize())
            or not close.columns.is_unique or set(close.columns) != {'0050', '2330'}
            or not close.index.equals(flags.index) or not close.columns.equals(flags.columns)
            or any(not is_bool_dtype(dtype) for dtype in flags.dtypes)):
        raise ValueError('Unique ordered date-only prices and aligned boolean flags for 0050/2330 are required')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start.tzinfo is not None or end.tzinfo is not None or end < start:
        raise ValueError('A valid timezone-naive start/end interval is required')
    window = close.loc[(close.index >= start) & (close.index <= end)].astype(float)
    if len(window) < 2:
        raise ValueError('At least two trading sessions are required in the requested interval')
    days = window.index
    observed = window.where(np.isfinite(window) & window.gt(0))
    # Keep earlier observations only to mark an existing position, never to fill.
    marks = close.astype(float).where(np.isfinite(close) & close.gt(0)).ffill().loc[days]
    tradable = flags.loc[days].fillna(False) & observed.notna()
    buy_rate = COMMISSION + slippage
    sell_rate = {sid: buy_rate + tax for sid, tax in SELL_TAX.items()}
    cash = 1.
    units = {'0050': 0., '2330': 0.}
    executions, trades, rejections, curve = [], [], [], []
    active = None
    cumulative_cost = cumulative_notional = 0.
    blocked_exit_sessions = 0
    previous_nav = 1.
    previous_marks = None
    previous_units = units.copy()
    seen_ids, scheduled = set(), {}

    for supplied in entries:
        event = dict(supplied)
        if not event.get('event_id') or 'entry_date' not in event or 'stock_id' not in event:
            raise ValueError('Every event requires event_id, entry_date, and stock_id')
        event_id = str(event['event_id'])
        if event_id in seen_ids:
            raise ValueError(f'Duplicate event_id: {event_id}')
        seen_ids.add(event_id)
        try:
            date = pd.Timestamp(event['entry_date'])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid entry_date for {event_id}') from exc
        if pd.isna(date) or date.tzinfo is not None:
            raise ValueError(f'Invalid timezone-naive entry_date for {event_id}')
        item = {'event_id': event_id, 'entry_date': date.isoformat(),
                'stock_id': str(event['stock_id'])}
        reason = ('mode_does_not_use_signals' if mode != 'signals'
                  else 'unsupported_stock' if item['stock_id'] != '2330'
                  else 'outside_window' if date < days[0] or date > days[-1]
                  else 'entry_date_not_trading_session' if date not in days else None)
        if reason:
            rejections.append({**item, 'reason': reason})
        else:
            scheduled.setdefault(date, []).append(item)

    def can_trade(day, *symbols):
        return all(bool(tradable.at[day, sid]) for sid in symbols)

    def nav(day):
        return float(cash + sum(quantity * marks.at[day, sid]
                                for sid, quantity in units.items() if quantity))

    def record_fill(day, sid, side, quantity, notional, cost, event_id, reason, cash_before):
        nonlocal cumulative_cost, cumulative_notional
        cumulative_cost += cost
        cumulative_notional += notional
        executions.append({'date': str(day.date()), 'event_id': event_id,
                           'stock_id': sid, 'side': side, 'units': float(quantity),
                           'price': float(observed.at[day, sid]),
                           'notional': float(notional), 'cost': float(cost),
                           'commission': float(notional * COMMISSION),
                           'slippage': float(notional * slippage),
                           'sell_tax': float(notional * SELL_TAX[sid] if side == 'sell' else 0.),
                           'cash_before': float(cash_before), 'cash_after': float(cash),
                           'reason': reason})

    def buy(day, sid, budget, event_id, reason):
        nonlocal cash
        if not can_trade(day, sid):
            raise ValueError(f'Cannot buy {sid} on {day.date()} without a tradable current price')
        if budget < 0 or budget > cash + 1e-12:
            raise ArithmeticError('Buy budget exceeds available cash')
        budget = min(float(budget), cash)
        if not budget:
            return {'notional': 0., 'cost': 0., 'units': 0.}
        notional = budget / (1 + buy_rate)
        # Round down rather than borrow a floating-point residual or add cash.
        while notional + notional * buy_rate > budget:
            notional = float(np.nextafter(notional, 0.))
        cost = notional * buy_rate
        quantity = notional / observed.at[day, sid]
        before = cash
        cash -= notional + cost
        units[sid] += quantity
        record_fill(day, sid, 'buy', quantity, notional, cost, event_id, reason, before)
        return {'notional': float(notional), 'cost': float(cost), 'units': float(quantity)}

    def sell(day, sid, quantity, event_id, reason):
        nonlocal cash
        if not can_trade(day, sid):
            raise ValueError(f'Cannot sell {sid} on {day.date()} without a tradable current price')
        if quantity < 0 or quantity > units[sid] + 1e-12:
            raise ArithmeticError('Sell quantity exceeds the funded holding')
        quantity = min(float(quantity), units[sid])
        notional = quantity * observed.at[day, sid]
        cost = notional * sell_rate[sid]
        before = cash
        units[sid] -= quantity
        cash += notional - cost
        record_fill(day, sid, 'sell', quantity, notional, cost, event_id, reason, before)
        return {'notional': float(notional), 'cost': float(cost), 'units': float(quantity)}

    def close_active(day, index, reason, terminal=False):
        nonlocal active
        entry = active
        before = nav(day)
        sold = sell(day, '2330', units['2330'], entry['event_id'], reason)
        repurchased = ({'notional': 0., 'cost': 0., 'units': 0.} if terminal else
                       buy(day, '0050', cash, entry['event_id'], 'return_to_benchmark'))
        stock_cost = entry['stock_buy_cost'] + sold['cost']
        conversion_cost = entry['benchmark_sell_cost'] + repurchased['cost']
        stock_pnl = sold['notional'] - entry['stock_entry_notional'] - stock_cost
        trades.append({**{k: v for k, v in entry.items() if k != 'entry_index'},
                       'exit_date': str(day.date()), 'exit_price': float(observed.at[day, '2330']),
                       'exit_nav': before, 'exit_nav_after': nav(day),
                       'stock_exit_notional': sold['notional'], 'stock_sell_cost': sold['cost'],
                       'stock_cost': float(stock_cost), 'benchmark_buy_cost': repurchased['cost'],
                       'benchmark_units_rebought': repurchased['units'],
                       'conversion_cost': float(conversion_cost),
                       'total_cost': float(stock_cost + conversion_cost),
                       'stock_net_pnl': float(stock_pnl),
                       'cycle_net_pnl': float(stock_pnl - conversion_cost),
                       'stock_net_return': float(stock_pnl / (entry['stock_entry_notional'] + entry['stock_buy_cost'])),
                       'holding_sessions': int(index - entry['entry_index']),
                       'exit_reason': reason})
        active = None

    for index, day in enumerate(days):
        day_cost_before = cumulative_cost
        terminal = index == len(days) - 1
        if index == 0:
            needed = ('0050', '2330') if mode == 'static_mix' and allocation > 0 else ('0050',)
            if not can_trade(day, *needed):
                raise ValueError(f'Initial portfolio cannot trade on {day.date()}')
            if mode == 'static_mix' and allocation > 0:
                before = nav(day)
                bought = buy(day, '2330', allocation, 'static_mix', 'initial_static_mix')
                active = {'event_id': 'static_mix', 'stock_id': '2330',
                          'entry_date': str(day.date()), 'entry_index': index,
                          'entry_nav': before, 'entry_price': float(observed.at[day, '2330']),
                          'stock_units': bought['units'], 'stock_entry_notional': bought['notional'],
                          'stock_buy_cost': bought['cost'], 'benchmark_sell_cost': 0.,
                          'benchmark_units_sold': 0., 'funding_budget': float(allocation)}
            buy(day, '0050', cash, None, 'initial_benchmark')
            if active:
                active['entry_nav_after'] = nav(day)
                active['realized_entry_weight'] = float(units['2330'] * marks.at[day, '2330'] / nav(day))

        if terminal:
            held = [sid for sid in units if units[sid] > 0]
            if not can_trade(day, *held):
                raise ValueError(f'Cannot liquidate all holdings on final session {day.date()}; no final NAV is valid')
            if active:
                close_active(day, index, 'terminal_liquidation', terminal=True)
            if units['0050'] > 0:
                sell(day, '0050', units['0050'], None, 'terminal_liquidation')
        elif mode == 'signals' and active and index >= active['entry_index'] + horizon:
            if can_trade(day, '2330', '0050'):
                close_active(day, index, 'scheduled_exit')
            else:
                blocked_exit_sessions += 1

        for event in scheduled.get(day, []):
            reason = ('terminal_session' if terminal else 'overlapping_position' if active
                      else 'zero_allocation' if allocation == 0
                      else 'entry_instruments_not_tradable' if not can_trade(day, '2330', '0050') else None)
            if reason:
                rejections.append({**event, 'reason': reason})
                continue
            before = nav(day)
            funding = min(allocation * before, units['0050'] * observed.at[day, '0050'])
            sold = sell(day, '0050', funding / observed.at[day, '0050'], event['event_id'], 'fund_active_position')
            # Only this sale's net proceeds fund the active allocation. Existing
            # rounding dust remains cash and is never credited as free funding.
            bought = buy(day, '2330', sold['notional'] - sold['cost'], event['event_id'], 'event_entry')
            active = {'event_id': event['event_id'], 'stock_id': '2330',
                      'entry_date': str(day.date()), 'entry_index': index,
                      'entry_nav': before, 'entry_nav_after': nav(day),
                      'entry_price': float(observed.at[day, '2330']), 'stock_units': bought['units'],
                      'stock_entry_notional': bought['notional'], 'stock_buy_cost': bought['cost'],
                      'benchmark_sell_cost': sold['cost'], 'benchmark_units_sold': sold['units'],
                      'funding_budget': float(funding),
                      'realized_entry_weight': float(units['2330'] * marks.at[day, '2330'] / nav(day))}

        equity = nav(day)
        daily_cost = cumulative_cost - day_cost_before
        market_pnl = (sum(previous_units[sid] * (marks.at[day, sid] - previous_marks[sid])
                          for sid in units if previous_units[sid]) if previous_marks is not None else 0.)
        if cash < 0 or any(q < 0 for q in units.values()) or not math.isfinite(equity) or equity <= 0:
            raise ArithmeticError('Non-finite NAV, negative cash, or an unfunded position')
        if not math.isclose(equity, previous_nav + market_pnl - daily_cost, rel_tol=1e-10, abs_tol=1e-12):
            raise ArithmeticError('Daily cash/position/cost conservation failed')
        weight = units['2330'] * marks.at[day, '2330'] / equity if units['2330'] else 0.
        curve.append({'date': str(day.date()), 'nav': float(equity), 'active_weight': float(weight),
                      'cash': float(cash), 'benchmark_units': float(units['0050']),
                      'active_units': float(units['2330']), 'daily_cost': float(daily_cost),
                      'cumulative_cost': float(cumulative_cost), 'market_pnl': float(market_pnl)})
        previous_nav, previous_marks, previous_units = equity, marks.loc[day].to_dict(), units.copy()

    navs = np.array([1., *(row['nav'] for row in curve)])
    years = (days[-1] - days[0]).days / 365.25
    summary = {'total_return': float(navs[-1] - 1),
               'cagr': float(navs[-1] ** (1 / years) - 1),
               'max_drawdown': float(np.min(navs / np.maximum.accumulate(navs) - 1)),
               'total_cost': float(cumulative_cost), 'turnover': float(cumulative_notional),
               'trade_count': len(executions), 'active_trade_count': len(trades),
               'mean_active_weight': float(np.mean([row['active_weight'] for row in curve])),
               'blocked_exit_sessions': blocked_exit_sessions, 'rejected_event_count': len(rejections),
               'initial_nav': 1., 'final_nav': float(navs[-1]),
               'start': str(days[0].date()), 'end': str(days[-1].date()),
               'requested_start': str(start.date()), 'requested_end': str(end.date()),
               'mode': mode, 'allocation': float(allocation), 'horizon': int(horizon),
               'slippage_per_side': float(slippage), 'commission_per_side': COMMISSION,
               'turnover_definition': 'sum of gross buy and sell notionals / initial NAV',
               'allocation_policy': (
                   '0050 market value funding budget includes both conversion legs and their costs' if mode == 'signals'
                   else 'initial cash budgets split by allocation and include purchase costs; no rebalancing' if mode == 'static_mix'
                   else 'all initial cash funds 0050 including purchase costs; hold until final liquidation'),
               'valuation_units': 'adjusted-price total-return units, not executable shares'}
    return {'summary': summary, 'curve': curve, 'trades': trades,
            'executions': executions, 'rejections': rejections}
