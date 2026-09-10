"""Causal, cash-funded basket replacements of a 0050 research portfolio.

Prices are adjusted accounting units, not executable shares. This module only
accounts for supplied signals; it never selects a substitute on execution day,
fetches data, or places an order. Slippage is an explicit cost on notional.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype


COMMISSION = .001425
STOCK_SELL_TAX = .003
ETF_SELL_TAX = .001


def _date(value, field):
    try:
        date = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must be a timezone-naive date') from exc
    if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise ValueError(f'{field} must be a timezone-naive date')
    return date


def simulate_baskets(close: pd.DataFrame, flags: pd.DataFrame, events: list[dict], *,
                     start='2022-01-03', end='2026-06-23', slots=3, horizon=63,
                     slippage=.003, mode='events') -> dict:
    """Hold 0050 between non-overlapping, equal-cash event baskets.

    Events require unique event_id, signal_date, entry_date, members and finite
    priority. An entry must follow its signal strictly and is never shifted to
    another session. On ties, descending priority then event_id determines who
    can use a slot. Due exits precede entries, and rejected entries do not queue.

    A basket sells at most current NAV / slots of 0050; its net sale proceeds
    fund equal member cash budgets including purchase costs. At entry+horizon
    sessions, all members and 0050 must be tradable to close the whole cohort.
    Otherwise the exit waits. A final blocked cohort remains explicitly marked
    and unliquidated. Such a final NAV is not a realized liquidation return.
    """
    if mode not in {'events', 'benchmark'}:
        raise ValueError('mode must be events or benchmark')
    if isinstance(slots, bool) or not isinstance(slots, (int, np.integer)) or slots < 1:
        raise ValueError('slots must be a positive integer')
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError('horizon must be a positive integer number of sessions')
    if not math.isfinite(slippage) or not 0 <= slippage < 1 - COMMISSION - STOCK_SELL_TAX:
        raise ValueError('slippage must leave positive sale proceeds')
    if (not isinstance(close.index, pd.DatetimeIndex) or close.index.tz is not None
            or close.index.hasnans or not close.index.is_unique
            or not close.index.is_monotonic_increasing
            or not close.index.equals(close.index.normalize())
            or not close.columns.is_unique or '0050' not in close.columns
            or any(not isinstance(sid, str) or not re.fullmatch(r'\d{4}', sid) for sid in close.columns)
            or not close.index.equals(flags.index) or not close.columns.equals(flags.columns)
            or any(not is_bool_dtype(dtype) for dtype in flags.dtypes)):
        raise ValueError('Ordered date-only prices and aligned boolean flags for four-digit stocks and 0050 are required')
    start, end = _date(start, 'start'), _date(end, 'end')
    if end < start:
        raise ValueError('end must not precede start')
    days = close.index[(close.index >= start) & (close.index <= end)]
    if len(days) < 2:
        raise ValueError('At least two trading sessions are required')
    session_indices = {day: i for i, day in enumerate(days)}
    scheduled, seen, rejections, wanted = {}, set(), [], {'0050'}
    for supplied in events:
        required = {'event_id', 'signal_date', 'entry_date', 'members', 'priority'}
        if not required.issubset(supplied) or not str(supplied['event_id'] or ''):
            raise ValueError('Every event requires event_id, signal_date, entry_date, members, priority')
        event_id = str(supplied['event_id'])
        if event_id in seen:
            raise ValueError(f'Duplicate event_id: {event_id}')
        seen.add(event_id)
        signal = _date(supplied['signal_date'], f'{event_id}.signal_date')
        entry = _date(supplied['entry_date'], f'{event_id}.entry_date')
        if entry <= signal:
            raise ValueError(f'{event_id}: entry_date must be strictly after signal_date')
        members = supplied['members']
        if (not isinstance(members, (list, tuple)) or not members
                or any(not isinstance(sid, str) or not re.fullmatch(r'\d{4}', sid)
                       or sid == '0050' for sid in members) or len(set(members)) != len(members)):
            raise ValueError(f'{event_id}: members must be unique four-digit stocks excluding 0050')
        try:
            priority = float(supplied['priority'])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{event_id}: priority must be finite') from exc
        if not math.isfinite(priority):
            raise ValueError(f'{event_id}: priority must be finite')
        event = {'event_id': event_id, 'signal_date': str(signal.date()),
                 'entry_date': str(entry.date()), 'members': sorted(members), 'priority': priority}
        reason = ('mode_does_not_use_events' if mode == 'benchmark'
                  else 'outside_window' if entry < days[0] or entry > days[-1]
                  else 'entry_date_not_trading_session' if entry not in session_indices
                  else 'unknown_member' if any(sid not in close.columns for sid in members) else None)
        if reason:
            rejections.append({**event, 'reason': reason})
        else:
            scheduled.setdefault(session_indices[entry], []).append(event)
            wanted.update(members)
    for batch in scheduled.values():
        batch.sort(key=lambda event: (-event['priority'], event['event_id']))

    # Trim before materializing arrays; repeated variants avoid per-cell pandas
    # access and do not need prices for any stock they cannot hold.
    symbols = ['0050', *sorted(wanted - {'0050'})]
    symbol_index = {sid: i for i, sid in enumerate(symbols)}
    selected = close.loc[:, symbols].astype(float)
    observed = selected.where(np.isfinite(selected) & selected.gt(0))
    prices = observed.loc[days].to_numpy()
    marks = observed.ffill().loc[days].to_numpy()
    tradable = flags.loc[days, symbols].fillna(False).to_numpy(dtype=bool) & np.isfinite(prices)
    buy_rate = COMMISSION + slippage
    taxes = np.array([ETF_SELL_TAX, *[STOCK_SELL_TAX] * (len(symbols) - 1)])
    sell_rates = buy_rate + taxes
    cash = 1.
    units = np.zeros(len(symbols))
    executions, completed, active, curve = [], [], [], []
    cumulative_cost = cumulative_notional = 0.
    blocked_exit_sessions = peak_active_cohorts = 0
    previous_nav = 1.
    previous_units = units.copy()
    previous_marks = None

    def nav(i):
        held = units > 0
        return float(cash + np.dot(units[held], marks[i, held]))

    def fill(i, column, side, amount, event_id, reason):
        nonlocal cash, cumulative_cost, cumulative_notional
        if not tradable[i, column]:
            raise ArithmeticError('Attempted execution without a valid current tradable price')
        before = cash
        if side == 'buy':
            if amount < 0 or amount > cash + 1e-12:
                raise ArithmeticError('Purchase exceeds available cash')
            budget = min(float(amount), cash)
            notional = budget / (1 + buy_rate)
            while notional + notional * buy_rate > budget:
                notional = float(np.nextafter(notional, 0.))
            cost = notional * buy_rate
            quantity = notional / prices[i, column]
            cash -= notional + cost
            units[column] += quantity
        else:
            if amount < 0 or amount > units[column] + 1e-12:
                raise ArithmeticError('Sale exceeds the funded holding')
            quantity = min(float(amount), units[column])
            notional = quantity * prices[i, column]
            cost = notional * sell_rates[column]
            cash += notional - cost
            units[column] -= quantity
        cumulative_cost += cost
        cumulative_notional += notional
        result = {'date': str(days[i].date()), 'event_id': event_id,
                  'stock_id': symbols[column], 'side': side, 'units': float(quantity),
                  'price': float(prices[i, column]), 'notional': float(notional),
                  'cost': float(cost), 'commission': float(notional * COMMISSION),
                  'slippage': float(notional * slippage),
                  'sell_tax': float(notional * taxes[column] if side == 'sell' else 0.),
                  'cash_before': float(before), 'cash_after': float(cash), 'reason': reason}
        executions.append(result)
        return result

    def close_cohort(i, cohort, terminal):
        entry_nav = nav(i)
        member_results, exit_proceeds, sell_cost = [], 0., 0.
        reason = 'terminal_liquidation' if terminal else 'scheduled_exit'
        for member in cohort['member_entries']:
            sold = fill(i, symbol_index[member['stock_id']], 'sell', member['units'],
                        cohort['event_id'], reason)
            pnl = sold['notional'] - member['entry_notional'] - member['buy_cost'] - sold['cost']
            member_results.append({**member, 'exit_price': sold['price'],
                                   'exit_notional': sold['notional'], 'sell_cost': sold['cost'],
                                   'net_pnl': float(pnl),
                                   'net_return': float(pnl / (member['entry_notional'] + member['buy_cost']))})
            exit_proceeds += sold['notional'] - sold['cost']
            sell_cost += sold['cost']
        repurchased = (fill(i, 0, 'buy', exit_proceeds, cohort['event_id'], 'return_to_benchmark')
                       if not terminal else {'cost': 0., 'units': 0.})
        member_pnl = sum(item['net_pnl'] for item in member_results)
        conversion_cost = cohort['benchmark_sell_cost'] + repurchased['cost']
        completed.append({**{k: v for k, v in cohort.items() if k not in {'entry_index', 'columns'}},
                          'status': 'completed', 'exit_date': str(days[i].date()),
                          'exit_reason': reason, 'holding_sessions': i - cohort['entry_index'],
                          'exit_nav_before': entry_nav, 'exit_nav_after': nav(i),
                          'member_results': member_results, 'stock_net_pnl': float(member_pnl),
                          'benchmark_buy_cost': float(repurchased['cost']),
                          'benchmark_units_rebought': float(repurchased['units']),
                          'conversion_cost': float(conversion_cost),
                          'total_cost': float(cohort['stock_buy_cost'] + sell_cost + conversion_cost),
                          'cycle_net_pnl': float(member_pnl - conversion_cost)})

    for i, day in enumerate(days):
        terminal = i == len(days) - 1
        day_cost_before = cumulative_cost
        if i == 0:
            if not tradable[i, 0]:
                raise ValueError(f'0050 must be tradable on initial session {day.date()}')
            fill(i, 0, 'buy', cash, None, 'initial_benchmark')

        for cohort in list(active):
            if not terminal and i < cohort['entry_index'] + horizon:
                continue
            required = [0, *cohort['columns']]
            if not bool(np.all(tradable[i, required])):
                blocked_exit_sessions += 1
                cohort['blocked_exit_sessions'] += 1
                cohort['blocked_exits'].append({'date': str(day.date()),
                                               'stock_ids': [symbols[j] for j in required if not tradable[i, j]],
                                               'terminal': terminal})
                continue
            close_cohort(i, cohort, terminal)
            active.remove(cohort)

        if terminal and units[0] > 0 and tradable[i, 0]:
            fill(i, 0, 'sell', units[0], None, 'terminal_liquidation')

        for event in scheduled.get(i, []):
            columns = [symbol_index[sid] for sid in event['members']]
            overlapping = sorted({sid for cohort in active for sid in cohort['members']} & set(event['members']))
            reason = ('terminal_session' if terminal else 'overlapping_member' if overlapping
                      else 'slots_full' if len(active) >= slots
                      else 'entry_instruments_not_tradable' if not np.all(tradable[i, [0, *columns]])
                      else 'no_benchmark_funding' if units[0] <= 0 else None)
            if reason:
                rejected = {**event, 'reason': reason}
                if overlapping:
                    rejected['overlapping_members'] = overlapping
                rejections.append(rejected)
                continue
            before = nav(i)
            funding = min(before / slots, units[0] * prices[i, 0])
            sold = fill(i, 0, 'sell', funding / prices[i, 0], event['event_id'], 'fund_event_basket')
            budget = (sold['notional'] - sold['cost']) / len(columns)
            member_entries = []
            for column in columns:
                bought = fill(i, column, 'buy', budget, event['event_id'], 'event_entry')
                member_entries.append({'stock_id': symbols[column], 'units': bought['units'],
                                       'entry_price': bought['price'], 'entry_notional': bought['notional'],
                                       'buy_cost': bought['cost'], 'cash_budget': float(budget)})
            cohort = {**event, 'entry_index': i, 'columns': columns, 'entry_nav': before,
                      'entry_nav_after': nav(i), 'funding_budget': float(funding),
                      'benchmark_sell_cost': sold['cost'], 'benchmark_units_sold': sold['units'],
                      'member_entries': member_entries,
                      'stock_buy_cost': float(sum(item['buy_cost'] for item in member_entries)),
                      'realized_entry_weight': float(sum(item['entry_notional'] for item in member_entries) / nav(i)),
                      'blocked_exit_sessions': 0, 'blocked_exits': [],
                      'due_date': str(days[i + horizon].date()) if i + horizon < len(days) else None}
            active.append(cohort)
            peak_active_cohorts = max(peak_active_cohorts, len(active))

        equity = nav(i)
        daily_cost = cumulative_cost - day_cost_before
        previously_held = previous_units > 0
        market_pnl = (float(np.dot(previous_units[previously_held],
                                  marks[i, previously_held] - previous_marks[previously_held]))
                      if previous_marks is not None else 0.)
        if cash < 0 or np.any(units < 0) or not math.isfinite(equity) or equity <= 0:
            raise ArithmeticError('Non-finite NAV, negative cash, or an unfunded holding')
        if not math.isclose(equity, previous_nav + market_pnl - daily_cost, rel_tol=1e-10, abs_tol=1e-12):
            raise ArithmeticError('Daily cash/position/cost conservation failed')
        held_active = units[1:] > 0
        active_value = float(np.dot(units[1:][held_active], marks[i, 1:][held_active]))
        curve.append({'date': str(day.date()), 'nav': equity, 'cash': float(cash),
                      'active_weight': active_value / equity, 'active_value': active_value,
                      'benchmark_value': float(units[0] * marks[i, 0]) if units[0] else 0.,
                      'active_cohorts': len(active), 'daily_cost': float(daily_cost),
                      'cumulative_cost': float(cumulative_cost), 'market_pnl': market_pnl})
        previous_nav, previous_units, previous_marks = equity, units.copy(), marks[i].copy()

    unliquidated = []
    for column in np.flatnonzero(units > 0):
        valid_dates = observed.index[(observed.index <= days[-1]) & observed.iloc[:, column].notna()]
        unliquidated.append({'stock_id': symbols[column], 'units': float(units[column]),
                             'mark': float(marks[-1, column]),
                             'marked_value': float(units[column] * marks[-1, column]),
                             'mark_date': str(valid_dates[-1].date()),
                             'current_price_available': bool(np.isfinite(prices[-1, column])),
                             'tradable_on_final_session': bool(tradable[-1, column])})
    open_cohorts = []
    for cohort in active:
        member_results = []
        for member in cohort['member_entries']:
            column = symbol_index[member['stock_id']]
            value = member['units'] * marks[-1, column]
            member_results.append({**member, 'mark': float(marks[-1, column]), 'marked_value': float(value),
                                   'unrealized_pnl_after_buy_cost': float(value - member['entry_notional'] - member['buy_cost'])})
        open_cohorts.append({**{k: v for k, v in cohort.items() if k not in {'entry_index', 'columns'}},
                             'status': 'open', 'exit_date': None, 'exit_reason': 'final_exit_blocked',
                             'holding_sessions': len(days) - 1 - cohort['entry_index'],
                             'member_results': member_results,
                             'total_cost': float(cohort['stock_buy_cost'] + cohort['benchmark_sell_cost'])})
    navs = np.array([1., *(row['nav'] for row in curve)])
    years = (days[-1] - days[0]).days / 365.25
    summary = {'total_return': float(navs[-1] - 1), 'cagr': float(navs[-1] ** (1 / years) - 1),
               'max_drawdown': float(np.min(navs / np.maximum.accumulate(navs) - 1)),
               'total_cost': float(cumulative_cost), 'turnover': float(cumulative_notional),
               'trade_count': len(executions), 'completed_cohorts': len(completed),
               'entered_cohorts': len(completed) + len(open_cohorts), 'peak_active_cohorts': peak_active_cohorts,
               'mean_active_weight': float(np.mean([row['active_weight'] for row in curve])),
               'blocked_exit_sessions': blocked_exit_sessions, 'rejected_event_count': len(rejections),
               'unliquidated_positions': unliquidated, 'unliquidated_position_count': len(unliquidated),
               'final_liquidation_complete': not unliquidated, 'final_nav_is_marked': bool(unliquidated),
               'initial_nav': 1., 'final_nav': float(navs[-1]), 'final_cash': float(cash),
               'start': str(days[0].date()), 'end': str(days[-1].date()),
               'requested_start': str(start.date()), 'requested_end': str(end.date()),
               'mode': mode, 'slots': int(slots), 'horizon': int(horizon),
               'slippage_per_side': float(slippage), 'commission_per_side': COMMISSION,
               'stock_sell_tax': STOCK_SELL_TAX, 'benchmark_sell_tax': ETF_SELL_TAX,
               'turnover_definition': 'sum of gross execution notionals / initial NAV',
               'allocation_policy': 'up to current NAV / slots of 0050 sold; net proceeds fund equal member budgets including costs',
               'valuation_units': 'adjusted-price accounting units, not executable shares'}
    return {'summary': summary, 'curve': curve, 'executions': executions,
            'cohorts': sorted([*completed, *open_cohorts], key=lambda row: (row['entry_date'], -row['priority'], row['event_id'])),
            'rejections': sorted(rejections, key=lambda row: (row['entry_date'], -row['priority'], row['event_id']))}
