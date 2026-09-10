"""Combine independently funded accounts without cross-account rebalancing.

The weights refer to initial capital, not daily target weights. This is an
accounting aggregation of complete simulations, never a splice of selected
equity-curve periods. Component ledgers remain with their original simulations.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd


AMOUNTS = ('nav', 'cash', 'active_value', 'benchmark_value', 'daily_cost',
           'cumulative_cost', 'market_pnl')
FEES = ('slippage_per_side', 'commission_per_side', 'stock_sell_tax', 'benchmark_sell_tax')


def _number(value, field, *, minimum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{field} must be a finite number')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must be a finite number') from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f'{field} must be finite and >= {minimum}')
    return result


def _count(value, field):
    result = _number(value, field, minimum=0)
    if not result.is_integer():
        raise ValueError(f'{field} must be a nonnegative integer')
    return int(result)


def _date(value, field):
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must be a timezone-naive date') from exc
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError(f'{field} must be a timezone-naive date')
    return result


def _equal(left, right, field):
    if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f'{field} is inconsistent with the component account')


def _component(sim, sleeve):
    if not isinstance(sim, dict) or not isinstance(sim.get('summary'), dict):
        raise ValueError(f'{sleeve} requires a simulation summary')
    summary, curve = sim['summary'], sim.get('curve')
    if not isinstance(curve, list) or len(curve) < 2:
        raise ValueError(f'{sleeve} requires at least two curve dates')
    try:
        days = pd.DatetimeIndex([_date(row['date'], f'{sleeve}.date') for row in curve])
        values = np.array([[_number(row[field], f'{sleeve}.{field}',
                                    minimum=None if field == 'market_pnl' else 0)
                            for field in AMOUNTS] for row in curve])
        counts = {field: _count(summary[field], f'{sleeve}.{field}')
                  for field in ('trade_count', 'entered_cohorts', 'completed_cohorts',
                                'peak_active_cohorts', 'blocked_exit_sessions', 'rejected_event_count')}
        active_counts = [_count(row['active_cohorts'], f'{sleeve}.active_cohorts') for row in curve]
        fees = {field: _number(summary[field], f'{sleeve}.{field}', minimum=0) for field in FEES}
        _equal(_number(summary['initial_nav'], f'{sleeve}.initial_nav'), 1., f'{sleeve}.initial_nav')
        start, end = _date(summary['start'], 'start'), _date(summary['end'], 'end')
        requested = (_date(summary['requested_start'], 'requested_start'),
                     _date(summary['requested_end'], 'requested_end'))
        total_cost = _number(summary['total_cost'], f'{sleeve}.total_cost', minimum=0)
        turnover = _number(summary['turnover'], f'{sleeve}.turnover', minimum=0)
        final_nav = _number(summary['final_nav'], f'{sleeve}.final_nav', minimum=0)
        final_cash = _number(summary['final_cash'], f'{sleeve}.final_cash', minimum=0)
        holdings = summary['unliquidated_positions']
    except KeyError as exc:
        raise ValueError(f'{sleeve} is missing required account field {exc.args[0]}') from exc
    if not days.is_unique or not days.is_monotonic_increasing or days[-1] <= days[0]:
        raise ValueError(f'{sleeve} dates must be strictly increasing and unique')
    if start != days[0] or end != days[-1] or requested[0] > start or requested[1] < end:
        raise ValueError(f'{sleeve} start/end dates do not match its curve')
    if (values[:, 0] <= 0).any():
        raise ValueError(f'{sleeve} NAV must be strictly positive')
    if not np.allclose(values[:, 0], values[:, 1:4].sum(axis=1), rtol=1e-10, atol=1e-12):
        raise ValueError(f'{sleeve} NAV must equal cash plus stock and benchmark values')
    previous = np.r_[1., values[:-1, 0]]
    if not np.allclose(values[:, 0], previous + values[:, 6] - values[:, 4], rtol=1e-10, atol=1e-12):
        raise ValueError(f'{sleeve} daily market P&L/cost conservation failed')
    if not np.allclose(values[:, 5], np.cumsum(values[:, 4]), rtol=1e-10, atol=1e-12):
        raise ValueError(f'{sleeve} cumulative cost does not match daily costs')
    _equal(total_cost, values[-1, 5], f'{sleeve}.total_cost')
    _equal(final_nav, values[-1, 0], f'{sleeve}.final_nav')
    _equal(final_cash, values[-1, 1], f'{sleeve}.final_cash')
    if not isinstance(holdings, list):
        raise ValueError(f'{sleeve} unliquidated_positions must be a list')
    stock_value = benchmark_value = 0.
    seen = set()
    for holding in holdings:
        try:
            sid = holding['stock_id']
            units = _number(holding['units'], f'{sleeve}.units', minimum=0)
            mark = _number(holding['mark'], f'{sleeve}.mark', minimum=0)
            marked_value = _number(holding['marked_value'], f'{sleeve}.marked_value', minimum=0)
        except KeyError as exc:
            raise ValueError(f'{sleeve} holding missing {exc.args[0]}') from exc
        if not isinstance(sid, str) or not re.fullmatch(r'\d{4}', sid) or sid in seen or units <= 0 or mark <= 0:
            raise ValueError(f'{sleeve} holdings require unique four-digit ids and positive units/marks')
        seen.add(sid)
        _equal(units * mark, marked_value, f'{sleeve}.{sid}.marked_value')
        if sid == '0050':
            benchmark_value += marked_value
        else:
            stock_value += marked_value
    _equal(stock_value, values[-1, 2], f'{sleeve}.final_stock_holdings')
    _equal(benchmark_value, values[-1, 3], f'{sleeve}.final_benchmark_holdings')
    if (summary.get('unliquidated_position_count') != len(holdings)
            or summary.get('final_liquidation_complete') is not (not holdings)
            or summary.get('final_nav_is_marked') is not bool(holdings)):
        raise ValueError(f'{sleeve} final liquidation flags disagree with its holdings')
    if counts['completed_cohorts'] > counts['entered_cohorts']:
        raise ValueError(f'{sleeve} cannot complete more cohorts than it entered')
    return {'summary': summary, 'days': days, 'values': values, 'counts': counts,
            'active_counts': active_counts, 'fees': fees, 'requested': requested,
            'turnover': turnover, 'holdings': holdings}


def mix_portfolios(strategy: dict, benchmark: dict, weight=.5) -> dict:
    """Initially fund two sleeves, then sum their independently evolving NAVs.

    Inputs are complete unit-initial-NAV simulation accounts on the same dates
    and fee basis. ``benchmark`` must be the passive 0050 account, without stock
    events. Fees and dollar turnover scale by initial capital; transaction and
    cohort counts remain actual counts, not fractional trades. Open positions
    retain their own sleeve identities, with units and marked values scaled.

    Only summary, curve and component references are returned. The caller must
    retain the original component ledgers instead of treating this aggregation
    as a newly simulated or merged trade ledger.
    """
    weight = _number(weight, 'weight')
    if not 0 < weight < 1:
        raise ValueError('weight must be strictly between zero and one')
    a, b = _component(strategy, 'strategy'), _component(benchmark, 'benchmark')
    if not a['days'].equals(b['days']) or a['requested'] != b['requested']:
        raise ValueError('Component dates and requested start/end must align exactly')
    if a['fees'] != b['fees']:
        raise ValueError('Component fee rates must be identical')
    for field in ('valuation_units', 'turnover_definition'):
        if not a['summary'].get(field) or a['summary'].get(field) != b['summary'].get(field):
            raise ValueError(f'Component {field} must be explicit and identical')
    if (b['summary'].get('mode') != 'benchmark'
            or b['counts']['entered_cohorts'] or b['counts']['completed_cohorts']
            or any(b['active_counts']) or np.any(b['values'][:, 2] != 0)):
        raise ValueError('Benchmark sleeve must be passive 0050 without stock cohorts')
    slots = _count(a['summary'].get('slots'), 'strategy.slots')
    horizon = _count(a['summary'].get('horizon'), 'strategy.horizon')
    if not slots or not horizon:
        raise ValueError('Strategy slots and horizon must be positive')
    combined = weight * a['values'] + (1. - weight) * b['values']
    curve = []
    annual, previous = {}, 1.
    for i, day in enumerate(a['days']):
        row = {'date': str(day.date()), **dict(zip(AMOUNTS, map(float, combined[i])))}
        row.update(active_weight=row['active_value'] / row['nav'],
                   cash_weight=row['cash'] / row['nav'],
                   active_cohorts=a['active_counts'][i],
                   strategy_sleeve_value=float(weight * a['values'][i, 0]),
                   benchmark_sleeve_value=float((1. - weight) * b['values'][i, 0]))
        curve.append(row)
        year = str(day.year)
        annual[year] = annual.get(year, 1.) * row['nav'] / previous
        previous = row['nav']
    holdings = []
    for name, fraction, component in [('strategy', weight, a), ('benchmark', 1. - weight, b)]:
        for holding in component['holdings']:
            holdings.append({**holding, 'sleeve': name, 'units': float(holding['units']) * fraction,
                             'marked_value': float(holding['marked_value']) * fraction})
    navs = np.r_[1., combined[:, 0]]
    years = (a['days'][-1] - a['days'][0]).days / 365.25
    summary = {
        'mode': 'fixed_mix', 'initial_nav': 1., 'initial_strategy_weight': weight,
        'total_return': float(navs[-1] - 1.), 'cagr': float(navs[-1] ** (1. / years) - 1.),
        'max_drawdown': float(np.min(navs / np.maximum.accumulate(navs) - 1.)),
        'mean_active_weight': float(np.mean([row['active_weight'] for row in curve])),
        'mean_cash_weight': float(np.mean([row['cash_weight'] for row in curve])),
        'total_cost': float(combined[-1, 5]),
        'turnover': float(weight * a['turnover'] + (1. - weight) * b['turnover']),
        'trade_count': a['counts']['trade_count'] + b['counts']['trade_count'],
        **{field: value for field, value in a['counts'].items() if field != 'trade_count'},
        'final_nav': float(navs[-1]), 'final_cash': float(combined[-1, 1]),
        'unliquidated_positions': holdings, 'unliquidated_position_count': len(holdings),
        'final_liquidation_complete': not holdings, 'final_nav_is_marked': bool(holdings),
        'start': str(a['days'][0].date()), 'end': str(a['days'][-1].date()),
        'requested_start': str(a['requested'][0].date()), 'requested_end': str(a['requested'][1].date()),
        'horizon': horizon, 'slots': slots, **a['fees'],
        'turnover_definition': a['summary']['turnover_definition'],
        'valuation_units': a['summary']['valuation_units'],
        'allocation_policy': 'two independently funded sleeves; fixed initial capital weights; no cross-sleeve rebalancing',
        'annual_returns': {year: value - 1. for year, value in annual.items()},
    }
    return {'summary': summary, 'curve': curve,
            'components': [{'sleeve': 'strategy', 'initial_weight': weight},
                           {'sleeve': 'benchmark', 'initial_weight': 1. - weight}]}
