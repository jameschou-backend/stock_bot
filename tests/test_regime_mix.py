"""Synthetic independent-account checks; no historical inputs or downloads."""
from copy import deepcopy
import json
import math

import numpy as np
import pandas as pd
import pytest

from skills.diffusion_portfolio import simulate_baskets
from skills.regime_mix import mix_portfolios


def account(navs, *, benchmark=False, costs=None, marked=False):
    """Construct an explicit self-financing accounting curve, not daily returns."""
    days = ['2022-01-03', '2023-01-03', '2024-01-03']
    costs = costs or [0., 0., 0.]
    previous, cumulative = 1., 0.
    curve = []
    for i, (day, nav, cost) in enumerate(zip(days, navs, costs)):
        cumulative += cost
        cash = nav if i == len(days) - 1 and not marked else 0.
        stock = 0. if benchmark or cash else nav * .8
        etf = nav - cash - stock
        curve.append({'date': day, 'nav': nav, 'cash': cash,
                      'active_value': stock, 'benchmark_value': etf,
                      'active_weight': stock / nav, 'active_cohorts': int(stock > 0),
                      'daily_cost': cost, 'cumulative_cost': cumulative,
                      'market_pnl': nav - previous + cost})
        previous = nav
    holdings = []
    if marked:
        for sid, value in [('1101', curve[-1]['active_value']), ('0050', curve[-1]['benchmark_value'])]:
            if value:
                holdings.append({'stock_id': sid, 'units': value / 100., 'mark': 100.,
                                 'marked_value': value, 'mark_date': days[-2],
                                 'current_price_available': False, 'tradable_on_final_session': False})
    all_navs = np.r_[1., navs]
    years = (pd.Timestamp(days[-1]) - pd.Timestamp(days[0])).days / 365.25
    summary = {
        'initial_nav': 1., 'final_nav': navs[-1], 'final_cash': curve[-1]['cash'],
        'total_return': navs[-1] - 1, 'cagr': navs[-1] ** (1 / years) - 1,
        'max_drawdown': float(np.min(all_navs / np.maximum.accumulate(all_navs) - 1)),
        'total_cost': sum(costs), 'turnover': 3. if benchmark else 9.,
        'trade_count': 2 if benchmark else 6, 'entered_cohorts': 0 if benchmark else 1,
        'completed_cohorts': 0 if benchmark or marked else 1,
        'peak_active_cohorts': 0 if benchmark else 1,
        'blocked_exit_sessions': int(marked and not benchmark), 'rejected_event_count': 0,
        'unliquidated_positions': holdings, 'unliquidated_position_count': len(holdings),
        'final_liquidation_complete': not holdings, 'final_nav_is_marked': bool(holdings),
        'start': days[0], 'end': days[-1], 'requested_start': days[0], 'requested_end': days[-1],
        'mode': 'benchmark' if benchmark else 'events', 'slots': 3, 'horizon': 63,
        'slippage_per_side': .0045, 'commission_per_side': .001425,
        'stock_sell_tax': .003, 'benchmark_sell_tax': .001,
        'turnover_definition': 'sum of gross execution notionals / initial NAV',
        'valuation_units': 'adjusted-price accounting units, not executable shares',
    }
    return {'summary': summary, 'curve': curve, 'executions': [{'do_not_merge': True}]}


def test_mixed_drawdown_and_cagr_are_computed_from_the_mixed_nav_path():
    strategy = account([.99, 1.8, 1.0], costs=[.01, 0., .02])
    benchmark = account([.995, .8, 1.5], benchmark=True, costs=[.005, 0., .005])
    result = mix_portfolios(strategy, benchmark)
    assert [row['nav'] for row in result['curve']] == pytest.approx([.9925, 1.3, 1.25])
    summary = result['summary']
    years = (pd.Timestamp('2024-01-03') - pd.Timestamp('2022-01-03')).days / 365.25
    assert summary['cagr'] == pytest.approx(1.25 ** (1 / years) - 1)
    assert summary['max_drawdown'] == pytest.approx(1.25 / 1.3 - 1)
    for field in ('cagr', 'max_drawdown'):
        assert summary[field] != pytest.approx((strategy['summary'][field] + benchmark['summary'][field]) / 2)
    assert math.prod(1 + value for value in summary['annual_returns'].values()) == pytest.approx(1.25)


def test_initial_costs_cash_and_dollar_turnover_scale_but_trade_counts_do_not():
    strategy = account([.98, 1.2, 1.1], costs=[.02, .01, .03])
    benchmark = account([.99, 1.1, 1.2], benchmark=True, costs=[.01, 0., .01])
    result = mix_portfolios(strategy, benchmark, weight=.25)
    summary, curve = result['summary'], result['curve']
    assert curve[0]['nav'] == pytest.approx(.25 * .98 + .75 * .99)
    assert curve[0]['daily_cost'] == pytest.approx(.25 * .02 + .75 * .01)
    assert curve[0]['market_pnl'] == pytest.approx(0.)
    assert summary['total_cost'] == pytest.approx(.25 * .06 + .75 * .02)
    assert summary['turnover'] == pytest.approx(.25 * 9 + .75 * 3)
    assert summary['trade_count'] == 8
    assert summary['entered_cohorts'] == summary['completed_cohorts'] == 1
    assert summary['final_cash'] == pytest.approx(summary['final_nav'])
    assert curve[-1]['cash_weight'] == 1.
    assert summary['mean_cash_weight'] == pytest.approx(1 / 3)
    for i, row in enumerate(curve):
        previous = 1. if not i else curve[i - 1]['nav']
        assert row['nav'] == pytest.approx(previous + row['market_pnl'] - row['daily_cost'])
        assert row['nav'] == pytest.approx(row['cash'] + row['active_value'] + row['benchmark_value'])
        assert row['active_weight'] == pytest.approx(row['active_value'] / row['nav'])


def test_initial_half_sleeves_drift_instead_of_rebalancing_each_day():
    result = mix_portfolios(account([1., 2., 2.]), account([1., 1., 2.], benchmark=True))
    curve = result['curve']
    assert [row['nav'] for row in curve] == [1., 1.5, 2.]
    assert curve[1]['strategy_sleeve_value'] / curve[1]['nav'] == pytest.approx(2 / 3)
    assert curve[-1]['nav'] != 2.25  # Daily 50/50 rebalance: 1.5 * 1.5.
    assert result['components'] == [{'sleeve': 'strategy', 'initial_weight': .5},
                                    {'sleeve': 'benchmark', 'initial_weight': .5}]
    assert set(result) == {'summary', 'curve', 'components'}
    assert result['summary']['mode'] == 'fixed_mix'


def test_unliquidated_units_and_marks_remain_consistent_in_separate_sleeves():
    strategy = account([1., 1.5, 2.], marked=True)
    benchmark = account([1., 1.2, 1.4], benchmark=True, marked=True)
    before = deepcopy((strategy, benchmark))
    result = mix_portfolios(strategy, benchmark, weight=.25)
    summary = result['summary']
    assert not summary['final_liquidation_complete']
    assert summary['final_nav_is_marked']
    assert summary['unliquidated_position_count'] == 3
    holdings = summary['unliquidated_positions']
    assert [(row['sleeve'], row['stock_id']) for row in holdings] == [
        ('strategy', '1101'), ('strategy', '0050'), ('benchmark', '0050')]
    assert sum(row['marked_value'] for row in holdings) == pytest.approx(summary['final_nav'])
    for row in holdings:
        fraction = .25 if row['sleeve'] == 'strategy' else .75
        original = strategy if row['sleeve'] == 'strategy' else benchmark
        prior = next(x for x in original['summary']['unliquidated_positions'] if x['stock_id'] == row['stock_id'])
        assert row['units'] == pytest.approx(prior['units'] * fraction)
        assert row['marked_value'] == pytest.approx(row['units'] * row['mark'])
        assert row['mark'] == prior['mark']
        assert row['mark_date'] == prior['mark_date']
    assert (strategy, benchmark) == before
    json.dumps(result, allow_nan=False)


def test_actual_account_engines_compose_on_tiny_synthetic_prices():
    days = pd.bdate_range('2022-01-03', periods=7)
    close = pd.DataFrame({'0050': [100., 101., 100., 102., 103., 104., 105.],
                          '1101': [100., 100., 105., 95., 105., 108., 109.]}, index=days)
    flags = pd.DataFrame(True, index=days, columns=close.columns)
    events = [{'event_id': 'synthetic', 'signal_date': str(days[0].date()),
               'entry_date': str(days[1].date()), 'members': ['1101'], 'priority': 1.}]
    strategy = simulate_baskets(close, flags, events, start=days[0], end=days[-1], horizon=3)
    benchmark = simulate_baskets(close, flags, [], start=days[0], end=days[-1], mode='benchmark')
    result = mix_portfolios(strategy, benchmark)
    for i, row in enumerate(result['curve']):
        assert row['nav'] == pytest.approx((strategy['curve'][i]['nav'] + benchmark['curve'][i]['nav']) / 2)
    assert result['summary']['horizon'] == 3
    assert result['summary']['final_liquidation_complete']


@pytest.mark.parametrize('weight', [0, 1, -.1, 1.1, np.nan, np.inf, True])
def test_invalid_initial_weights_fail(weight):
    with pytest.raises(ValueError):
        mix_portfolios(account([1., 2., 2.]), account([1., 1., 2.], benchmark=True), weight=weight)


@pytest.mark.parametrize('problem', ['fee', 'date', 'summary_date', 'initial_nav', 'negative_cash',
                                     'zero_nav', 'infinite_nav', 'missing_field', 'cost', 'nonpassive',
                                     'wrong_holding_units', 'false_liquidation'])
def test_inconsistent_or_misaligned_component_accounts_fail(problem):
    strategy = account([1., 1.2, 1.4], marked=True)
    benchmark = account([1., 1.1, 1.2], benchmark=True, marked=True)
    if problem == 'fee':
        benchmark['summary']['slippage_per_side'] = .003
    elif problem == 'date':
        benchmark['curve'][1]['date'] = '2023-01-04'
    elif problem == 'summary_date':
        benchmark['summary']['start'] = '2022-01-04'
    elif problem == 'initial_nav':
        strategy['summary']['initial_nav'] = 2.
    elif problem == 'negative_cash':
        strategy['curve'][1]['cash'] = -.01
    elif problem == 'zero_nav':
        strategy['curve'][1]['nav'] = 0.
    elif problem == 'infinite_nav':
        strategy['curve'][1]['nav'] = np.inf
    elif problem == 'missing_field':
        del strategy['curve'][1]['market_pnl']
    elif problem == 'cost':
        strategy['summary']['total_cost'] = .01
    elif problem == 'nonpassive':
        benchmark['summary']['entered_cohorts'] = 1
    elif problem == 'wrong_holding_units':
        strategy['summary']['unliquidated_positions'][0]['units'] *= 2
    else:
        strategy['summary']['final_liquidation_complete'] = True
    with pytest.raises(ValueError):
        mix_portfolios(strategy, benchmark)
