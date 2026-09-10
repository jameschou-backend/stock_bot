"""Causal controls, independently funded cash/ETF switching, and latched exits."""
import numpy as np
import pandas as pd
import pytest

from skills.diffusion_portfolio import simulate_baskets
from skills.regime_portfolio import simulate_regime


def data(n=12):
    days = pd.bdate_range('2022-01-03', periods=n)
    prices = pd.DataFrame(100., index=days, columns=['0050', '1101', '1102', '1103', '1104'])
    flags = pd.DataFrame(True, index=days, columns=prices.columns)
    return prices, flags


def controls(prices, states=None):
    states = ['ON'] * len(prices) if states is None else states
    return pd.DataFrame({'state': states,
                         'decision_date': [str((day - pd.Timedelta(days=1)).date()) for day in prices.index]},
                        index=prices.index)


def event(prices, index=1, members=('1101',), name='e', priority=1.):
    return {'event_id': name, 'signal_date': str((prices.index[index] - pd.Timedelta(days=1)).date()),
            'entry_date': str(prices.index[index].date()), 'members': list(members), 'priority': priority}


def run(prices, flags, events=(), state=None, **kwargs):
    return simulate_regime(prices, flags, list(events), controls(prices) if state is None else state,
                           start=prices.index[0], end=prices.index[-1], **kwargs)


@pytest.mark.parametrize('policy', ['idle_cash', 'exit_cash'])
def test_all_on_matches_sealed_diffusion_accounting(policy):
    prices, flags = data(40)
    prices['0050'] = np.linspace(100, 108, len(prices))
    prices['1101'] = np.linspace(100, 150, len(prices))
    prices['1102'] = np.linspace(100, 80, len(prices))
    events = [event(prices, members=('1101', '1102')),
              event(prices, index=2, members=('1103',), name='second')]
    old = simulate_baskets(prices, flags, events, start=prices.index[0], end=prices.index[-1], horizon=5)
    new = run(prices, flags, events, policy=policy, horizon=5)
    assert new['executions'] == old['executions']
    for key in ('total_return', 'cagr', 'max_drawdown', 'total_cost', 'turnover', 'trade_count', 'mean_active_weight'):
        assert new['summary'][key] == pytest.approx(old['summary'][key])
    assert [row['nav'] for row in new['curve']] == [row['nav'] for row in old['curve']]
    assert new['summary']['forced_exit_requested_cohorts'] == 0


@pytest.mark.parametrize('initial_state', ['OFF', 'UNKNOWN'])
def test_initial_off_or_unknown_holds_cash_even_without_any_benchmark_quote(initial_state):
    prices, flags = data()
    prices['0050'] = np.nan
    flags['0050'] = False
    state = controls(prices, [initial_state] * len(prices))
    if initial_state == 'UNKNOWN':
        state['decision_date'] = None
    result = run(prices, flags, [event(prices)], state=state)
    assert result['executions'] == []
    assert result['summary']['final_nav'] == 1
    assert result['summary']['mean_cash_weight'] == 1
    assert result['summary']['final_liquidation_complete']
    assert result['rejections'][0]['reason'] == 'market_state_not_on'


@pytest.mark.parametrize('decision_offset', [0, 1])
def test_command_cannot_use_same_day_or_future_decision(decision_offset):
    prices, flags = data()
    state = controls(prices)
    state.loc[prices.index[2], 'decision_date'] = str(prices.index[2 + decision_offset].date())
    with pytest.raises(ValueError, match='strictly before'):
        run(prices, flags, state=state)


def test_known_command_requires_date_and_controls_must_align():
    prices, flags = data()
    state = controls(prices)
    state.loc[prices.index[1], 'decision_date'] = None
    with pytest.raises(ValueError, match='Known commands require'):
        run(prices, flags, state=state)
    with pytest.raises(ValueError, match='align exactly'):
        run(prices, flags, state=controls(prices).iloc[1:])


def test_initial_on_untradable_etf_waits_in_cash_without_fabricated_fill():
    prices, flags = data()
    flags.loc[prices.index[0], '0050'] = False
    result = run(prices, flags, [event(prices, index=0)])
    assert result['curve'][0]['cash'] == result['curve'][0]['nav'] == 1
    assert result['executions'][0]['date'] == str(prices.index[1].date())
    assert result['summary']['etf_blocked_sessions'] == 1
    assert result['rejections'][0]['reason'] == 'entry_instruments_not_tradable'


def test_delayed_command_is_not_applied_on_its_source_day():
    prices, flags = data()
    state = controls(prices, ['ON', 'ON', 'ON', *['OFF'] * 9])
    state.loc[prices.index[3], 'decision_date'] = str(prices.index[1].date())
    result = run(prices, flags, state=state)
    sales = [fill for fill in result['executions'] if fill['reason'] == 'park_off']
    assert len(sales) == 1
    assert sales[0]['date'] == str(prices.index[3].date())
    assert result['curve'][2]['benchmark_value'] > 0
    assert result['curve'][3]['benchmark_value'] == 0


def test_unknown_keeps_last_park_target_blocks_entries_and_does_not_force_exit():
    prices, flags = data()
    state = controls(prices, ['ON', 'ON', *['UNKNOWN'] * 10])
    result = run(prices, flags, [event(prices), event(prices, index=3, members=('1102',), name='unknown')],
                 state=state, policy='exit_cash', horizon=6)
    assert result['cohorts'][0]['exit_date'] == str(prices.index[7].date())
    assert result['cohorts'][0]['exit_reason'] == 'scheduled_exit'
    assert result['summary']['forced_exit_requested_cohorts'] == 0
    assert result['curve'][3]['state'] == 'UNKNOWN'
    assert result['curve'][3]['park_state'] == 'ON'
    assert result['rejections'][0]['state'] == 'UNKNOWN'
    assert result['rejections'][0]['reason'] == 'market_state_not_on'


def test_unknown_after_off_keeps_cash_until_a_new_known_on():
    prices, flags = data()
    state = controls(prices, ['ON', 'OFF', 'UNKNOWN', 'UNKNOWN', *['ON'] * 8])
    result = run(prices, flags, state=state)
    assert result['curve'][2]['cash_weight'] == 1
    assert result['curve'][3]['park_state'] == 'OFF'
    buys = [fill for fill in result['executions'] if fill['reason'] == 'park_on']
    assert len(buys) == 1 and buys[0]['date'] == str(prices.index[4].date())


@pytest.mark.parametrize('intermediate_state', ['ON', 'UNKNOWN'])
def test_forced_stock_exit_latches_through_off_to_on_and_does_not_reenter_old_signal(intermediate_state):
    prices, flags = data()
    state = controls(prices, ['ON', 'ON', 'ON', 'OFF', intermediate_state, *['ON'] * 7])
    flags.loc[prices.index[3:5], '1101'] = False
    result = run(prices, flags, [event(prices)], state=state, policy='exit_cash')
    cohort = result['cohorts'][0]
    assert cohort['forced_exit_requested_on'] == str(prices.index[3].date())
    assert cohort['exit_date'] == str(prices.index[5].date())
    assert cohort['exit_reason'] == 'forced_exit'
    assert cohort['blocked_exit_sessions'] == 2
    assert result['summary']['forced_exit_requested_cohorts'] == 1
    assert result['summary']['forced_exit_completed_cohorts'] == 1
    stock_buys = [fill for fill in result['executions'] if fill['stock_id'] == '1101' and fill['side'] == 'buy']
    assert len(stock_buys) == 1
    assert any(fill['date'] == str(prices.index[5].date()) and fill['reason'] == 'park_on'
               for fill in result['executions'])


def test_forced_basket_exit_is_atomic_and_independent_of_etf_halt():
    prices, flags = data()
    state = controls(prices, ['ON', 'ON', 'OFF', 'OFF', *['ON'] * 8])
    flags.loc[prices.index[2], '1102'] = False
    flags.loc[prices.index[2:4], '0050'] = False
    result = run(prices, flags, [event(prices, members=('1101', '1102'))], state=state, policy='exit_cash')
    sales = [fill for fill in result['executions'] if fill['stock_id'] != '0050' and fill['side'] == 'sell']
    assert len(sales) == 2 and {fill['date'] for fill in sales} == {str(prices.index[3].date())}
    assert result['curve'][3]['cash'] > 0 and result['curve'][3]['benchmark_value'] > 0
    assert result['summary']['etf_blocked_sessions'] == 2
    assert not any(fill['reason'] == 'park_off' for fill in result['executions'])


def test_idle_cash_keeps_stocks_to_expiry_then_exits_without_waiting_for_halted_etf():
    prices, flags = data()
    state = controls(prices, ['ON', 'ON', *['OFF'] * 10])
    flags.loc[prices.index[2:6], '0050'] = False
    result = run(prices, flags, [event(prices)], state=state, policy='idle_cash', horizon=3)
    cohort = result['cohorts'][0]
    assert cohort['exit_date'] == str(prices.index[4].date())
    assert cohort['exit_reason'] == 'scheduled_exit'
    assert cohort['benchmark_buy_cost'] == 0
    assert result['summary']['forced_exit_requested_cohorts'] == 0
    etf_sales = [fill for fill in result['executions'] if fill['reason'] == 'park_off']
    assert len(etf_sales) == 1 and etf_sales[0]['date'] == str(prices.index[6].date())
    assert result['summary']['etf_blocked_sessions'] == 4


def test_scheduled_on_exit_requires_benchmark_for_repurchase():
    prices, flags = data()
    flags.loc[prices.index[3], '0050'] = False
    result = run(prices, flags, [event(prices)], horizon=2)
    assert result['cohorts'][0]['exit_date'] == str(prices.index[4].date())
    assert result['cohorts'][0]['blocked_exits'][0]['stock_ids'] == ['0050']
    assert result['summary']['etf_blocked_sessions'] == 1


def test_terminal_stock_liquidation_does_not_require_benchmark_quote():
    prices, flags = data()
    flags.loc[prices.index[-1], '0050'] = False
    result = run(prices, flags, [event(prices)])
    assert result['cohorts'][0]['exit_reason'] == 'terminal_liquidation'
    assert result['summary']['completed_cohorts'] == 1
    assert not result['summary']['final_liquidation_complete']
    assert {position['stock_id'] for position in result['summary']['unliquidated_positions']} == {'0050'}
    assert result['summary']['final_cash'] > 0


def test_forced_exit_pending_at_end_remains_latched_and_marked():
    prices, flags = data()
    state = controls(prices, ['ON', 'ON', 'OFF', *['ON'] * 9])
    flags.loc[prices.index[2]:, '1102'] = False
    prices.loc[prices.index[-1], '1102'] = np.nan
    result = run(prices, flags, [event(prices, members=('1101', '1102'))], state=state, policy='exit_cash')
    assert not result['summary']['final_liquidation_complete']
    assert result['summary']['forced_exit_pending_cohorts'] == 1
    assert result['cohorts'][0]['status'] == 'open'
    assert result['cohorts'][0]['forced_exit_requested_on'] == str(prices.index[2].date())
    assert {position['stock_id'] for position in result['summary']['unliquidated_positions']} == {'1101', '1102'}
    assert not any(fill['stock_id'] == '1101' and fill['side'] == 'sell' for fill in result['executions'])
    assert result['summary']['final_nav'] == pytest.approx(result['summary']['final_cash'] + sum(
        item['marked_value'] for item in result['summary']['unliquidated_positions']))


def test_switching_costs_and_every_cash_movement_reconcile_without_borrowing():
    prices, flags = data()
    state = controls(prices, ['ON', 'ON', 'OFF', 'ON', 'ON', 'OFF', 'UNKNOWN', 'ON', 'OFF', 'ON', 'ON', 'ON'])
    result = run(prices, flags, [event(prices), event(prices, index=4, name='second')],
                 state=state, policy='exit_cash')
    assert result['summary']['final_nav'] == pytest.approx(1 - result['summary']['total_cost'])
    assert all(row['cash'] >= 0 and 0 <= row['cash_weight'] <= 1 for row in result['curve'])
    for fill in result['executions']:
        expected = -fill['notional'] - fill['cost'] if fill['side'] == 'buy' else fill['notional'] - fill['cost']
        assert fill['cash_after'] - fill['cash_before'] == pytest.approx(expected)
        assert fill['cost'] == pytest.approx(fill['commission'] + fill['slippage'] + fill['sell_tax'])
        assert fill['units'] > 0


def test_on_parks_cash_before_funding_new_stock_event():
    prices, flags = data()
    state = controls(prices, ['OFF', 'OFF', *['ON'] * 10])
    result = run(prices, flags, [event(prices, index=2)], state=state)
    day_fills = [fill for fill in result['executions'] if fill['date'] == str(prices.index[2].date())]
    assert [(fill['stock_id'], fill['side']) for fill in day_fills] == [
        ('0050', 'buy'), ('0050', 'sell'), ('1101', 'buy')]
    cohort = result['cohorts'][0]
    assert cohort['funding_budget'] == pytest.approx(cohort['entry_nav'] / 3)


def test_future_control_changes_do_not_change_past_fills_or_nav():
    prices, flags = data()
    state = controls(prices)
    original = run(prices, flags, [event(prices)], state=state, policy='exit_cash')
    state.loc[prices.index[6]:, 'state'] = 'OFF'
    changed = run(prices, flags, [event(prices)], state=state, policy='exit_cash')
    assert original['curve'][:6] == changed['curve'][:6]
    cutoff = str(prices.index[6].date())
    assert [fill for fill in original['executions'] if fill['date'] < cutoff] == [
        fill for fill in changed['executions'] if fill['date'] < cutoff]


def test_large_stock_loss_is_not_clipped_before_forced_exit():
    prices, flags = data()
    prices.loc[prices.index[2]:, '1101'] = 1.
    state = controls(prices, ['ON', 'ON', *['OFF'] * 10])
    result = run(prices, flags, [event(prices)], state=state, policy='exit_cash', slots=1)
    assert result['summary']['total_return'] < -.99
    assert result['cohorts'][0]['member_results'][0]['net_return'] < -.99
    assert result['summary']['final_liquidation_complete']
