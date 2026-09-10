"""Synthetic accounting and causality checks for the offline basket simulator."""
import numpy as np
import pandas as pd
import pytest

from skills.diffusion_portfolio import COMMISSION, simulate_baskets


def data(n=12, symbols=('0050', '1101', '1102', '1103', '1104')):
    days = pd.bdate_range('2022-01-03', periods=n)
    prices = pd.DataFrame(100., index=days, columns=list(symbols))
    return prices, pd.DataFrame(True, index=days, columns=list(symbols))


def event(prices, index=1, members=('1101',), name='e1', priority=1.):
    return {'event_id': name, 'signal_date': str((prices.index[index] - pd.Timedelta(days=1)).date()),
            'entry_date': str(prices.index[index].date()), 'members': list(members), 'priority': priority}


def run(prices, flags, events=(), **kwargs):
    return simulate_baskets(prices, flags, list(events), start=prices.index[0], end=prices.index[-1], **kwargs)


def test_benchmark_cost_is_exact_buy_and_sell_and_nav_conserves():
    prices, flags = data()
    result = run(prices, flags, mode='benchmark', slippage=.003)
    buy = COMMISSION + .003
    expected = (1 - buy - .001) / (1 + buy)
    assert result['summary']['final_nav'] == pytest.approx(expected)
    assert result['summary']['total_cost'] == pytest.approx(1 - expected)
    assert result['summary']['trade_count'] == 2
    assert result['summary']['final_cash'] == pytest.approx(expected)
    assert result['summary']['final_liquidation_complete']
    assert sum(row['cost'] for row in result['executions']) == pytest.approx(1 - expected)
    assert all(row['cash'] >= 0 for row in result['curve'])


def test_basket_budget_includes_costs_and_members_receive_equal_cash():
    prices, flags = data()
    result = run(prices, flags, [event(prices, members=('1101', '1102'))], slots=3, horizon=3)
    cohort = result['cohorts'][0]
    members = cohort['member_entries']
    assert cohort['funding_budget'] == pytest.approx(cohort['entry_nav'] / 3)
    assert members[0]['cash_budget'] == members[1]['cash_budget']
    assert sum(member['entry_notional'] + member['buy_cost'] for member in members) == pytest.approx(
        cohort['funding_budget'] - cohort['benchmark_sell_cost'])
    assert cohort['realized_entry_weight'] < 1 / 3
    assert cohort['exit_date'] == str(prices.index[4].date())
    assert cohort['holding_sessions'] == 3
    assert cohort['cycle_net_pnl'] == pytest.approx(-cohort['total_cost'])
    assert all(item['net_pnl'] < 0 for item in cohort['member_results'])
    assert result['summary']['total_return'] == pytest.approx(-result['summary']['total_cost'])


@pytest.mark.parametrize('signal_offset', [0, 1])
def test_entry_must_strictly_follow_signal(signal_offset):
    prices, flags = data()
    supplied = event(prices)
    supplied['signal_date'] = str(prices.index[1 + signal_offset].date())
    with pytest.raises(ValueError, match='strictly after'):
        run(prices, flags, [supplied])


def test_nontrading_entry_is_rejected_without_shifting_and_execution_starts_next_day():
    prices, flags = data()
    supplied = event(prices, 4)
    supplied['signal_date'] = '2022-01-07'
    supplied['entry_date'] = '2022-01-08'
    valid = event(prices, 6, name='valid')
    result = run(prices, flags, [supplied, valid])
    assert result['rejections'][0]['reason'] == 'entry_date_not_trading_session'
    fills = [item for item in result['executions'] if item['event_id'] == 'valid']
    assert min(item['date'] for item in fills) == valid['entry_date']
    assert all(item['date'] > valid['signal_date'] for item in fills)


def test_one_untradable_member_rejects_entire_basket_without_substitution():
    prices, flags = data()
    flags.loc[prices.index[1], '1102'] = False
    result = run(prices, flags, [event(prices, members=('1101', '1102'))])
    assert result['summary']['entered_cohorts'] == 0
    assert result['rejections'][0]['reason'] == 'entry_instruments_not_tradable'
    assert {fill['stock_id'] for fill in result['executions']} == {'0050'}


def test_missing_price_never_executes_even_when_flag_true():
    prices, flags = data()
    prices.loc[prices.index[1], '1101'] = np.nan
    result = run(prices, flags, [event(prices)])
    assert result['rejections'][0]['reason'] == 'entry_instruments_not_tradable'
    assert result['summary']['entered_cohorts'] == 0


def test_slots_and_overlap_are_deterministic_and_rejections_do_not_queue():
    prices, flags = data()
    entries = [event(prices, members=(sid,), name=name, priority=priority)
               for sid, name, priority in [('1101', 'a', 3), ('1102', 'b', 2),
                                            ('1103', 'c', 1), ('1104', 'd', 0)]]
    entries.append(event(prices, 2, members=('1101', '1104'), name='overlap'))
    result = run(prices, flags, entries[::-1], slots=3, horizon=3)
    assert result['summary']['peak_active_cohorts'] == 3
    assert [cohort['event_id'] for cohort in result['cohorts']] == ['a', 'b', 'c']
    assert {item['event_id']: item['reason'] for item in result['rejections']} == {
        'd': 'slots_full', 'overlap': 'overlapping_member'}
    assert all(row['active_weight'] <= 1 and row['cash'] >= 0 for row in result['curve'])
    assert not any(fill['event_id'] in {'d', 'overlap'} for fill in result['executions'])


def test_equal_priority_tie_breaks_by_event_id():
    prices, flags = data()
    entries = [event(prices, members=('1102',), name='z'), event(prices, name='a')]
    result = run(prices, flags, entries, slots=1)
    assert result['cohorts'][0]['event_id'] == 'a'
    assert result['rejections'][0]['event_id'] == 'z'


def test_due_exit_runs_before_same_day_entry_and_all_members_wait_together():
    prices, flags = data()
    flags.loc[prices.index[3], '1102'] = False
    entries = [event(prices, members=('1101', '1102')),
               event(prices, 4, members=('1101',), name='second')]
    result = run(prices, flags, entries, slots=1, horizon=2)
    first, second = result['cohorts']
    assert first['exit_date'] == second['entry_date'] == str(prices.index[4].date())
    assert first['blocked_exit_sessions'] == 1
    assert first['blocked_exits'][0]['stock_ids'] == ['1102']
    assert not any(fill['date'] == str(prices.index[3].date()) for fill in result['executions'])
    assert result['summary']['completed_cohorts'] == 2


def test_untradable_benchmark_blocks_whole_conversion():
    prices, flags = data()
    flags.loc[prices.index[3], '0050'] = False
    result = run(prices, flags, [event(prices)], horizon=2)
    assert result['cohorts'][0]['exit_date'] == str(prices.index[4].date())
    assert result['cohorts'][0]['blocked_exits'][0]['stock_ids'] == ['0050']


def test_missing_intermediate_price_is_marked_forward_but_not_used_for_exit():
    prices, flags = data()
    prices.loc[prices.index[2:5], '1101'] = np.nan
    prices.loc[prices.index[5]:, '1101'] = 110.
    result = run(prices, flags, [event(prices)], horizon=2)
    assert result['cohorts'][0]['exit_date'] == str(prices.index[5].date())
    assert result['cohorts'][0]['blocked_exit_sessions'] == 2
    assert result['curve'][2]['nav'] == pytest.approx(result['curve'][1]['nav'])
    assert result['curve'][4]['nav'] == pytest.approx(result['curve'][1]['nav'])
    assert result['curve'][5]['market_pnl'] > 0


def test_final_untradable_member_preserves_whole_basket_and_marked_nav():
    prices, flags = data()
    prices.loc[prices.index[-1], '1102'] = np.nan
    result = run(prices, flags, [event(prices, members=('1101', '1102'))], horizon=63)
    summary = result['summary']
    assert not summary['final_liquidation_complete']
    assert summary['final_nav_is_marked']
    assert summary['completed_cohorts'] == 0
    assert {item['stock_id'] for item in summary['unliquidated_positions']} == {'1101', '1102'}
    assert summary['final_cash'] > 0
    assert summary['final_nav'] == pytest.approx(summary['final_cash'] + sum(
        item['marked_value'] for item in summary['unliquidated_positions']))
    assert result['cohorts'][0]['status'] == 'open'
    assert result['cohorts'][0]['exit_date'] is None
    assert not any(fill['stock_id'] == '1101' and fill['side'] == 'sell' for fill in result['executions'])
    stale = next(item for item in summary['unliquidated_positions'] if item['stock_id'] == '1102')
    assert stale['mark_date'] == str(prices.index[-2].date())


def test_final_blocked_benchmark_is_not_invented_cash():
    prices, flags = data()
    flags.loc[prices.index[-1], '0050'] = False
    result = run(prices, flags, mode='benchmark')
    assert not result['summary']['final_liquidation_complete']
    assert result['summary']['unliquidated_position_count'] == 1
    assert result['summary']['final_cash'] == pytest.approx(0)
    assert len(result['executions']) == 1


def test_ninety_nine_percent_stock_loss_is_not_clipped():
    prices, flags = data()
    prices.loc[prices.index[2]:, '1101'] = 1.
    result = run(prices, flags, [event(prices)], slots=1, horizon=2)
    assert result['summary']['total_return'] < -.99
    assert result['cohorts'][0]['member_results'][0]['net_return'] < -.99
    assert result['summary']['final_nav'] > 0


def test_future_price_changes_do_not_change_past_fills_or_nav():
    prices, flags = data(30)
    supplied = event(prices)
    original = run(prices, flags, [supplied], horizon=15)
    changed_prices = prices.copy()
    changed_prices.iloc[15:, 1:] *= 1.4
    changed = run(changed_prices, flags, [supplied], horizon=15)
    assert original['curve'][:15] == changed['curve'][:15]
    cutoff = str(prices.index[15].date())
    assert [item for item in original['executions'] if item['date'] < cutoff] == [
        item for item in changed['executions'] if item['date'] < cutoff]


def test_duplicate_ids_and_non_boolean_flags_raise():
    prices, flags = data()
    with pytest.raises(ValueError, match='Duplicate event_id'):
        run(prices, flags, [event(prices), event(prices)])
    with pytest.raises(ValueError, match='aligned boolean'):
        run(prices, flags.astype(float))


def test_every_execution_cash_change_and_tax_are_reconcilable():
    prices, flags = data()
    prices['1101'] = np.linspace(100, 120, len(prices))
    result = run(prices, flags, [event(prices)], horizon=3)
    for fill in result['executions']:
        cash_change = fill['cash_after'] - fill['cash_before']
        expected = (-fill['notional'] - fill['cost'] if fill['side'] == 'buy'
                    else fill['notional'] - fill['cost'])
        assert cash_change == pytest.approx(expected)
        assert fill['cost'] == pytest.approx(fill['commission'] + fill['slippage'] + fill['sell_tax'])
        if fill['side'] == 'sell':
            assert fill['sell_tax'] == pytest.approx(fill['notional'] * (.001 if fill['stock_id'] == '0050' else .003))
    assert result['summary']['final_nav'] == pytest.approx(
        1 + sum(row['market_pnl'] for row in result['curve']) - result['summary']['total_cost'])
