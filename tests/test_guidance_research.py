import json

import numpy as np
import pandas as pd
import pytest

from skills.guidance_research import COMMISSION, simulate_core


def inputs(periods=12):
    days = pd.bdate_range('2023-01-03', periods=periods)
    close = pd.DataFrame({'0050': 100., '2330': 100.}, index=days)
    return days, close, close.notna()


def event(days, index=2, event_id='one'):
    return {'event_id': event_id, 'entry_date': str(days[index].date()), 'stock_id': '2330'}


def run(close, flags, entries=(), **kwargs):
    return simulate_core(close, flags, list(entries), start=close.index[0], end=close.index[-1], **kwargs)


def test_benchmark_closed_form_includes_both_fees_and_final_etf_tax():
    _, close, flags = inputs()
    close.iloc[-1, 0] = 150.
    result = run(close, flags, mode='benchmark', slippage=.003)
    buy = COMMISSION + .003
    expected = 1.5 * (1 - buy - .001) / (1 + buy)
    assert result['summary']['final_nav'] == pytest.approx(expected)
    assert result['summary']['trade_count'] == 2
    assert result['summary']['active_trade_count'] == 0
    assert result['curve'][-1]['cash'] == pytest.approx(expected)
    assert result['curve'][-1]['benchmark_units'] == 0
    assert result['trades'] == []
    json.dumps(result, allow_nan=False)


def test_static_mix_closed_form_with_distinct_stock_and_etf_sell_tax():
    _, close, flags = inputs()
    close.iloc[-1] = [120., 80.]
    result = run(close, flags, mode='static_mix', slippage=0)
    expected = (.7 * 1.2 * (1 - COMMISSION - .001)
                + .3 * .8 * (1 - COMMISSION - .003)) / (1 + COMMISSION)
    assert result['summary']['final_nav'] == pytest.approx(expected)
    assert result['summary']['trade_count'] == 4
    assert result['summary']['active_trade_count'] == 1
    assert result['curve'][0]['active_weight'] == pytest.approx(.3)
    assert result['curve'][-1]['active_weight'] == 0


def test_replacement_is_funded_and_every_cash_movement_and_daily_nav_reconcile():
    days, close, flags = inputs()
    close['2330'] = np.linspace(100., 130., len(days))
    result = run(close, flags, [event(days)], horizon=4)
    trade = result['trades'][0]
    assert trade['entry_date'] == str(days[2].date())
    assert trade['exit_date'] == str(days[6].date())
    assert trade['funding_budget'] <= .3 * trade['entry_nav'] + 1e-15
    assert trade['stock_entry_notional'] + trade['stock_buy_cost'] <= trade['funding_budget']
    assert 0 < trade['realized_entry_weight'] < .3
    assert all(row['cash'] >= 0 for row in result['curve'])
    assert all(row['benchmark_units'] >= 0 and row['active_units'] >= 0 for row in result['curve'])
    for fill in result['executions']:
        movement = (-fill['notional'] if fill['side'] == 'buy' else fill['notional']) - fill['cost']
        assert fill['cash_after'] - fill['cash_before'] == pytest.approx(movement)
        assert fill['cost'] == pytest.approx(fill['commission'] + fill['slippage'] + fill['sell_tax'])
    curve = result['curve']
    assert curve[-1]['nav'] == pytest.approx(1 + sum(row['market_pnl'] - row['daily_cost'] for row in curve))
    assert result['summary']['total_cost'] == pytest.approx(sum(fill['cost'] for fill in result['executions']))
    assert trade['total_cost'] == pytest.approx(sum(fill['cost'] for fill in result['executions'] if fill['event_id'] == 'one'))


def test_flat_prices_lose_exactly_costs_without_clipped_or_created_cash():
    days, close, flags = inputs()
    result = run(close, flags, [event(days)], horizon=3)
    assert result['summary']['total_return'] == pytest.approx(-result['summary']['total_cost'])
    assert result['summary']['max_drawdown'] == pytest.approx(result['summary']['total_return'])
    benchmark = run(close, flags, mode='benchmark')
    assert result['summary']['total_return'] < benchmark['summary']['total_return']


def test_overlaps_never_extend_and_blocked_conversion_waits_for_both_instruments():
    days, close, flags = inputs()
    flags.loc[days[5], '0050'] = False
    flags.loc[days[6], '2330'] = False
    result = run(close, flags, [event(days), event(days, 4, 'overlap'), event(days, 6, 'blocked')], horizon=3)
    assert result['trades'][0]['exit_date'] == str(days[7].date())
    assert result['trades'][0]['holding_sessions'] == 5
    assert result['summary']['blocked_exit_sessions'] == 2
    assert [row['reason'] for row in result['rejections']] == ['overlapping_position', 'overlapping_position']


def test_nontrading_and_blocked_entries_are_rejected_without_earlier_or_later_fill():
    days, close, flags = inputs()
    flags.loc[days[2], '2330'] = False
    weekend = {'event_id': 'weekend', 'entry_date': '2023-01-07', 'stock_id': '2330'}
    result = run(close, flags, [event(days), weekend, event(days, 7, 'valid')], horizon=2)
    assert [row['event_id'] for row in result['trades']] == ['valid']
    assert result['trades'][0]['entry_date'] == str(days[7].date())
    assert {row['reason'] for row in result['rejections']} == {'entry_date_not_trading_session', 'entry_instruments_not_tradable'}


def test_intraday_entry_timestamp_is_not_rounded_back_to_an_earlier_close():
    days, close, flags = inputs()
    item = event(days)
    item['entry_date'] = str(days[2].date()) + ' 15:00:00'
    result = run(close, flags, [item])
    assert result['trades'] == []
    assert result['rejections'][0]['reason'] == 'entry_date_not_trading_session'


def test_missing_or_zero_prices_only_forward_mark_and_never_fill():
    days, close, flags = inputs()
    close.loc[days[5], '2330'] = np.nan
    close.loc[days[6], '2330'] = 0.
    result = run(close, flags, [event(days)], horizon=3)
    assert result['trades'][0]['exit_date'] == str(days[7].date())
    assert result['curve'][4]['nav'] == pytest.approx(result['curve'][5]['nav'])
    assert result['curve'][5]['nav'] == pytest.approx(result['curve'][6]['nav'])


@pytest.mark.parametrize('invalid', ['flag', 'missing', 'zero'])
def test_terminal_liquidation_requires_actual_tradable_price(invalid):
    days, close, flags = inputs()
    if invalid == 'flag':
        flags.iloc[-1, 1] = False
    else:
        close.iloc[-1, 1] = np.nan if invalid == 'missing' else 0.
    with pytest.raises(ValueError, match='Cannot liquidate all holdings'):
        run(close, flags, [event(days)], horizon=63)


def test_terminal_exit_does_not_rebuy_benchmark_or_accept_new_entry():
    days, close, flags = inputs()
    result = run(close, flags, [event(days), event(days, len(days) - 1, 'late')], horizon=63)
    assert result['trades'][0]['exit_reason'] == 'terminal_liquidation'
    assert result['trades'][0]['benchmark_buy_cost'] == 0
    assert result['rejections'][-1]['reason'] == 'terminal_session'
    assert all(fill['side'] == 'sell' for fill in result['executions'] if fill['date'] == str(days[-1].date()))
    assert result['curve'][-1]['active_units'] == 0
    assert result['curve'][-1]['benchmark_units'] == 0


def test_large_losses_are_preserved_and_future_prices_do_not_modify_past_cashflows():
    days, close, flags = inputs()
    before = run(close, flags, [event(days)], horizon=63)
    close.loc[days[7]:, '2330'] = 1.
    after = run(close, flags, [event(days)], horizon=63)
    assert after['trades'][0]['stock_net_return'] < -.99
    assert before['curve'][:7] == after['curve'][:7]
    assert after['summary']['total_return'] < -.29


def test_same_day_due_exit_precedes_new_entry_and_preserves_separate_events():
    days, close, flags = inputs()
    result = run(close, flags, [event(days), event(days, 5, 'next')], horizon=3)
    assert len(result['trades']) == 2
    assert result['trades'][0]['exit_date'] == result['trades'][1]['entry_date']
    assert result['rejections'] == []


def test_bad_alignment_and_duplicate_event_identity_fail_explicitly():
    days, close, flags = inputs()
    with pytest.raises(ValueError, match='aligned boolean flags'):
        run(close, flags.iloc[::-1])
    with pytest.raises(ValueError, match='Duplicate event_id'):
        run(close, flags, [event(days), event(days, 3)])
    flags.iloc[0, 0] = False
    with pytest.raises(ValueError, match='Initial portfolio cannot trade'):
        run(close, flags)
