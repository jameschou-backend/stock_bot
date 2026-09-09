import numpy as np
import pandas as pd
import pytest

from skills.theme_research import COMMISSION, replay_event


def quotes(values=None):
    days = pd.bdate_range('2025-01-02', periods=10)
    frame = pd.DataFrame({'2408': values or [100.] * 10, '0050': [100.] * 10}, index=days)
    return frame, frame.notna()


def test_date_only_event_cannot_buy_same_day_and_keeps_full_fixed_window():
    close, flags = quotes()
    result = replay_event(close, flags, ['2408'], '2025-01-03', horizon=4)
    assert result['summary']['start'] == '2025-01-06'
    assert result['summary']['end'] == '2025-01-10'
    assert result['trades'][0]['signal_date'] == '2025-01-03'
    with pytest.raises(ValueError, match='full fixed event window'):
        replay_event(close, flags, ['2408'], '2025-01-10', horizon=126)


def test_costs_tax_and_initial_fee_are_included_in_return_and_drawdown():
    close, flags = quotes()
    stock = replay_event(close, flags, ['2408'], close.index[0], horizon=4)
    etf = replay_event(close, flags, ['0050'], close.index[0], horizon=4)
    expected = (1-.003)*(1-COMMISSION-.003)/((1+.003)*(1+COMMISSION))-1
    assert stock['summary']['total_return'] == pytest.approx(expected)
    assert stock['summary']['max_drawdown'] == pytest.approx(expected)
    assert stock['summary']['cost_per_initial_capital'] == pytest.approx(-expected)
    assert etf['summary']['total_return'] > stock['summary']['total_return']


def test_stop_fills_next_close_and_retries_blocked_exit_even_after_recovery():
    close, flags = quotes([100, 100, 80, 60, 95, 150, 150, 150, 150, 150])
    flags.iloc[3, 0] = False
    result = replay_event(close, flags, ['2408'], close.index[0], horizon=6, policy='risk_exit')
    sell = result['trades'][1]
    assert result['signals'][0]['signal_date'] == str(close.index[2].date())
    assert sell['date'] == str(close.index[4].date())
    assert sell['reference_price'] == 95
    assert result['summary']['blocked_exit_days'] == 1
    assert len(result['trades']) == 2  # No re-entry after recovery.


def test_trailing_stop_uses_only_past_closes_and_future_changes_do_not_rewrite_orders():
    close, flags = quotes([100, 100, 150, 115, 110, 120, 130, 130, 130, 130])
    before = replay_event(close, flags, ['2408'], close.index[0], horizon=6, policy='risk_exit')
    assert before['signals'][0]['reason'] == 'trailing_stop'
    assert before['trades'][1]['date'] == str(close.index[4].date())
    close.iloc[5:, 0] = 1000
    after = replay_event(close, flags, ['2408'], close.index[0], horizon=6, policy='risk_exit')
    assert before['signals'] == after['signals']
    assert before['trades'] == after['trades']


def test_blocked_entry_retains_budget_and_never_reallocates_or_buys_later():
    close, flags = quotes()
    flags.iloc[1, 0] = False
    result = replay_event(close, flags, ['2408', '0050'], close.index[0], horizon=4)
    assert result['summary']['blocked_entries'] == 1
    assert all(t['stock_id'] == '0050' for t in result['trades'])
    assert result['per_stock'][0]['pnl_contribution'] == 0
    assert result['curve'][1]['cash'] == .5
    assert sum(r['pnl_contribution'] for r in result['per_stock']) == pytest.approx(result['summary']['total_return'])


def test_missing_held_price_stays_marked_and_final_locked_bar_is_not_liquidated():
    close, flags = quotes([100, 100, 90, np.nan, 95, np.nan, 100, 100, 100, 100])
    result = replay_event(close, flags, ['2408'], close.index[0], horizon=4)
    assert result['summary']['held_missing_price_days'] == 2
    assert result['summary']['unliquidated_positions'] == 1
    assert len(result['trades']) == 1
    assert result['curve'][-1]['equity'] == result['curve'][-2]['equity']
    assert result['per_stock'][0]['unliquidated']


def test_large_adjusted_jump_is_reported_while_held_not_before_entry():
    close, flags = quotes([20, 100, 180, 180, 180, 180, 180, 180, 180, 180])
    result = replay_event(close, flags, ['2408'], close.index[0], horizon=4)
    assert len(result['large_move_exposures']) == 1
    assert result['large_move_exposures'][0]['adjusted_return'] == pytest.approx(.8)
