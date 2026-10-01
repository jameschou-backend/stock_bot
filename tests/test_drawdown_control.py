from types import SimpleNamespace
import pandas as pd
import pytest
from skills.drawdown_control import CoolingState, weak_exit, audit_drawdown


def test_two_distinct_stops_block_exactly_ten_later_sessions():
    state = CoolingState()
    assert not state.observe(5, [('a', 4), ('a', 4)])
    assert state.observe(7, [('b', 6)])
    assert state.observe(16, [])
    assert not state.observe(17, [])
    assert not state.observe(18, [('a', 17)])  # A later partial sale is not a new loss.


def test_old_loss_expires_and_current_or_future_fills_are_forbidden():
    state = CoolingState()
    assert not state.observe(2, [('a', 1)])
    assert not state.observe(22, [('b', 21)])
    with pytest.raises(ValueError, match='earlier'):
        state.observe(23, [('c', 23)])


def test_new_stop_can_extend_but_old_cluster_cannot_roll_forever():
    state = CoolingState()
    assert state.observe(10, [('a', 8), ('b', 9)])
    assert state.observe(15, [('c', 14)])
    assert state.until == 24
    assert not state.observe(25, [])


def test_weak_exit_ignores_execution_day_and_later_prices():
    days = pd.bdate_range('2020-01-01', periods=50)
    close = pd.DataFrame({'1101': [100.]*30+[99., 98.]+[500.]*18}, index=days)
    def signals(c):
        return SimpleNamespace(adjusted_close=c, ma20=c.rolling(20).mean(),
                               relative20=pd.DataFrame({'1101': [-.1]*len(c)}, index=c.index))
    assert weak_exit(32, 28, '1101', signals(close))
    future = close.copy(); future.iloc[32:] = .01
    assert weak_exit(32, 28, '1101', signals(future))
    assert not weak_exit(32, 31, '1101', signals(close))
    close.iloc[30] = float('nan')
    assert not weak_exit(32, 28, '1101', signals(close))


def test_independent_cooling_audit_rejects_blocked_buys_and_incomplete_journal():
    days = pd.bdate_range('2020-01-01', periods=45)
    date = lambda i: str(days[i].date())
    signals = SimpleNamespace(days=days)
    entries = [{'entry_date': date(32), 'event_id': 'candidate'}]
    account = dict(settings={'drawdown_arm': 'cooldown'}, cohorts=[],
                   trades=[dict(date=date(i), side='sell', reason='loss12', event_id=eid)
                           for i, eid in ((30, 'a'), (31, 'b'), (33, 'b'))],
                   daily=[dict(date=date(i)) for i in range(28, 45)],
                   cooling_log=[dict(date=date(i), signal_date=date(i-1),
                                     blocked=32 <= i <= 41, until_index=41 if i >= 32 else -1,
                                     blocked_events=['candidate'] if i == 32 else [])
                                for i in range(28, 45)])
    assert audit_drawdown(account, signals, entries)['cooling_days_rebuilt'] == 17
    account['trades'].append(dict(date=date(32), side='buy', reason='entry', event_id='c'))
    with pytest.raises(ValueError, match='Purchase executed'):
        audit_drawdown(account, signals, entries)
    account['trades'].pop()
    account['cooling_log'].pop()
    with pytest.raises(ValueError, match='every account day'):
        audit_drawdown(account, signals, entries)


def test_mathematically_equal_average_is_not_a_breakdown():
    days = pd.bdate_range('2020-01-01', periods=40)
    close = pd.DataFrame({'1101': [100.]*40}, index=days)
    signals = SimpleNamespace(adjusted_close=close, ma20=close+1e-13,
                              relative20=pd.DataFrame({'1101': [-.1]*40}, index=days))
    assert not weak_exit(32, 28, '1101', signals)
    signals.ma20 = close+.001
    assert weak_exit(32, 28, '1101', signals)
