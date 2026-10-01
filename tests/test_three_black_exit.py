from copy import deepcopy
import pandas as pd
import pytest

from skills.three_black_exit import ThreeBlackControl, ThreeBlackSignals, audit_three_black


def inputs():
    days = pd.bdate_range('2020-01-01', periods=20)
    close = [100.]*20
    close[7:10] = [99., 98., 97.]
    adjusted = pd.DataFrame({'1101': close}, index=days)
    quotes = pd.DataFrame(dict(date=days, stock_id='1101', close=close,
                               open=[v+1 for v in close], high=[v+2 for v in close],
                               low=[v-2 for v in close], volume=1000))
    return days, adjusted, quotes


def test_three_black_requires_three_held_bars_and_four_descending_closes():
    days, adjusted, quotes = inputs()
    signals = ThreeBlackSignals(adjusted, quotes, days)
    assert signals.exits(10, 7, '1101')
    assert not signals.exits(10, 8, '1101')
    assert not signals.exits(9, 6, '1101')
    adjusted.iloc[6] = 98.  # First black candle rose compared with the previous close.
    assert not ThreeBlackSignals(adjusted, quotes, days).exits(10, 7, '1101')


@pytest.mark.parametrize('column,value', [('open', 98.), ('open', 97.), ('volume', 0),
                                         ('open', float('nan')), ('close', float('inf'))])
def test_doji_green_missing_or_nontrading_candle_breaks_the_sequence(column, value):
    days, adjusted, quotes = inputs()
    quotes.loc[8, column] = value
    assert not ThreeBlackSignals(adjusted, quotes, days).exits(10, 7, '1101')


def test_execution_day_and_later_observations_cannot_change_the_exit():
    days, adjusted, quotes = inputs()
    original = ThreeBlackSignals(adjusted, quotes, days).exits(10, 7, '1101')
    adjusted.iloc[10:] = .01
    quotes.loc[10:, ['open', 'close', 'volume']] = 999999.
    assert original and ThreeBlackSignals(adjusted, quotes, days).exits(10, 7, '1101')


def test_mechanical_raw_price_drop_and_roundoff_are_not_a_lower_adjusted_close():
    days, adjusted, quotes = inputs()
    adjusted.iloc[7] = adjusted.iloc[6, 0]-1e-13
    assert not ThreeBlackSignals(adjusted, quotes, days).exits(10, 7, '1101')
    adjusted.iloc[7] = 101.  # Raw close fell while the adjusted close rose.
    assert not ThreeBlackSignals(adjusted, quotes, days).exits(10, 7, '1101')


def test_impossible_open_in_a_used_candle_is_blocked_not_silently_ignored():
    days, adjusted, quotes = inputs()
    quotes.loc[8, 'open'] = 999.
    with pytest.raises(ValueError, match='impossible OHLC'):
        ThreeBlackSignals(adjusted, quotes, days).exits(10, 7, '1101')


def test_independent_audit_rejects_same_day_sale_wrong_signal_and_duplicate_decision():
    days, adjusted, quotes = inputs()
    signals = ThreeBlackSignals(adjusted, quotes, days)
    date = lambda i: str(days[i].date())
    account = dict(cohorts=[dict(event_id='e', stock_id='1101', entry_date=date(7))],
        black_log=[dict(event_id='e', stock_id='1101', date=date(10), signal_date=date(9), trigger=True)],
        trades=[dict(event_id='e', stock_id='1101', side='sell', reason='three_black',
                     date=date(10), signal_date=date(9))])
    assert audit_three_black(account, signals)['exits'] == 1
    for change in ('same_day', 'wrong_signal', 'duplicate'):
        bad = deepcopy(account)
        if change == 'same_day': bad['trades'][0]['date'] = date(9)
        elif change == 'wrong_signal': bad['black_log'][0]['trigger'] = False
        else: bad['black_log'].append(deepcopy(bad['black_log'][0]))
        with pytest.raises(ValueError): audit_three_black(bad, signals)


@pytest.mark.parametrize('prior_reason', [None, 'loss12', 'time63'])
def test_new_exit_runs_after_original_protection_and_latches_the_next_session(prior_reason):
    days, adjusted, quotes = inputs()
    signals = ThreeBlackSignals(adjusted, quotes, days)

    class ExistingAccount:
        def __init__(self):
            self.days, self.positions = days, {day: i for i, day in enumerate(days)}
            self.holdings = {'1101': dict(event_id='e', qty=1200)}
            self.exit_states = {'e': dict(entry_index=7, trigger_reason=None)}
            self.opening_limit = 1000000.
            self.plans = []

        def corporate_day(self, day):
            if prior_reason:
                self.exit_states['e']['trigger_reason'] = prior_reason
            return 123.

        def _plan(self, *args):
            self.plans.append(args)

    class Account(ThreeBlackControl, ExistingAccount):
        pass

    engine = Account(drawdown_arm='three_black', black_signals=signals)
    assert engine.corporate_day(days[10]) == 123.
    state = engine.exit_states['e']
    if prior_reason:
        assert state['trigger_reason'] == prior_reason and not engine.plans
    else:
        assert state['trigger_reason'] == 'three_black'
        assert state['signal_date'] == str(days[9].date())
        assert state['target_date'] == str(days[10].date())
        assert engine.holdings['1101']['due_index'] == 10 and len(engine.plans) == 1
        engine.corporate_day(days[11])
        assert len(engine.plans) == 1  # Existing exit execution owns partial-fill retries.
