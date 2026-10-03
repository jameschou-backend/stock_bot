from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skills.candle_volume_rules import (
    CandleVolumeExit, CandleVolumeSignals, EntryCandleUnavailable,
    audit_candle_volume, audit_entry_gate,
)
from skills.three_black_exit import ThreeBlackSignals


def sample(actions=()):
    days = pd.bdate_range('2024-01-01', periods=16)
    raw = {k: pd.DataFrame(v, index=days, columns=['1234'])
           for k, v in dict(open=99., high=110., low=90., close=100., volume=400.).items()}
    raw['volume'].iloc[3, 0] = 1000.
    adjusted = pd.DataFrame(100., index=days, columns=['1234'])
    adjusted.iloc[4, 0], adjusted.iloc[5, 0] = 99., 98.
    event = dict(event_id='a', members=['1234'], signal_date=str(days[3].date()),
                 entry_date=str(days[4].date()), priority=.2)
    black = SimpleNamespace(days=days, adjusted=adjusted, raw=raw)
    signals = CandleVolumeSignals(black, [event], actions)
    return signals, event


def decision(signals, index=6, entry=4, mode='dry'):
    return signals.evaluate(index, entry, '1234', signals.entries['a']['signal_date'], mode)


def logged(signals, index, eid='a', mode='dry'):
    event = signals.entries[eid]
    entry = signals.positions[pd.Timestamp(event['entry_date'])]
    result = signals.evaluate(index, entry, '1234', event['signal_date'], mode)
    return dict(event_id=eid, stock_id='1234', date=str(signals.days[index].date()),
                signal_date=result['decision_date'], **result)


def account_fixture(signals, event, mode='dry'):
    logs = [logged(signals, i, mode=mode) for i in (5, 6)]
    buy = dict(event_id='a', stock_id='1234', side='buy', qty=1250,
               date=event['entry_date'], signal_date=event['signal_date'], reason='entry')
    sell = dict(event_id='a', stock_id='1234', side='sell', qty=1000,
                date=str(signals.days[6].date()), signal_date=str(signals.days[5].date()),
                reason='volume_'+mode)
    return dict(cohorts=[dict(event_id='a', stock_id='1234', entry_date=event['entry_date'],
                              signal_date=event['signal_date'])],
                trades=[buy, sell], volume_exit_log=logs, exit_decisions=[],
                black_log=[{k: row[k] for k in ('event_id', 'date')} | {'trigger': False} for row in logs])


def test_red_gate_only_signal_day_and_preserves_order_identity():
    signals, event = sample()
    signals.raw['open'].iloc[4:, 0] = 109.  # Entry and later candles are black.
    original = deepcopy(event)
    kept, rows = signals.filter_entries([event])
    assert kept == [event] == [original]
    assert kept[0] is not event
    assert rows[0]['status'] == 'red'
    signals.raw['open'].iloc[3, 0] = 101.
    signals.raw['open'].iloc[4, 0] = 95.
    assert signals.filter_entries([event])[0] == []
    assert signals.filter_entries([event])[1][0]['status'] == 'black'
    signals.raw['open'].iloc[3, 0] = 100.
    assert signals.filter_entries([event])[1][0]['status'] == 'doji'


@pytest.mark.parametrize('field,value', [('open', np.nan), ('low', 105.), ('high', 95.), ('close', 0.)])
def test_unknown_red_gate_stops_with_explicit_ledger(field, value):
    signals, event = sample()
    signals.raw[field].iloc[3, 0] = value
    with pytest.raises(EntryCandleUnavailable) as exc:
        signals.filter_entries([event])
    assert exc.value.decisions[0]['passed'] is None
    assert exc.value.decisions[0]['status'] == 'unknown_or_invalid_ohlc'


def test_entry_is_exact_next_market_session_and_identity_cannot_change():
    signals, event = sample()
    wrong = dict(event, entry_date=str(signals.days[5].date()))
    with pytest.raises(ValueError, match='one market session'):
        signals.filter_entries([wrong])
    with pytest.raises(ValueError, match='identity'):
        signals.filter_entries([dict(event, priority=99)])
    with pytest.raises(ValueError, match='duplicated'):
        signals.filter_entries([event, event])


def test_two_complete_post_entry_sessions_and_five_day_boundary():
    signals, _ = sample()
    assert decision(signals, 5)['status'] == 'insufficient_post_entry_sessions'
    assert decision(signals, 6)['trigger']
    assert decision(signals, 9)['trigger']  # E+4 close, E+5 sale.
    assert decision(signals, 10)['status'] == 'outside_early_window'
    assert decision(signals, 6)['observations'][0]['date'] == str(signals.days[4].date())
    assert decision(signals, 6)['decision_date'] == str(signals.days[5].date())
    assert decision(signals, 6)['execution_date'] == str(signals.days[6].date())


@pytest.mark.parametrize('index', [4, 5])
def test_equality_at_half_volume_is_not_contraction(index):
    signals, _ = sample()
    signals.raw['volume'].iloc[index, 0] = 500.
    result = decision(signals)
    assert result['status'] == 'evaluated' and not result['trigger']
    assert result['threshold'] == 500.


def test_volume_denominator_fixed_to_original_signal_not_later_maximum():
    signals, _ = sample()
    signals.raw['volume'].iloc[4, 0] = 10000.
    signals.raw['volume'].iloc[5:7, 0] = 600.
    result = decision(signals, 7)
    assert not result['trigger'] and result['baseline_volume'] == 1000.


@pytest.mark.parametrize('value,status', [(0., 'nontrading_volume'), (np.nan, 'unknown_or_invalid_volume'),
                                          (-1., 'unknown_or_invalid_volume'), (np.inf, 'unknown_or_invalid_volume')])
def test_missing_zero_or_invalid_session_is_never_skipped(value, status):
    signals, _ = sample()
    signals.raw['volume'].iloc[5, 0] = value
    assert decision(signals, 6)['status'] == status
    assert decision(signals, 7)['status'] == status
    assert not decision(signals, 7)['trigger']
    assert decision(signals, 8)['trigger']  # Two truly adjacent valid days can restart.


def test_weak_requires_both_adjusted_close_comparisons_and_known_prices():
    signals, _ = sample()
    assert decision(signals, mode='dry_weak')['trigger']
    signals.adjusted.iloc[5, 0] = 99.
    assert not decision(signals, mode='dry_weak')['trigger']
    signals.adjusted.iloc[4:6, 0] = [102., 101.]
    assert not decision(signals, mode='dry_weak')['trigger']
    signals.adjusted.iloc[5, 0] = np.nan
    assert decision(signals, mode='dry_weak')['status'] == 'unknown_or_invalid_adjusted_close'
    assert decision(signals, mode='dry')['trigger']


@pytest.mark.parametrize('action_index,blocks', [(2, False), (3, True), (4, True), (5, True), (6, False)])
def test_corporate_guard_uses_original_T_through_decision_not_execution(action_index, blocks):
    signals, _ = sample()
    signals.action_dates = {('1234', signals.days[action_index]), ('9999', signals.days[4])}
    result = decision(signals)
    assert result['trigger'] is (not blocks)
    assert (result['status'] == 'corporate_action_window') is blocks


def test_future_mutation_does_not_change_red_or_exit_decision_and_no_adjusted_volume():
    signals, event = sample()
    expected, gate = decision(signals, mode='dry_weak'), signals.filter_entries([event])
    for frame in signals.raw.values():
        frame.iloc[6:] = np.nan
    signals.adjusted.iloc[6:] = 10000000.
    signals.action_dates.add(('1234', signals.days[6]))
    assert decision(signals, mode='dry_weak') == expected
    assert signals.filter_entries([event]) == gate
    signals.adjusted.iloc[:6] *= .01
    result = decision(signals, mode='dry_weak')
    assert result['trigger'] and result['baseline_volume'] == expected['baseline_volume']
    assert result['threshold'] == expected['threshold']


def test_actual_fill_resets_clock_and_previous_entry_candles_do_not_count():
    signals, _ = sample()
    assert decision(signals, index=7, entry=6)['status'] == 'insufficient_post_entry_sessions'
    assert decision(signals, index=8, entry=6)['trigger']
    with pytest.raises(ValueError, match='existing earlier'):
        decision(signals, index=6, entry=6)


def test_existing_three_black_requires_entry_day_and_two_later_candles():
    signals, _ = sample()
    quotes = pd.DataFrame([dict(date=d, stock_id='1234', open=101., high=110., low=80.,
                                close=100.-i, volume=400.) for i, d in enumerate(signals.days)])
    adjusted = quotes.pivot(index='date', columns='stock_id', values='close')
    black = ThreeBlackSignals(adjusted, quotes, signals.days)
    assert not black.exits(6, 4, '1234')  # Before E+3 ignores old black candles.
    assert black.exits(7, 4, '1234')
    assert not black.exits(7, 6, '1234')  # Fresh filled event resets the clock.


class ExistingEngine:
    def __init__(self, signals, reason=None, qty=1250):
        self.days, self.positions = signals.days, signals.positions
        self.holdings = {'1234': dict(qty=qty, event_id='a', due_index=99)}
        self.exit_states = {'a': dict(entry_index=4, trigger_reason=None)}
        self.opening_limit, self.plans, self.original_reason = 1., [], reason

    def corporate_day(self, day):
        if self.original_reason:
            self.exit_states['a']['trigger_reason'] = self.original_reason
        return 17.

    def _plan(self, *args):
        self.plans.append(args)

    def run(self):
        return {'settings': {'old': True}, 'trades': [], 'value': 123.}


class Engine(CandleVolumeExit, ExistingEngine):
    pass


@pytest.mark.parametrize('reason', ['loss12', 'time63', 'three_black'])
def test_original_stop_and_three_black_priority(reason):
    signals, _ = sample()
    engine = Engine(signals, reason=reason, candle_volume_signals=signals, volume_exit_mode='dry')
    assert engine.corporate_day(signals.days[6]) == 17.
    assert engine.exit_states['a']['trigger_reason'] == reason
    assert engine.volume_exit_log == [] and engine.plans == []


def test_mixin_latches_first_close_then_partial_retry_cannot_be_cancelled():
    signals, _ = sample()
    engine = Engine(signals, candle_volume_signals=signals, volume_exit_mode='dry')
    engine.corporate_day(signals.days[6])
    state = deepcopy(engine.exit_states['a'])
    assert state['trigger_reason'] == 'volume_dry'
    assert state['signal_date'] == str(signals.days[5].date())
    assert state['target_index'] == 6 and engine.holdings['1234']['due_index'] == 6
    assert engine.plans[0][5] == 1000
    engine.holdings['1234']['qty'] = 250  # Existing sell engine handles later odd retry.
    signals.raw['volume'].iloc[6:, 0] = 99999.
    engine.corporate_day(signals.days[7])
    assert engine.exit_states['a'] == state and len(engine.plans) == 1
    assert len(engine.volume_exit_log) == 1


def test_none_account_is_exact_and_zero_fill_cannot_start_volume_exit():
    signals, _ = sample()
    engine = Engine(signals, candle_volume_signals=signals)
    assert engine.run() == ExistingEngine(signals).run()
    engine.corporate_day(signals.days[6])
    assert engine.volume_exit_log == []
    empty = Engine(signals, qty=0, candle_volume_signals=signals, volume_exit_mode='dry')
    empty.corporate_day(signals.days[6])
    assert empty.volume_exit_log == []


def test_independent_audit_rebuilds_without_production_evaluator(monkeypatch):
    signals, event = sample()
    account = account_fixture(signals, event)
    retry = dict(account['trades'][-1], qty=250, date=str(signals.days[7].date()))
    account['trades'].append(retry)
    monkeypatch.setattr(signals, 'evaluate', lambda *a: (_ for _ in ()).throw(AssertionError('oracle reused evaluator')))
    monkeypatch.setattr(signals, 'filter_entries', lambda *a: (_ for _ in ()).throw(AssertionError('oracle reused gate')))
    assert audit_candle_volume(account, signals, 'dry')['exits'] == 1
    assert audit_entry_gate(account, signals)['funded_events_checked'] == 1


@pytest.mark.parametrize('mutation,match', [
    ('missing_decision', 'missing or extra'), ('earlier_sell', 'first prior-close'),
    ('wrong_signal', 'first prior-close'), ('wrong_clock', 'first positive fill'),
    ('decision_forgery', 'scalar audit'), ('original_priority', 'priority'),
    ('repeat_latch', 'first latch'),
])
def test_scalar_audit_detects_omitted_first_decision_and_forged_exit(mutation, match):
    signals, event = sample()
    account = account_fixture(signals, event)
    if mutation == 'missing_decision':
        account['volume_exit_log'].pop(0)
    elif mutation == 'earlier_sell':
        account['trades'][-1]['date'] = str(signals.days[5].date())
    elif mutation == 'wrong_signal':
        account['trades'][-1]['signal_date'] = str(signals.days[4].date())
    elif mutation == 'wrong_clock':
        account['cohorts'][0]['entry_date'] = str(signals.days[3].date())
    elif mutation == 'decision_forgery':
        account['volume_exit_log'][-1]['baseline_volume'] = 2000.
    elif mutation == 'original_priority':
        account['exit_decisions'] = [dict(event_id='a', date=str(signals.days[6].date()), exit=True)]
    else:
        account['volume_exit_log'].append(logged(signals, 7))
        account['black_log'].append(dict(event_id='a', date=str(signals.days[7].date()), trigger=False))
    with pytest.raises(ValueError, match=match):
        audit_candle_volume(account, signals, 'dry')


def test_oracle_reentry_uses_new_event_clock_and_baseline():
    signals, event = sample()
    other = dict(event, event_id='b', signal_date=str(signals.days[7].date()),
                 entry_date=str(signals.days[8].date()))
    signals.entries['b'] = other
    signals.raw['volume'].iloc[7, 0] = 2000.
    account = account_fixture(signals, event)
    account['cohorts'].append(dict(event_id='b', stock_id='1234', signal_date=other['signal_date'], entry_date=other['entry_date']))
    account['trades'].append(dict(event_id='b', stock_id='1234', side='buy', qty=1,
                                  date=other['entry_date'], signal_date=other['signal_date'], reason='entry'))
    logs = [logged(signals, i, eid='b') for i in (9, 10)]
    account['volume_exit_log'].extend(logs)
    account['black_log'].extend(dict(event_id='b', date=r['date'], trigger=False) for r in logs)
    assert logs[0]['status'] == 'insufficient_post_entry_sessions'
    assert logs[1]['baseline_volume'] == 2000.
    assert audit_candle_volume(account, signals, 'dry')['exits'] == 2


def test_oracle_independently_rejects_black_signal_entry():
    signals, event = sample()
    account = account_fixture(signals, event)
    signals.raw['open'].iloc[3, 0] = 105.
    with pytest.raises(ValueError, match='red signal-T'):
        audit_entry_gate(account, signals)
