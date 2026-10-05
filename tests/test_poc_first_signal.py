from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skills.candle_volume_rules import CandleVolumeSignals, EntryCandleUnavailable
from skills.poc_first_signal import POLICY, audit_first_signal, first_signal_entries


DAYS = ['2023-12-26', '2023-12-27', '2023-12-28', '2023-12-29',
        '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08']


def fixture(spec):
    """spec = [(signal-day index, stock id, red/black/doji)]."""
    days = pd.DatetimeIndex(DAYS)
    sids = sorted({sid for _, sid, _ in spec})
    raw = {k: pd.DataFrame(value, index=days, columns=sids)
           for k, value in dict(open=99., high=110., low=90., close=100., volume=1000.).items()}
    adjusted = pd.DataFrame(100., index=days, columns=sids)
    entries = []
    for i, sid, status in sorted(spec):
        raw['open'].at[days[i], sid] = dict(red=99., black=101., doji=100.)[status]
        entries.append(dict(event_id=f'{DAYS[i]}-{sid}', members=[sid],
            signal_date=DAYS[i], entry_date=DAYS[i+1], priority=.2))
    source = SimpleNamespace(days=days, adjusted=adjusted, raw=raw)
    return entries, CandleVolumeSignals(source, entries, [])


def run(entries, signals, start='2024-01-02', end='2024-01-08'):
    return first_signal_entries(entries, signals, start, end)


def empty_account():
    return dict(cohorts=[], trades=[], orders=[])


def account_for(event):
    identity = dict(event_id=event['event_id'], stock_id=event['members'][0],
                    signal_date=event['signal_date'])
    entry = dict(identity, date=event['entry_date'])
    return dict(cohorts=[dict(identity, entry_date=event['entry_date'])],
        trades=[dict(entry, side='buy', qty=123)],
        orders=[dict(entry, side='buy', filled_qty=123)],
        tick_plans=[dict(entry, side='buy')], base_tick_plans=[dict(entry, side='buy')],
        resource_plans=[entry.copy()], slot_decisions=[dict(entry, attempted=True)])


def audit(account, decisions, entries, signals, start='2024-01-02', end='2024-01-08'):
    return audit_first_signal(account, decisions, entries=entries,
        candle_signals=signals, start=start, end=end)


def test_cross_year_warmup_filters_four_existing_red_runs():
    stocks = ['6415', '6204', '3338', '3413']
    entries, signals = fixture([(i, sid, 'red') for i in (2, 3) for sid in stocks])
    selected, red, first = run(entries, signals)
    assert selected == entries[:4]  # Out-of-period warmup is preserved, not tradable.
    assert len(red) == len(first) == 4 and all(d['passed'] for d in red)
    assert not any(d['passed'] for d in first)
    assert {d['previous_session'] for d in first} == {'2023-12-28'}
    assert {d['run_start'] for d in first} == {'2023-12-28'}
    assert {d['run_sessions'] for d in first} == {2}
    assert audit(empty_account(), first, entries, signals)['first_red_signals'] == 0


def test_weekend_and_market_holiday_do_not_restart_run():
    entries, signals = fixture([(3, '1234', 'red'), (4, '1234', 'red'),
                                (5, '1234', 'red'), (7, '1234', 'red')])
    selected, red, first = run(entries, signals)
    assert selected == [entries[0], entries[3]]
    assert [d['passed'] for d in first] == [True, False, False, True]
    assert first[1]['previous_session'] == '2023-12-29'
    assert first[1]['run_start'] == '2023-12-29'
    assert first[2]['run_sessions'] == 3
    assert first[3]['run_sessions'] == 1  # 1/4 had no candidate; 1/5 is fresh.
    assert first[3]['previous_red_status'] == 'no_raw_signal'


@pytest.mark.parametrize('previous_status', ['black', 'doji'])
def test_nonred_raw_candidate_breaks_red_run(previous_status):
    entries, signals = fixture([(1, '1234', 'red'), (2, '1234', previous_status),
                                (3, '1234', 'red'), (4, '1234', 'red')])
    selected, red, first = run(entries, signals)
    assert selected == entries[:3]
    assert [d['passed'] for d in first] == [True, False]
    assert first[0]['previous_event_id'] == entries[1]['event_id']
    assert first[0]['previous_red_status'] == previous_status
    assert first[0]['run_start'] == '2023-12-29'


def test_red_gate_is_exact_original_scope_and_sources_are_unchanged():
    entries, signals = fixture([(2, '1234', 'red'), (3, '1234', 'red'),
                                (4, '1234', 'black'), (5, '1234', 'red')])
    original = deepcopy(entries)
    selected, red, first = run(entries, signals)
    _, expected = signals.filter_entries([e for e in entries if e['entry_date'] >= '2024-01-02'])
    assert red == expected
    assert first[1]['status'] == 'signal_not_red'
    assert first[1]['run_start'] is None and first[1]['run_sessions'] == 0
    assert entries == original
    selected[0]['priority'] = 999
    first[0]['event_id'] = 'changed'
    assert entries == original


def test_unfilled_first_order_does_not_authorize_next_signal_or_late_entry():
    entries, signals = fixture([(3, '1234', 'red'), (4, '1234', 'red')])
    _, _, decisions = run(entries, signals)
    zero_order = account_for(entries[0])
    zero_order['cohorts'] = []; zero_order['trades'] = []
    zero_order['orders'][0]['filled_qty'] = 0
    assert audit(zero_order, decisions, entries, signals)['resets_on_order_failure'] is False
    with pytest.raises(ValueError, match='not an authorized first-day'):
        audit(account_for(entries[1]), decisions, entries, signals)
    late = account_for(entries[0])
    late['trades'][0]['date'] = '2024-01-03'
    with pytest.raises(ValueError, match='not an authorized first-day'):
        audit(late, decisions, entries, signals)
    assert run(entries, signals)[2] == decisions


def test_future_append_and_future_prices_cannot_change_past_decisions():
    entries, signals = fixture([(3, '1234', 'red'), (4, '1234', 'red')])
    expected = run(entries, signals, end='2024-01-03')
    expanded, later = fixture([(3, '1234', 'red'), (4, '1234', 'red'), (7, '1234', 'red')])
    for frame in later.raw.values():
        frame.loc['2024-01-04':] = np.nan
    selected, red, first = run(expanded, later, end='2024-01-03')
    assert red == expected[1] and first == expected[2]
    assert selected == expected[0] + [expanded[-1]]
    audit(empty_account(), first, expanded, later, end='2024-01-03')


def test_later_research_end_keeps_old_decision_prefix_exact():
    entries, signals = fixture([(3, '1234', 'red'), (4, '1234', 'red'), (5, '1234', 'red')])
    short = run(entries, signals, end='2024-01-03')[2]
    full = run(entries, signals)[2]
    assert full[:len(short)] == short


@pytest.mark.parametrize('field,value', [('open', np.nan), ('low', 109.), ('close', 0.)])
def test_unknown_warmup_candle_is_not_a_false_signal(field, value):
    entries, signals = fixture([(2, '1234', 'red'), (3, '1234', 'red')])
    signals.raw[field].at[pd.Timestamp('2023-12-28'), '1234'] = value
    with pytest.raises(EntryCandleUnavailable):
        run(entries, signals)
    with pytest.raises(EntryCandleUnavailable):
        audit(empty_account(), [], entries, signals)


def test_missing_warmup_registry_or_duplicate_event_is_rejected():
    entries, signals = fixture([(2, '1234', 'red'), (3, '1234', 'red')])
    with pytest.raises(ValueError, match='complete candle candidate registry'):
        run(entries[1:], signals)
    with pytest.raises(ValueError, match='duplicated'):
        run([entries[0], entries[0], entries[1]], signals)
    duplicate = dict(entries[0], event_id='different-id-same-observation')
    signals.entries[duplicate['event_id']] = duplicate
    with pytest.raises(ValueError, match='stock/session candidate is duplicated'):
        run([entries[0], duplicate, entries[1]], signals)


def test_missing_market_session_or_inconsistent_calendar_is_rejected():
    entries, signals = fixture([(2, '1234', 'red'), (3, '1234', 'red')])
    signals.days = signals.days.delete(2)
    with pytest.raises(ValueError, match='calendar positions differ'):
        run(entries, signals)
    signals.positions = {day: i for i, day in enumerate(signals.days)}
    with pytest.raises(ValueError, match='candles and market calendar differ'):
        run(entries, signals)


@pytest.mark.parametrize('start,end', [('2024-01-01', '2024-01-08'),
                                     ('2024-01-02', '2024-01-07'),
                                     ('2024-01-08', '2024-01-02')])
def test_invalid_research_boundaries(start, end):
    entries, signals = fixture([(3, '1234', 'red')])
    with pytest.raises(ValueError):
        run(entries, signals, start, end)


def test_audit_rebuilds_independently_and_checks_every_buy_journal(monkeypatch):
    entries, signals = fixture([(3, '1234', 'red')])
    decisions = run(entries, signals)[2]
    monkeypatch.setattr(signals, 'filter_entries', lambda _: pytest.fail('audit called production gate'))
    account = account_for(entries[0])
    result = audit(account, decisions, entries, signals)
    assert result['policy'] == POLICY and result['decision_reconstruction_verified']
    assert result['journals_checked'] == {k: 1 for k in ['cohorts', 'buy_trades', 'orders',
        'tick_plans', 'base_tick_plans', 'resource_plans', 'slot_decisions']}
    for key in ['orders', 'tick_plans', 'base_tick_plans', 'resource_plans', 'slot_decisions']:
        corrupt = deepcopy(account)
        corrupt[key][0]['signal_date'] = '2023-12-28'
        with pytest.raises(ValueError, match='not an authorized first-day'):
            audit(corrupt, decisions, entries, signals)


@pytest.mark.parametrize('field,value', [('passed', False), ('previous_red_qualified', True),
    ('run_start', '2023-12-28'), ('previous_session', '2023-12-27'),
    ('event_id', 'forged'), ('stock_id', '9999'), ('run_sessions', 2),
    ('passed', 1), ('portfolio_state_used', 0), ('run_sessions', 1.0)])
def test_tampered_decision_evidence_rejected(field, value):
    entries, signals = fixture([(3, '1234', 'red')])
    decisions = run(entries, signals)[2]
    decisions[0][field] = value
    with pytest.raises(ValueError, match='decision ledger differs'):
        audit(empty_account(), decisions, entries, signals)


def test_missing_extra_or_reordered_decisions_are_rejected():
    entries, signals = fixture([(3, '1234', 'red'), (3, '5678', 'red')])
    decisions = run(entries, signals)[2]
    for bad in [decisions[:1], decisions + decisions[:1], decisions[::-1]]:
        with pytest.raises(ValueError, match='decision ledger differs'):
            audit(empty_account(), bad, entries, signals)
