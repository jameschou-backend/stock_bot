"""Real scanner rules feed immutable T+1 cash-account candidate inputs."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from scripts.prepare_strategy_account_comparison import (
    PARAMETERS, digest, load_candidates, prepare_entries, write_inputs,
)


def market_fixture():
    # Explicit observed calendar: New Year's Day is not an invented next session.
    days = pd.bdate_range('2023-10-02', '2024-03-15').difference(pd.DatetimeIndex(['2024-01-01']))
    close = 200. - np.arange(len(days)) * .5
    close[days.get_loc('2023-12-29')] += 15.
    close[-1] += 15.
    rows = []
    for sid, volume in [('0050', 3_000_000), ('2317', 1_000_000),
                        ('2330', 1_000_000), ('2454', 2_000_000), ('3500', 1)]:
        for day, value in zip(days, close):
            rows.append(dict(date=day, stock_id=sid, open=value-1, high=value+1,
                             low=value-2, close=value, volume=volume,
                             amount=value*volume, adjusted_close=value*.5,
                             quality=True, eligible=True))
    return pd.DataFrame(rows), days


def prepare(bars=None, days=None, **kwargs):
    if bars is None:
        bars, days = market_fixture()
    return prepare_entries(bars, days, start='2024-01-02', end='2024-03-15', **kwargs)


def test_real_rsi_rule_uses_previous_year_signal_and_observed_t_plus_one():
    payload, days = prepare()
    boundary = [r for r in payload['entries'] if r['signal_date'] == '2023-12-29']
    assert [r['members'][0] for r in boundary] == ['2454', '2317', '2330']
    assert all(r['entry_date'] == '2024-01-02' for r in boundary)
    assert payload['counts']['boundary_before_start'] == 3
    assert payload['signal_start'] == '2023-12-29'
    assert payload['source_history_start'] == '2023-10-02'
    assert payload['parameters'] == PARAMETERS
    for row in payload['entries']:
        assert days.index(row['entry_date']) == days.index(row['signal_date']) + 1
        assert row['leader_evidence']['previous_rsi14'] < 30 <= row['leader_evidence']['rsi14']
        assert row['leader_evidence']['information_cutoff'] == row['signal_date']


def test_terminal_close_remains_pending_without_fabricating_next_session():
    payload, days = prepare()
    pending = payload['pending_entries']
    assert len(pending) == 3
    assert all(r['signal_date'] == days[-1] and r['entry_date'] is None for r in pending)
    assert not any(r['signal_date'] == days[-1] for r in payload['entries'])
    assert payload['counts']['executable'] == len(payload['entries'])
    assert payload['counts']['pending'] == len(pending)


def test_candidate_priority_is_raw_twenty_day_amount_and_stock_code_tiebreak():
    bars, days = market_fixture()
    payload, _ = prepare(bars, days)
    rows = [r for r in payload['entries'] if r['signal_date'] == '2023-12-29']
    for row in rows:
        sid = row['members'][0]
        history = bars[bars.stock_id.eq(sid) & bars.date.le('2023-12-29')].tail(20)
        expected = (history.close*history.volume).mean()
        assert row['priority'] == pytest.approx(expected)
        assert row['leader_evidence']['amount20'] == pytest.approx(expected)
    assert rows[1]['priority'] == rows[2]['priority']
    assert rows[1]['members'][0] < rows[2]['members'][0]


def test_etf_and_insufficient_turnover_cannot_be_candidates():
    payload, _ = prepare()
    assert {r['members'][0] for r in payload['entries']+payload['pending_entries']} == {'2317', '2330', '2454'}
    assert all(not r['members'][0].startswith('0') for r in payload['entries'])


def test_future_rows_and_prices_do_not_change_prior_candidates_or_pending_status():
    bars, days = market_fixture()
    cutoff = '2024-01-02'
    expected, calendar = prepare_entries(bars, days, start=cutoff, end=cutoff)
    future = bars.date.gt(cutoff)
    bars.loc[future, ['open', 'high', 'low', 'close', 'adjusted_close']] = 10_000
    bars.loc[future, 'volume'] = 0
    actual, actual_calendar = prepare_entries(bars, days, start=cutoff, end=cutoff)
    assert actual == expected
    assert actual_calendar == calendar
    assert calendar[-1] == cutoff


def test_unknown_previous_rule_is_not_relabelled_as_a_known_first_signal():
    bars, days = market_fixture()
    # Wilder has a long flat history: RSI is unknown (0/0) until one decline.
    # Today's RSI reclaim is known, but yesterday's complete crossing rule was
    # unknown because its own previous RSI was missing. It is not a first event.
    for field in ['open', 'high', 'low', 'close']:
        bars[field] = 100.
    bars['adjusted_close'] = 50.
    for day, price in [('2023-12-28', 90.), ('2023-12-29', 110.)]:
        at = bars.date.eq(day)
        bars.loc[at, ['open', 'high', 'low', 'close']] = price
        bars.loc[at, 'adjusted_close'] = price*.5
    bars['amount'] = bars.close*bars.volume
    payload, _ = prepare(bars, days)
    assert not any(r['signal_date'] == '2023-12-29' for r in payload['entries'])
    assert payload['counts']['matched_prior_unknown'] == 3


def test_missing_previous_market_observation_does_not_compress_history():
    bars, days = market_fixture()
    bars = bars[~(bars.stock_id.eq('2330') & bars.date.eq('2023-12-28'))]
    payload, _ = prepare(bars, days)
    boundary = [r for r in payload['entries'] if r['signal_date'] == '2023-12-29']
    assert [r['members'][0] for r in boundary] == ['2454', '2317']


def test_stock_batching_preserves_exact_rules_counts_and_order():
    one, days = prepare(batch_size=1)
    many, other_days = prepare(batch_size=128)
    assert one == many and days == other_days


def test_frozen_output_and_source_hashes_are_required_for_loading(tmp_path):
    payload, days = prepare()
    source = tmp_path/'source.json'
    source.write_text('{"frozen":true}')
    output = tmp_path/'inputs'
    manifest = write_inputs(output, payload, days, {'source.json':digest(source)}, root=tmp_path)
    actual, actual_days, actual_manifest = load_candidates(output, root=tmp_path)
    assert actual['entries'] == payload['entries'] and actual_days == days
    assert actual_manifest == manifest
    source.write_text('{"frozen":false}')
    with pytest.raises(ValueError, match='source SHA mismatch'):
        load_candidates(output, root=tmp_path)
    source.write_text('{"frozen":true}')
    with (output/'rsi-entries.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='output SHA mismatch'):
        load_candidates(output, root=tmp_path)
    with pytest.raises(ValueError, match='new empty'):
        write_inputs(output, payload, days, {'source.json':digest(source)}, root=tmp_path)


def test_loader_rejects_wrong_entry_timing_even_with_consistent_file_hashes(tmp_path):
    payload, days = prepare()
    changed = deepcopy(payload)
    changed['entries'][0]['entry_date'] = changed['entries'][0]['signal_date']
    output = tmp_path/'bad-inputs'
    write_inputs(output, changed, days, {}, root=tmp_path)
    with pytest.raises(ValueError, match=r'observed T\+1'):
        load_candidates(output, root=tmp_path)


def test_bad_endpoints_codes_and_calendar_are_explicit_errors():
    bars, days = market_fixture()
    with pytest.raises(ValueError, match='observed market sessions'):
        prepare_entries(bars, days, start='2024-01-01', end='2024-03-15')
    with pytest.raises(ValueError, match='preceding observed'):
        prepare_entries(bars, days, start=str(days[0].date()), end='2024-03-15')
    bars.loc[0, 'stock_id'] = '00631L'
    with pytest.raises(ValueError, match='four-digit'):
        prepare(bars, days)
    bars, days = market_fixture()
    with pytest.raises(ValueError, match='ordered unique'):
        prepare(bars, days[::-1])
