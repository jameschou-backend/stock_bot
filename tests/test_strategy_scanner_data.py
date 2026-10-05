from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from skills.strategy_scanner.data import load_bundle


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def seal(bundle):
    old = json.loads((bundle / 'manifest.json').read_text())
    old['files_sha256'] = {p.name: digest(p) for p in bundle.iterdir()
                          if p.name not in ('manifest.json', 'manifest.sha256')}
    write_json(bundle / 'manifest.json', old)
    (bundle / 'manifest.sha256').write_text(digest(bundle / 'manifest.json'))


@pytest.fixture
def bundle(tmp_path):
    b = tmp_path / 'inputs'; b.mkdir()
    days = pd.bdate_range('2024-01-02', periods=450)
    ids = ['0050', '2330', '9999']
    values = np.arange(450) * .05 + 100
    for name in ('close-official', 'close-quality'):
        pd.DataFrame({'date': days, **{sid: values for sid in ids}}).to_parquet(b / (name + '.parquet'), index=False)
    pd.DataFrame({'date': days, '0050': True, '2330': True, '9999': False}).to_parquet(b / 'eligibility.parquet', index=False)
    quotes = pd.DataFrame([dict(stock_id=sid, date=day, open=value-.5,
        high=value+1, low=value-1, close=value, volume=10000.)
        for sid in ids for day, value in zip(days, values)])
    quotes = quotes.loc[~((quotes.stock_id == '9999') & (quotes.date == days[-1]))]
    quotes.to_parquet(b / 'quotes-unmasked.parquet', index=False)
    pd.DataFrame([dict(stock_id=sid, name='Name'+sid, listed_date=pd.Timestamp('2000-01-01'),
                       industry='test', market='TWSE') for sid in ids[1:]]).to_parquet(b / 'companies.parquet', index=False)
    write_json(b / 'identity.json', dict(coverage_start=str(days[0].date()), coverage_end=str(days[-1].date()),
        complete_historical_universe=False, continuous_eligibility_proven=False,
        publication_time_archive_complete=False, extension_observation=dict(provisional=True,
        source='current_snapshot', official_identity_verified_through=str(days[-10].date()))))
    event = dict(event_id='liquid_universe-'+str(days[-2].date())+'-2330', signal_date=str(days[-2].date()),
                 entry_date=str(days[-1].date()), members=['2330'], priority=.2, custom={'preserved': True})
    pending = dict(event, event_id='liquid_universe-'+str(days[-1].date())+'-2330',
                   signal_date=str(days[-1].date()), entry_date=None)
    write_json(b / 'signals.json', dict(entries={'median50m': [event]}, pending_signal_file='pending-signals.json'))
    write_json(b / 'pending-signals.json', dict(schema='pending_last_close_signals_v1', entries=[pending]))
    write_json(b / 'manifest.json', dict(schema='poc_latest_input_bundle_v1',
        start=str(days[0].date()), end=str(days[-1].date()), historical_end=str(days[-10].date()),
        officially_crosschecked_extension=False, limitations=['Later price sources are not independent'],
        executable_candidate_count=1, pending_candidate_count=1,
        source_sha256={'ancestor/not-present.json': 'a'*64}, files_sha256={}))
    seal(b)
    return b, days


def run(bundle, start=None, end=None):
    b, days = bundle
    return load_bundle(b, start or str(days[-2].date()), end or str(days[-1].date()))


def mutate_parquet(bundle, name, change):
    b, _ = bundle
    f = pd.read_parquet(b / name)
    change(f)
    f.to_parquet(b / name, index=False)
    seal(b)


def test_full_universe_warmup_and_terminal_signals_survive_without_portfolio(bundle):
    value = run(bundle)
    assert len(value['calendar']) == 422
    assert value['universe'] == ['0050', '2330', '9999']
    assert len(value['bars']) == 422*3
    last = value['bars'].query('stock_id == "9999"').iloc[-1]
    assert pd.isna(last.close) and not last.quality and not last.eligible
    assert value['original_signals'][-1]['entry_date'] is None
    assert value['original_signals'][0]['custom'] == {'preserved': True}
    p = value['provenance']
    assert p['warmup_sessions_available'] == 420
    assert p['direct_input_files_verified'] is True
    assert p['upstream_source_count'] == 1 and p['upstream_source_closure_reverified'] is False
    assert p['complete_historical_universe'] is False
    assert p['identity_extension_provisional'] is True
    assert p['original_candidates_complete'] is True
    assert p['original_signal_start'] == str(bundle[1][-2].date())
    assert p['original_signal_end'] == str(bundle[1][-1].date())
    assert p['network_requests'] == 0 and p['database_mutations'] is False
    assert 'proxy_not_actual_turnover' in p['amount_basis']
    assert value['poc'] is None


def test_short_history_is_explicit_never_padded(bundle):
    _, days = bundle
    value = run(bundle, str(days[3].date()), str(days[5].date()))
    assert value['calendar'].equals(days[:6])
    assert value['provenance']['warmup_sessions_available'] == 3
    assert value['bars'].date.max() == days[5]
    assert value['original_signals'] == []


@pytest.mark.parametrize('source', ['quotes-unmasked.parquet', 'close-quality.parquet', 'eligibility.parquet', 'signals.json'])
def test_changed_direct_source_fails_even_if_not_candidate_stock(bundle, source):
    b, _ = bundle
    with (b / source).open('ab') as f:
        f.write(b'tampered')
    with pytest.raises(ValueError, match='SHA mismatch'):
        run(bundle)


def test_manifest_cannot_be_changed_without_sidecar(bundle):
    b, _ = bundle
    with (b / 'manifest.json').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='SHA mismatch'):
        run(bundle)


def test_symlink_cannot_escape_bundle(bundle, tmp_path):
    b, _ = bundle
    p = b / 'signals.json'; raw = p.read_bytes(); p.unlink()
    other = tmp_path / 'outside.json'; other.write_bytes(raw); p.symlink_to(other)
    with pytest.raises(ValueError, match='escapes bundle'):
        run(bundle)


@pytest.mark.parametrize('which', ['end_future', 'start_before', 'reversed', 'not_session'])
def test_bounds_must_be_real_observed_market_sessions(bundle, which):
    _, days = bundle
    start, end = str(days[-2].date()), str(days[-1].date())
    if which == 'end_future': end = str((days[-1]+pd.Timedelta(days=1)).date())
    if which == 'start_before': start = '2023-12-01'
    if which == 'reversed': start, end = end, start
    if which == 'not_session': start = '2024-01-06'
    with pytest.raises(ValueError): run(bundle, start, end)


def test_matrix_calendar_or_columns_cannot_silently_reindex(bundle):
    def change(f): f.loc[1, 'date'] = f.loc[0, 'date']
    mutate_parquet(bundle, 'close-quality.parquet', change)
    with pytest.raises(ValueError, match='calendar'):
        run(bundle)


def test_duplicate_quotes_are_rejected(bundle):
    b, _ = bundle
    q = pd.read_parquet(b / 'quotes-unmasked.parquet')
    pd.concat([q, q.tail(1)], ignore_index=True).to_parquet(b / 'quotes-unmasked.parquet', index=False)
    seal(b)
    with pytest.raises(ValueError, match='Duplicate raw'):
        run(bundle)


def test_unknown_eligibility_and_prices_remain_unknown_or_false(bundle):
    def change(f):
        f['2330'] = f['2330'].astype('boolean')
        f.loc[len(f)-1, '2330'] = pd.NA
    mutate_parquet(bundle, 'eligibility.parquet', change)
    def missing(f): f.loc[len(f)-1, '2330'] = np.nan
    mutate_parquet(bundle, 'close-quality.parquet', missing)
    row = run(bundle)['bars'].query('stock_id == "2330"').iloc[-1]
    assert pd.isna(row.eligible) and not row.quality
    assert pd.isna(row.source_disagreement) and not row.source_return_known
    assert row.quality_reason == 'missing_or_invalid_adjusted_price'


def test_no_arbitrary_twenty_percent_return_filter_and_disagreement_separate(bundle):
    for name in ('close-official.parquet', 'close-quality.parquet'):
        mutate_parquet(bundle, name, lambda f: f.__setitem__('2330', f['2330'].where(f.index != len(f)-1, 200.)))
    row = run(bundle)['bars'].query('stock_id == "2330"').iloc[-1]
    assert row.quality and not row.source_disagreement
    mutate_parquet(bundle, 'close-quality.parquet', lambda f: f.__setitem__('2330', f['2330'].where(f.index != len(f)-1, 210.)))
    row = run(bundle)['bars'].query('stock_id == "2330"').iloc[-1]
    assert row.quality and row.source_disagreement and row.source_return_known


@pytest.mark.parametrize('column,value,reason', [('open', 999., 'range_conflict'),
    ('volume', 0., 'share_volume'), ('volume', 2.5, 'share_volume'), ('close', np.nan, 'raw_ohlc')])
def test_bad_observations_remain_in_denominator(bundle, column, value, reason):
    b, days = bundle
    def change(f): f.loc[(f.stock_id=='2330') & (f.date==days[-1]), column] = value
    mutate_parquet(bundle, 'quotes-unmasked.parquet', change)
    row = run(bundle)['bars'].query('stock_id == "2330"').iloc[-1]
    assert not row.quality and reason in row.quality_reason


def test_candidate_identical_duplicate_allowed_conflict_rejected(bundle):
    b, _ = bundle
    pending = json.loads((b / 'pending-signals.json').read_text())
    pending['entries'].append(deepcopy(pending['entries'][0]))
    write_json(b / 'pending-signals.json', pending)
    manifest = json.loads((b / 'manifest.json').read_text())
    manifest['pending_candidate_count'] = 2
    write_json(b / 'manifest.json', manifest); seal(b)
    assert run(bundle)['provenance']['original_signal_duplicates_identical'] == 1
    pending['entries'][-1]['priority'] += .1
    write_json(b / 'pending-signals.json', pending); seal(b)
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        run(bundle)


def test_candidate_cannot_use_same_day_entry(bundle):
    b, _ = bundle
    value = json.loads((b / 'signals.json').read_text())
    row = value['entries']['median50m'][0]; row['entry_date'] = row['signal_date']
    write_json(b / 'signals.json', value); seal(b)
    with pytest.raises(ValueError, match='observed T\\+1'):
        run(bundle)


def test_future_prices_cannot_change_prior_rows(bundle):
    b, days = bundle
    end = str(days[-3].date()); start = str(days[-5].date())
    before = run(bundle, start, end)
    for name in ('close-official.parquet', 'close-quality.parquet'):
        mutate_parquet(bundle, name, lambda f: f.__setitem__('2330', f['2330'].where(f.index < len(f)-2, 999.)))
    after = run(bundle, start, end)
    pd.testing.assert_frame_equal(before['bars'], after['bars'])
    assert before['original_signals'] == after['original_signals']


@pytest.mark.parametrize('key', ['executable_candidate_count', 'pending_candidate_count'])
def test_complete_source_counts_verified_before_requested_end_truncation(bundle, key):
    b, days = bundle
    manifest = json.loads((b / 'manifest.json').read_text())
    manifest[key] += 1
    write_json(b / 'manifest.json', manifest); seal(b)
    with pytest.raises(ValueError, match='source count mismatch'):
        run(bundle, str(days[-5].date()), str(days[-3].date()))


def test_missing_source_counts_leave_original_completeness_unknown(bundle):
    b, _ = bundle
    manifest = json.loads((b / 'manifest.json').read_text())
    del manifest['executable_candidate_count']
    write_json(b / 'manifest.json', manifest); seal(b)
    assert run(bundle)['provenance']['original_candidates_complete'] is False


def test_candidate_coverage_is_not_moved_by_requested_scan_bounds(bundle):
    _, days = bundle
    result = run(bundle, str(days[-5].date()), str(days[-3].date()))
    assert result['original_signals'] == []
    assert result['provenance']['original_candidates_complete'] is True
    assert result['provenance']['original_signal_start'] == str(days[-2].date())
    assert result['provenance']['original_signal_end'] == str(days[-1].date())
