"""Synthetic input preparation checks; no network, database, or return runs."""
from datetime import date
import json

import pandas as pd
import pytest

from scripts import prepare_million_prices as prep


def companies():
    return pd.DataFrame([
        {'stock_id': '1101', 'name': 'A', 'listed_date': pd.Timestamp('2000-01-01'), 'industry': '01', 'market': 'TWSE'},
        {'stock_id': '2002', 'name': 'B', 'listed_date': pd.Timestamp('2026-07-01'), 'industry': '01', 'market': 'TPEX'},
        {'stock_id': '3718', 'name': 'never quoted', 'listed_date': pd.Timestamp('2020-01-01'), 'industry': '01', 'market': 'TWSE'},
    ])


def quote(sid='1101', day='2026-06-30', close=10., volume=12345):
    return {'stock_id': sid, 'date': day, 'open': close, 'high': close, 'low': close, 'close': close, 'volume': volume}


def calendar(days):
    return pd.DataFrame({'date': pd.to_datetime(days), 'is_open': True, 'session_type': 'FULL',
                         'note': 'fixture', 'calendar_source': 'local_trading_calendar'})


def test_price_identity_listing_mask_and_benchmark_exception_are_explicit():
    data = pd.DataFrame([quote(), quote('2002'), quote('2002', '2026-07-01'),
                         quote('0050'), quote('11010'), quote('not-stock')])
    before = data.copy(deep=True)
    result = prep.normalize_quotes(data, companies())
    pd.testing.assert_frame_equal(data, before)
    assert result.stock_id.tolist() == ['0050', '1101', '2002']
    assert result.volume.tolist() == [12345, 12345, 12345]  # Already shares; never multiply by 1000.
    assert result.iloc[-1].date == pd.Timestamp('2026-07-01')


@pytest.mark.parametrize('change', ['duplicate', 'numeric_id', 'fraction_volume', 'negative_volume',
                                   'infinity', 'bad_date', 'time_of_day', 'missing_high', 'wrong_requested_day'])
def test_invalid_source_shapes_are_not_silently_accepted(change):
    values = pd.DataFrame([quote()])
    kwargs = {}
    if change == 'duplicate': values = pd.concat([values, values], ignore_index=True)
    elif change == 'numeric_id': values.loc[0, 'stock_id'] = 1101
    elif change == 'fraction_volume':
        values['volume'] = values.volume.astype(float)
        values.loc[0, 'volume'] = .5
    elif change == 'negative_volume': values.loc[0, 'volume'] = -1
    elif change == 'infinity': values.loc[0, 'close'] = float('inf')
    elif change == 'bad_date': values.loc[0, 'date'] = 'unknown'
    elif change == 'time_of_day': values.loc[0, 'date'] = '2026-06-30T12:00:00'
    elif change == 'missing_high': values = values.drop(columns='high')
    else: kwargs['requested_day'] = date(2026, 7, 1)
    with pytest.raises(ValueError): prep.normalize_quotes(values, companies(), **kwargs)


def test_raw_missing_and_zero_observations_are_preserved_not_forward_filled():
    values = pd.DataFrame([quote(), quote(day='2026-07-01', close=0., volume=0),
                           quote(day='2026-07-02', close=None, volume=10)])
    result = prep.normalize_quotes(values, companies())
    assert result.close.iloc[1] == 0.
    assert pd.isna(result.close.iloc[2])
    assert result.volume.iloc[1] == 0


def test_finmind_mapping_and_out_of_date_rows_are_validated_before_cohort_filter():
    values = pd.DataFrame([quote()]).rename(columns={'high': 'max', 'low': 'min', 'volume': 'Trading_Volume'})
    result = prep.normalize_quotes(values, companies(), requested_day=date(2026, 6, 30))
    assert set(result) == set(prep.QUOTE_COLUMNS)
    values = pd.concat([values, pd.DataFrame([{'stock_id': 'not-cohort', 'date': '2026-07-01'}])], ignore_index=True)
    with pytest.raises(ValueError, match='requested date'):
        prep.normalize_quotes(values, companies(), requested_day=date(2026, 6, 30))


def test_coverage_keeps_never_quoted_companies_and_excludes_prelisting_denominator():
    values = prep.normalize_quotes(pd.DataFrame([quote(), quote('0050')]), companies())
    result = prep.market_coverage(values, companies(), calendar(['2026-06-30', '2026-07-01']))
    before = result[(result.date == pd.Timestamp('2026-06-30')) & result.market.eq('TWSE')].iloc[0]
    assert before.listed_cohort_count == 2 and before.quote_rows == 1
    assert before.quote_coverage == .5 and before.suspected_partial_feed
    tpex = result[result.market.eq('TPEX')]
    assert tpex.listed_cohort_count.tolist() == [0, 1]
    assert tpex.missing_market.tolist() == [False, True]


def test_bad_ohlc_remains_visible_in_coverage():
    row = quote()
    row['high'] = 5.
    result = prep.market_coverage(prep.normalize_quotes(pd.DataFrame([row]), companies()),
                                companies(), calendar(['2026-06-30']))
    twse = result[result.market.eq('TWSE')].iloc[0]
    assert twse.quote_rows == 1 and twse.positive_close == 1 and twse.valid_ohlcv == 0


@pytest.mark.parametrize('payload', [None, {}, {'stat': 'OK', 'data': []},
                                    {'stat': '查無資料', 'data': []},
                                    {'stat': 'OK', 'data': [['115/08/03']]},
                                    {'stat': 'OK', 'data': [['115/07/01'], ['115/07/01']]}])
def test_calendar_empty_bad_and_ignored_requested_month_fail_closed(payload):
    with pytest.raises((ValueError, prep.TWSEError)):
        prep.parse_calendar_month(payload, date(2026, 7, 1))


def test_official_calendar_only_changes_explicit_month_with_no_invented_market_close():
    values = prep.normalize_quotes(pd.DataFrame([quote(day='2026-07-01')]), companies())
    cal = calendar(['2026-07-01', '2026-07-02', '2026-08-01'])
    original = cal.copy(deep=True)
    result, changes = prep.apply_calendar_evidence(cal, values, {date(2026, 7, 1): {date(2026, 7, 1)}})
    pd.testing.assert_frame_equal(cal, original)
    assert result.is_open.tolist() == [True, False, True]
    assert changes == [{'date': '2026-07-02', 'old_is_open': True, 'is_open': False}]
    with pytest.raises(ValueError, match='conflicts'):
        prep.apply_calendar_evidence(cal, values, {date(2026, 7, 1): {date(2026, 7, 2)}})


def test_market_gap_patch_never_changes_existing_quotes_or_other_market():
    values = prep.normalize_quotes(pd.DataFrame([quote(day='2026-07-01')]), companies())
    incoming = pd.DataFrame([quote(day='2026-07-01', close=99.), quote('2002', '2026-07-01'),
                             quote('3718', '2026-07-01'), quote('0050', '2026-07-01')])
    result, count = prep.patch_quotes(values, incoming, companies(), date(2026, 7, 1), ['TPEX'])
    assert count == 1 and set(result.stock_id) == {'1101', '2002'}
    assert result.loc[result.stock_id.eq('1101'), 'close'].item() == 10.
    with pytest.raises(ValueError, match='no frozen-cohort'):
        prep.patch_quotes(values, incoming[incoming.stock_id.eq('1101')], companies(), date(2026, 7, 1), ['TPEX'])


def test_parquet_snapshots_preserve_hash_and_detect_changes(tmp_path):
    path = tmp_path / 'quotes.parquet'
    values = prep.normalize_quotes(pd.DataFrame([quote()]), companies())
    prep.save_frame(path, values, source='fixture')
    pd.testing.assert_frame_equal(prep.verified_frame(path), values)
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Changed prepared snapshot'):
        prep.verified_frame(path)


def test_calendar_cache_is_reused_without_another_http_request(tmp_path, monkeypatch):
    calls = []
    class Client:
        def __init__(self, **kwargs): pass
        def _get_json(self, url, params):
            calls.append((url, params))
            return {'stat': 'OK', 'data': [['115/07/01'], ['115/07/02']]}
    monkeypatch.setattr(prep, 'TWSEClient', Client)
    month = date(2026, 7, 1)
    first, provenance = prep.calendar_evidence(tmp_path, {month})
    second, _ = prep.calendar_evidence(tmp_path, {month})
    assert first == second and len(calls) == 1
    assert provenance[0]['query']['date'] == '20260701'
    path = tmp_path / provenance[0]['path']
    path.write_text('{}')
    with pytest.raises(ValueError, match='source changed'):
        prep.calendar_evidence(tmp_path, {month})


def test_complete_calendar_required_and_boolean_flags_explicit():
    cal = calendar(pd.date_range(prep.START, prep.END))
    result = prep.normalize_calendar(cal)
    assert result.is_open.dtype == bool
    with pytest.raises(ValueError, match='every calendar date'):
        prep.normalize_calendar(cal.iloc[1:])
    cal['is_open'] = cal.is_open.astype(object)
    cal.loc[0, 'is_open'] = 'yes'
    with pytest.raises(ValueError): prep.normalize_calendar(cal)


def test_preparation_seals_only_owned_prices_and_sources_not_concurrent_root_feeds(tmp_path, monkeypatch):
    root, output = tmp_path / 'root', tmp_path / 'output'
    root.mkdir(); output.mkdir()
    monkeypatch.setattr(prep, 'ROOT', root)
    for name in (*prep.CODE, prep.COHORT_PATH, prep.COHORT_MANIFEST, prep.OLD_EVENTS, prep.OLD_EVENTS_META):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixed-parent-' + name)
    cohort = companies()
    raw = prep.normalize_quotes(pd.DataFrame([quote(sid, '2026-07-01') for sid in ['1101', '2002', '3718', '0050']]), cohort)
    cal = calendar(['2026-07-01'])
    prep.save_frame(output / 'companies.parquet', cohort)
    (output / 'sources').mkdir()
    prep.save_frame(output / 'sources/finmind-price-0050-full.parquet', raw[raw.stock_id.eq('0050')],
                    provider_attrs={'cache_hit': True})
    (output / 'sources/db-fixture.json').write_text('{"frozen": true}')
    (output / 'dividends').mkdir()
    changing = output / 'dividends/updating.json'
    changing.write_text('{"not_owned": 1}')
    (output / 'execution-feeds').mkdir()
    (output / 'execution-feeds/new.json').write_text('{}')
    monkeypatch.setattr(prep, 'load_local', lambda folder: (raw, cohort, cal))
    monkeypatch.setattr(prep, 'load_config', lambda: object())
    monkeypatch.setattr(prep, 'calendar_evidence', lambda folder, months: ({}, []))
    def actions(folder):
        prep.save_frame(folder / 'events.parquet', pd.DataFrame(columns=['stock_id', 'event_date']))
        return []
    monkeypatch.setattr(prep, 'prepare_actions', actions)
    monkeypatch.setattr(prep, 'fetch_dataset', lambda *args, **kwargs: pytest.fail('No feed gap means no API request'))
    result = prep.prepare(output)
    assert result['finmind_requests'] == 0 and result['unresolved_market_date_gaps'] == []
    assert 'sources/db-fixture.json' in result['files_sha256']
    assert not any(name.startswith(('dividends/', 'execution-feeds/')) for name in result['files_sha256'])
    changing.write_text('{"not_owned": 2}')
    assert prep.verify(output) == result
    (output / 'sources/db-fixture.json').write_text('{"frozen": false}')
    with pytest.raises(ValueError, match='Prepared input changed'):
        prep.verify(output)


def test_benchmark_requires_every_open_day_except_explicit_official_suspension():
    raw = prep.normalize_quotes(pd.DataFrame([quote('0050', '2025-06-10'),
        quote('0050', '2025-06-11', close=0., volume=0), quote('0050', '2025-06-18')]), companies())
    audit = prep.benchmark_coverage(raw, calendar(['2025-06-10', '2025-06-11', '2025-06-12', '2025-06-18', '2025-06-19']))
    assert audit.status.tolist() == ['observed', 'official_split_suspension', 'official_split_suspension', 'observed', 'unresolved']
    assert not audit.iloc[2].quote_present  # Missing suspended rows are not invented.
    assert not audit.iloc[4].quote_present
    with pytest.raises(ValueError, match='apparent fill'):
        prep.benchmark_coverage(prep.normalize_quotes(pd.DataFrame([quote('0050', '2025-06-11')]), companies()),
                                calendar(['2025-06-11']))


def test_complete_ordinary_markets_do_not_hide_missing_benchmark():
    cohort = companies()
    raw = prep.normalize_quotes(pd.DataFrame([quote(sid, '2026-07-01') for sid in cohort.stock_id]), cohort)
    cal = calendar(['2026-07-01'])
    coverage = prep.market_coverage(raw, cohort, cal)
    assert coverage.quote_coverage.eq(1.).all()
    assert prep.benchmark_coverage(raw, cal).status.tolist() == ['unresolved']


def test_single_full_history_benchmark_request_is_shared_bounded_and_cached(tmp_path, monkeypatch):
    from types import SimpleNamespace
    cohort = companies()
    before = prep.normalize_quotes(pd.DataFrame([quote('0050', '2026-07-01'), quote('1101', '2026-07-01')]), cohort)
    response = pd.DataFrame([quote('0050', '2026-07-01'), quote('0050', '2026-07-02')])
    response.attrs = {'cache_hit': False, 'retrieved_at': 123.}
    calls = []
    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return response
    monkeypatch.setattr(prep, 'fetch_dataset', fetch)
    config = SimpleNamespace(finmind_token='fixture-token')
    result, audit, requests = prep.ensure_benchmark(tmp_path, before, cohort, calendar(['2026-07-01', '2026-07-02']), config)
    assert requests == 1 and audit.status.eq('observed').all()
    assert len(result) == 3
    assert calls[0][0] == ('TaiwanStockPrice', prep.START, prep.END)
    assert calls[0][1]['data_id'] == '0050' and calls[0][1]['requests_per_hour'] == 5400
    assert calls[0][1]['max_retries'] == 0
    again, _, requests = prep.ensure_benchmark(tmp_path, before, cohort, calendar(['2026-07-01', '2026-07-02']), config)
    pd.testing.assert_frame_equal(result, again)
    assert requests == 0 and len(calls) == 1


def test_conflicting_benchmark_source_is_not_silently_overwritten(tmp_path, monkeypatch):
    from types import SimpleNamespace
    cohort = companies()
    old = prep.normalize_quotes(pd.DataFrame([quote('0050', '2026-07-01')]), cohort)
    source = pd.DataFrame([quote('0050', '2026-07-01', close=99.)])
    source.attrs = {'cache_hit': False}
    monkeypatch.setattr(prep, 'fetch_dataset', lambda *args, **kwargs: source)
    with pytest.raises(ValueError, match='conflict'):
        prep.ensure_benchmark(tmp_path, old, cohort, calendar(['2026-07-01']), SimpleNamespace(finmind_token='fixture'))
    assert (tmp_path / 'benchmark_conflicts.parquet').exists()
    assert old.close.iloc[0] == 10.
