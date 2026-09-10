"""Synthetic source verification, date-safe caching, and offline execution data."""
from copy import deepcopy
from datetime import date
import json

import pandas as pd
import pytest
import requests

from app.finmind import FinMindError, FinMindQuotaError
from skills.replay_market_feeds import (END, FIELDS, START, URLS, ReplayDataUnavailable,
                                       ReplayMarketFeeds, parse_limits, parse_odd)


def odd_record(market='twse', day='2022-01-04'):
    values = {'stock_id': '0050' if market == 'twse' else '6488', 'odd_shares': '1,200',
              'odd_last': '101.00', 'odd_low': '99.00', 'odd_high': '102.00',
              'odd_bid': '100.95', 'odd_ask': '101.05', 'bid_qty': '1,032', 'ask_qty': '42'}
    fields = list(FIELDS[market].values())
    data = [[values[key] for key in FIELDS[market]]]
    table = {'fields': fields, 'data': data}
    if market == 'twse':
        table.update(title='111年01月04日 盤中零股交易行情單', total=1)
        payload = {**table, 'date': day.replace('-', ''), 'stat': 'OK'}
    else:
        table.update(title='盤中零股每日收盤行情', totalCount=1, date=day.replace('-', ''))
        payload = {'tables': [table], 'date': day.replace('-', ''), 'stat': 'ok'}
    return {'schema': 1, 'provider': market, 'day': day, 'url': URLS[market],
            'params': {'date': day.replace('-', '' if market == 'twse' else '/'), 'response': 'json'},
            'retrieved_at': '2026-09-10T01:00:00+00:00', 'http_status': 200, 'payload': payload}


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return deepcopy(self.payload)


def feeds(tmp_path, **kwargs):
    return ReplayMarketFeeds(tmp_path, token='synthetic-token', official_min_interval=0, **kwargs)


def limit_record(data=None):
    return {'schema': 1, 'provider': 'finmind', 'dataset': 'TaiwanStockPriceLimit', 'stock_id': '0050',
            'start_date': START.isoformat(), 'end_date': END.isoformat(), 'status': 'success',
            'data': [{'date': '2022-01-04', 'stock_id': '0050', 'limit_up': 110., 'limit_down': 90.}]
            if data is None else data}


@pytest.mark.parametrize('market,sid', [('twse', '0050'), ('tpex', '6488')])
def test_official_named_fields_and_exact_date_cache_replay(tmp_path, market, sid):
    calls = []
    source = odd_record(market)
    def http(url, **kwargs):
        calls.append((url, kwargs))
        return Response(source['payload'])
    online = feeds(tmp_path, http_get=http)
    row = online.get_odd('2022-01-04', sid, market.upper())
    assert row['odd_shares'] == 1200 and row['odd_last'] == 101
    assert row['bid_qty'] == 1032 and row['ask_qty'] == 42
    assert row['daily_participation_ceiling'] == .05 and 'not a fill guarantee' in row['execution_limitation']
    assert len(calls) == 1 and calls[0][1]['params'] == source['params']
    assert 'd' not in calls[0][1]['params']
    assert online.get_odd(date(2022, 1, 4), '9999', market) is None
    assert len(calls) == 1  # Whole-market daily response serves other stocks.
    before = online.manifest()
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline must never call any source')
    offline = feeds(tmp_path, offline=True, http_get=forbidden, finmind_fetch=forbidden)
    assert offline.get_odd('2022-01-04', sid, market) == row
    row['odd_shares'] = 1
    assert offline.get_odd('2022-01-04', sid, market)['odd_shares'] == 1200
    assert offline.manifest() == before
    assert before['request_counters']['official_http_requests'] == 1


def test_verified_empty_day_or_absent_stock_is_not_a_failed_request(tmp_path):
    source = odd_record()
    source['payload'].update(data=[], total=0)
    store = feeds(tmp_path, http_get=lambda *args, **kwargs: Response(source['payload']))
    assert store.get_odd('2022-01-04', '0050', 'twse') is None
    manifest = store.manifest()
    assert manifest['entries']['odd:twse:2022-01-04']['empty_verified_response']
    assert manifest['entries']['odd:twse:2022-01-04']['first_date'] == '2022-01-04'


def test_no_trades_preserves_zero_volume_and_available_quotes():
    source = odd_record()
    row = source['payload']['data'][0]
    for key in ('odd_shares', 'odd_last', 'odd_low', 'odd_high'):
        row[list(FIELDS['twse']).index(key)] = '0' if key == 'odd_shares' else '--'
    result = parse_odd(source, 'twse', '2022-01-04')['0050']
    assert result['odd_shares'] == 0 and result['odd_last'] is None and result['odd_bid'] == 100.95


@pytest.mark.parametrize('market', ['twse', 'tpex'])
def test_optional_quote_quantities_are_never_invented(market):
    source = odd_record(market)
    table = source['payload'] if market == 'twse' else source['payload']['tables'][0]
    table['fields'] = table['fields'][:-2]
    table['data'][0] = table['data'][0][:-2]
    row = next(iter(parse_odd(source, market, '2022-01-04').values()))
    assert row['bid_qty'] is None and row['ask_qty'] is None


@pytest.mark.parametrize('mutation,pattern', [
    (lambda s: s['payload'].update(date='20260909'), 'date/status'),
    (lambda s: s['payload'].update(stat='not available'), 'date/status'),
    (lambda s: s['payload'].update(title='115年09月09日 盤中零股交易行情單'), 'title/date'),
    (lambda s: s['payload']['fields'].__setitem__(1, '未知欄位'), 'fields changed'),
    (lambda s: s['payload'].update(total=2), 'incomplete'),
    (lambda s: s['payload']['data'][0].pop(), 'field names'),
    (lambda s: s.update(http_status=503), 'HTTP status'),
    (lambda s: s.update(params={'d': '111/01/04', 'response': 'json'}), 'provenance'),
])
def test_invalid_response_never_means_no_odd_trade(mutation, pattern):
    source = odd_record()
    mutation(source)
    with pytest.raises(ReplayDataUnavailable, match=pattern):
        parse_odd(source, 'twse', '2022-01-04')


def test_tpex_checks_nested_date_even_when_top_level_matches():
    source = odd_record('tpex')
    source['payload']['tables'][0]['date'] = '20260909'
    with pytest.raises(ReplayDataUnavailable, match='table date'):
        parse_odd(source, 'tpex', '2022-01-04')


@pytest.mark.parametrize('key,value', [('odd_shares', '1.5'), ('odd_shares', '--'),
                                     ('odd_shares', '9007199254740993'), ('odd_low', '103'),
                                     ('odd_last', 'NaN'), ('odd_bid', '-1'), ('odd_ask', '--')])
def test_bad_numeric_evidence_is_not_coerced_to_zero(key, value):
    source = odd_record()
    source['payload']['data'][0][list(FIELDS['twse']).index(key)] = value
    with pytest.raises(ReplayDataUnavailable):
        parse_odd(source, 'twse', '2022-01-04')


def test_duplicate_rows_fail_but_out_of_scope_ids_remain_only_in_raw():
    source = odd_record()
    source['payload']['data'].append(deepcopy(source['payload']['data'][0]))
    source['payload']['total'] = 2
    with pytest.raises(ReplayDataUnavailable, match='Duplicate'):
        parse_odd(source, 'twse', '2022-01-04')
    source['payload']['data'][1][0] = '006201'
    assert list(parse_odd(source, 'twse', '2022-01-04')) == ['0050']


@pytest.mark.parametrize('failure', ['http', 'network', 'date'])
def test_failure_retains_provenance_and_does_not_cache_a_no_trade_day(tmp_path, failure):
    source = odd_record()
    if failure == 'date':
        source['payload']['date'] = '20260909'
    def http(*args, **kwargs):
        if failure == 'network':
            raise requests.Timeout('synthetic-token must not appear')
        return Response(source['payload'], 503 if failure == 'http' else 200)
    store = feeds(tmp_path, http_get=http)
    with pytest.raises(ReplayDataUnavailable):
        store.get_odd('2022-01-04', '0050', 'twse')
    manifest = store.manifest()
    assert not manifest['entries'] and manifest['request_counters']['official_http_requests'] == 1
    assert any(name.startswith('failures/') for name in manifest['files_sha256'])
    assert all('synthetic-token' not in (tmp_path / name).read_text() for name in manifest['files_sha256'])
    with pytest.raises(ReplayDataUnavailable, match='Offline replay'):
        feeds(tmp_path, offline=True).get_odd('2022-01-04', '0050', 'twse')


def test_limits_one_call_per_stock_with_shared_quota_and_complete_request_range(tmp_path):
    calls = []
    def fetch(dataset, start, end, **kwargs):
        calls.append((dataset, start, end, kwargs))
        frame = pd.DataFrame(limit_record()['data'])
        frame.attrs.update(cache_hit=False, source='finmind', retrieved_at=1.)
        return frame
    store = feeds(tmp_path, finmind_fetch=fetch)
    result = store.get_limits('0050')
    assert result == {'2022-01-04': {'upper': 110., 'lower': 90.}}
    assert store.get_limits('0050') == result and len(calls) == 1
    assert calls[0][:3] == ('TaiwanStockPriceLimit', START, END)
    assert calls[0][3]['requests_per_hour'] == 6000 and calls[0][3]['max_retries'] == 0
    assert calls[0][3]['data_id'] == '0050'
    state = store.manifest()
    assert state['request_counters']['finmind_requests'] == 1
    assert state['entries']['limits:0050']['row_count'] == 1
    assert feeds(tmp_path, offline=True).get_limits('0050') == result
    assert 'synthetic-token' not in json.dumps(state)
    assert all('synthetic-token' not in (tmp_path / name).read_text() for name in state['files_sha256'])


def test_empty_limits_are_explicit_verified_coverage_and_not_inferred(tmp_path):
    store = feeds(tmp_path, finmind_fetch=lambda *args, **kwargs: pd.DataFrame())
    assert store.get_limits('0050') == {}
    meta = store.manifest()['entries']['limits:0050']
    assert meta['empty_verified_response'] and meta['first_date'] is None and meta['last_date'] is None
    assert feeds(tmp_path, offline=True).get_limits('0050') == {}


@pytest.mark.parametrize('field,value', [('date', '2021-12-31'), ('date', '2026-09-10'),
                                       ('stock_id', '2330'), ('limit_up', None), ('limit_down', -1),
                                       ('limit_up', 0), ('limit_down', 120)])
def test_limits_wrong_stock_date_or_price_is_explicit_data_error(field, value):
    record = limit_record()
    record['data'][0][field] = value
    with pytest.raises(ReplayDataUnavailable):
        parse_limits(record, '0050')


def test_zero_limits_mean_unbounded_not_missing_or_zero_price():
    record = limit_record()
    record['data'][0].update(limit_up=0, limit_down=0)
    assert parse_limits(record, '0050')['2022-01-04'] == {'upper': 0., 'lower': 0.}


@pytest.mark.parametrize('quota', [False, True])
def test_finmind_errors_do_not_become_empty_history_or_retry(tmp_path, quota):
    calls = []
    def fetch(*args, **kwargs):
        calls.append(1)
        raise FinMindQuotaError(3600) if quota else FinMindError('source failed')
    store = feeds(tmp_path, finmind_fetch=fetch)
    with pytest.raises(FinMindQuotaError if quota else ReplayDataUnavailable):
        store.get_limits('0050')
    state = store.manifest()
    assert not state['entries'] and len(calls) == 1
    assert state['request_counters']['finmind_fetch_calls'] == 1
    assert state['finmind_requests_upper_bound'] == 1


@pytest.mark.parametrize('suffix', ['raw.json', 'rows.json'])
def test_changed_raw_or_normalized_cache_fails_offline(tmp_path, suffix):
    source = odd_record()
    store = feeds(tmp_path, http_get=lambda *args, **kwargs: Response(source['payload']))
    store.get_odd('2022-01-04', '0050', 'twse')
    path = tmp_path / ('odd-twse-2022-01-04.' + suffix)
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ReplayDataUnavailable, match='changed'):
        feeds(tmp_path, offline=True).get_odd('2022-01-04', '0050', 'twse')
    with pytest.raises(ReplayDataUnavailable, match='changed'):
        store.manifest()


def test_offline_missing_directory_does_not_create_anything(tmp_path):
    directory = tmp_path / 'absent'
    store = feeds(directory, offline=True)
    with pytest.raises(ReplayDataUnavailable):
        store.get_limits('0050')
    assert not directory.exists()


def test_verified_probe_adoption_avoids_repeating_preparation_requests(tmp_path):
    folder = tmp_path / 'probes'
    folder.mkdir()
    (folder / 'odd-twse-2022-01-04.json').write_text(json.dumps(odd_record()))
    limits = limit_record()
    limits.update(finmind_requests=1, attrs={'cache_hit': False})
    (folder / 'limits-0050.json').write_text(json.dumps(limits))
    def forbidden(*args, **kwargs):
        raise AssertionError('Probes must not be fetched twice')
    store = feeds(tmp_path, http_get=forbidden, finmind_fetch=forbidden)
    assert store.get_odd('2022-01-04', '0050', 'twse')['odd_last'] == 101
    assert store.get_limits('0050')['2022-01-04']['upper'] == 110
    counters = store.manifest()['request_counters']
    assert counters['official_http_requests'] == counters['finmind_requests'] == 1
