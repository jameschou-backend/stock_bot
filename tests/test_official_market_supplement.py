from copy import deepcopy
import json

import pytest

from skills.market_input_validation import MarketEvidenceError
from skills.official_market_supplement import (SCHEMA, collect_supplement, compare_quote,
    execution_scope, merge_source_rows, parse_tpex_daily_quotes, sha, validate_entry)


def modern_payload():
    fields = ['代號', '名稱', '收盤', '開盤', '最高', '最低', '成交股數']
    return dict(date='20260909', stat='ok', tables=[
        dict(title='上櫃股票行情', date='115/09/09', listedCompanies='1', totalCount=2,
             fields=fields, data=[['2330', '測試', '105', '100', '110', '90', '100,000'],
                                 ['12345P', '權證', '1', '1', '1', '1', '1']]),
        dict(title='管理股票', totalCount=0, fields=fields, data=[]),
    ])


def save_fixture(root, payload=None, receipt_changes=None):
    raw, receipt_path = root/'day.json', root/'day.source.json'
    raw.write_text(json.dumps(payload or modern_payload(), ensure_ascii=False))
    receipt = dict(url='https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes',
        params=dict(date='2026/09/09', response='json'), http_status=200,
        raw_path='day.json', raw_sha256=sha(raw), market='TPEX', date='2026-09-09',
        volume_scope='unclassified_daily', retrieved_at='2026-10-02T00:00:00+00:00')
    receipt.update(receipt_changes or {})
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False))
    return dict(market='TPEX', date='2026-09-09', raw_path='day.json', raw_sha256=sha(raw),
        receipt_path='day.source.json', receipt_sha256=sha(receipt_path))


def manifest(root, entries):
    path = root/'manifest.json'
    path.write_text(json.dumps(dict(schema=SCHEMA, entries=entries, live_qualified=False)))
    path.with_suffix('.sha256').write_text(sha(path)+'\n')
    return path


def test_modern_daily_quotes_never_verify_unknown_volume():
    rows = parse_tpex_daily_quotes(modern_payload(), '2026-09-09')
    assert list(rows) == ['2330']
    row = rows['2330']
    assert row['volume_scope'] == 'unclassified_daily'
    result = compare_quote(dict(open=100, high=110, low=90, close=105, volume=100000), row)
    assert result['close'] == 'matched'
    assert result['total_volume'] == 'unverified_scope'
    assert result['ordinary_volume'] == 'unverified'


@pytest.mark.parametrize('mutation', ['date', 'table_date', 'count', 'managed_table', 'duplicate',
    'duplicate_warrant', 'headers', 'width', 'negative_volume', 'fractional_volume', 'impossible'])
def test_modern_full_scope_rejects_malformed_tables(mutation):
    p = modern_payload()
    table = p['tables'][0]
    if mutation == 'date': p['date'] = '20260908'
    elif mutation == 'table_date': table['date'] = '115/09/08'
    elif mutation == 'count': table['totalCount'] = 3
    elif mutation == 'managed_table': p['tables'].pop()
    elif mutation in ('duplicate', 'duplicate_warrant'):
        table['data'].append(deepcopy(table['data'][0 if mutation == 'duplicate' else 1]))
        table['totalCount'] += 1
    elif mutation == 'headers': table['fields'][-1] = '收盤'
    elif mutation == 'width': table['data'][0].pop()
    elif mutation == 'negative_volume': table['data'][0][-1] = '-1'
    elif mutation == 'fractional_volume': table['data'][0][-1] = '1.5'
    else: table['data'][0][4] = '99'
    with pytest.raises(MarketEvidenceError):
        parse_tpex_daily_quotes(p, '2026-09-09')


def test_hash_bound_receipt_and_manifest_collect_exact_primary_rows(tmp_path):
    entry = save_fixture(tmp_path)
    refs = {}
    rows, sources = collect_supplement(manifest(tmp_path, [entry]), tmp_path, refs)
    assert list(rows) == [('TPEX', '2026-09-09', '2330')]
    assert sources[0]['http_status'] == 200
    assert sources[0]['http_status_evidence'] == 'recorded_http_200'
    assert refs['day.json'] == entry['raw_sha256']
    assert refs['day.source.json'] == entry['receipt_sha256']
    assert 'manifest.json' in refs


@pytest.mark.parametrize('changes', [
    dict(http_status=403), dict(status_code=403), dict(volume_scope='ordinary_session'),
    dict(url='https://example.com/www/zh-tw/afterTrading/dailyQuotes'),
    dict(url='https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes?date=2026/09/08'),
    dict(url='https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes?response=json&response=json'),
    dict(params=dict(date='2026/09/09', response='json', code='2330')),
    dict(raw_sha256='0'*64), dict(date='2026-09-08'), dict(market='TWSE'),
])
def test_receipt_scope_status_and_hash_cannot_be_overridden_by_entry(tmp_path, changes):
    entry = save_fixture(tmp_path, receipt_changes=changes)
    with pytest.raises(MarketEvidenceError):
        validate_entry(entry, tmp_path)


def test_unknown_http_status_is_not_silently_promoted(tmp_path):
    entry = save_fixture(tmp_path)
    p = tmp_path/'day.source.json'
    receipt = json.loads(p.read_text())
    del receipt['http_status']
    p.write_text(json.dumps(receipt))
    entry['receipt_sha256'] = sha(p)
    with pytest.raises(MarketEvidenceError, match='HTTP status missing'):
        validate_entry(entry, tmp_path)


def standard_download_receipt():
    return dict(schema='official_daily_receipt_v1', accepted=True, status='verified_market_day',
        automatic_redirects_disabled=True, redirect_statuses=[], security_denied=False)


def test_standard_downloader_receipt_business_status_is_not_an_http_status(tmp_path):
    entry = save_fixture(tmp_path, receipt_changes=standard_download_receipt())
    rows, source = validate_entry(entry, tmp_path)
    assert source['http_status'] == 200
    assert rows['2330']['volume_scope'] == 'unclassified_daily'


@pytest.mark.parametrize('field,value', [('accepted', False), ('status', 'blocked'),
    ('automatic_redirects_disabled', False), ('redirect_statuses', [302]), ('security_denied', True),
    ('security_denied', None)])
def test_standard_downloader_receipt_requires_acceptance_and_no_security_denial(tmp_path, field, value):
    receipt = standard_download_receipt()
    receipt[field] = value
    entry = save_fixture(tmp_path, receipt_changes=receipt)
    with pytest.raises(MarketEvidenceError, match='not safely accepted'):
        validate_entry(entry, tmp_path)


def test_manifest_source_closure_includes_plan_and_rejects_changed_plan(tmp_path):
    entry = save_fixture(tmp_path)
    plan = tmp_path/'plan.json'
    plan.write_text('{"days":1}')
    path = manifest(tmp_path, [entry])
    data = json.loads(path.read_text())
    data['source_sha256'] = {'plan.json': sha(plan)}
    path.write_text(json.dumps(data))
    path.with_suffix('.sha256').write_text(sha(path)+'\n')
    refs = {}
    collect_supplement(path, tmp_path, refs)
    assert refs['plan.json'] == sha(plan)
    plan.write_text('{"days":2}')
    with pytest.raises(MarketEvidenceError, match='Changed manifest source'):
        collect_supplement(path, tmp_path, {})


def test_changed_raw_or_manifest_bytes_fail_closed(tmp_path):
    entry = save_fixture(tmp_path)
    path = manifest(tmp_path, [entry])
    path.write_text(path.read_text()+' ')
    with pytest.raises(MarketEvidenceError, match='Changed supplement manifest'):
        collect_supplement(path, tmp_path, {})
    path = manifest(tmp_path, [entry])
    (tmp_path/'day.json').write_text('{}')
    with pytest.raises(MarketEvidenceError, match='Changed supplement raw'):
        collect_supplement(path, tmp_path, {})


def test_duplicate_sources_and_path_escape_are_rejected(tmp_path):
    entry = save_fixture(tmp_path)
    with pytest.raises(MarketEvidenceError, match='Duplicate supplement'):
        collect_supplement(manifest(tmp_path, [entry, entry]), tmp_path, {})
    entry['raw_path'] = '../day.json'
    with pytest.raises(MarketEvidenceError, match='Missing or escaped'):
        validate_entry(entry, tmp_path)


def test_overlapping_primary_sources_require_matching_prices_and_same_scope_volumes():
    row = parse_tpex_daily_quotes(modern_payload(), '2026-09-09')['2330']
    key = ('TPEX', '2026-09-09', '2330')
    ordinary = dict(row, volume_scope='ordinary_session', volume=95000)
    existing = {key: ordinary}
    merge_source_rows(existing, {'2330': row}, 'TPEX', '2026-09-09')
    assert existing[key] == ordinary
    with pytest.raises(MarketEvidenceError, match='primary daily prices'):
        merge_source_rows(existing, {'2330': dict(row, close=106)}, 'TPEX', '2026-09-09')
    with pytest.raises(MarketEvidenceError, match='same-scope primary volume'):
        merge_source_rows(existing, {'2330': dict(ordinary, volume=96000)}, 'TPEX', '2026-09-09')


def test_unclassified_daily_volume_cannot_certify_execution_capacity():
    row = parse_tpex_daily_quotes(modern_payload(), '2026-09-09')['2330']
    trade = dict(date='2026-09-09', stock_id='2330', channel='board', sequence=1, qty=1000,
        capacity_qty=1000, source_high=110, source_low=90, source_volume=100000, prior_avg_volume20=100000)
    episodes = [dict(stock_id='2330', market='TPEx', category='股票', start='2000-01-01', end=None)]
    result = execution_scope(dict(trades=[trade]), {('TPEX', '2026-09-09', '2330'): row},
        ['2026-09-09'], episodes, [])
    assert result['rows'][0]['status'] == 'ordinary_volume_missing'
    assert result['rows'][0]['price_fields']['ordinary_volume'] == 'unverified'
    assert result['all_capacity_verified'] is False


def test_overlap_keeps_managed_classification_even_when_ordinary_volume_is_stronger():
    row = parse_tpex_daily_quotes(modern_payload(), '2026-09-09')['2330']
    managed = dict(row, table_category='管理股票')
    ordinary = {k: v for k, v in dict(row, volume_scope='ordinary_session', volume=95000).items()
                if k != 'table_category'}
    key = ('TPEX', '2026-09-09', '2330')
    for first, second in [(ordinary, managed), (managed, ordinary)]:
        existing = {key: first}
        merge_source_rows(existing, {'2330': second}, 'TPEX', '2026-09-09')
        assert existing[key]['volume_scope'] == 'ordinary_session'
        assert existing[key]['table_category'] == '管理股票'
    with pytest.raises(MarketEvidenceError, match='market classifications'):
        merge_source_rows({key: managed}, {'2330': row}, 'TPEX', '2026-09-09')
