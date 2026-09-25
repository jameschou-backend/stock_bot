"""Offline data acceptance tests; synthetic exchange receipts make no HTTP calls."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import audit_external_source_followup as external
from scripts import audit_odd_lot_daily_gaps as daily
from skills.replay_market_feeds import FIELDS, ReplayDataUnavailable, ReplayMarketFeeds


PROJECT = Path(__file__).resolve().parents[1]
REQUESTS = [
    ('2022-02-11', '6104', 'TPEX'),
    ('2025-04-07', '2049', 'TWSE'), ('2025-04-07', '3163', 'TPEX'),
    ('2025-04-07', '4303', 'TPEX'), ('2025-04-08', '2049', 'TWSE'),
    ('2025-04-08', '3163', 'TPEX'), ('2025-04-08', '4303', 'TPEX'),
    ('2025-04-08', '4931', 'TPEX'), ('2025-04-08', '5439', 'TPEX'),
    ('2025-04-08', '6139', 'TWSE'), ('2026-07-17', '3105', 'TPEX'),
    ('2026-07-29', '6488', 'TPEX'), ('2026-07-30', '6488', 'TPEX'),
]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_files(root, names):
    for name in names:
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PROJECT/name).read_bytes())


def odd_payload(day, market, stocks):
    data = []
    for stock in stocks:
        if (day, stock) == ('2025-04-07', '3163'):
            continue  # A genuinely missing row must remain unknown.
        zero = (day, stock) == ('2025-04-07', '2049')
        values = dict(stock_id=stock, odd_shares='0' if zero else '100',
            odd_last='--' if zero else '10', odd_high='--' if zero else '11',
            odd_low='--' if zero else '9', odd_bid='--', odd_ask='--',
            bid_qty='0', ask_qty='0')
        data.append([values[key] for key in FIELDS[market]])
    compact = day.replace('-', '')
    table = dict(date=compact, fields=list(FIELDS[market].values()), data=data)
    if market == 'twse':
        year, month, date = day.split('-')
        return dict(table, stat='OK', type='ALL', total=len(data),
                    title=f'{int(year)-1911}年{month}月{date}日 盤中零股交易行情單')
    table.update(title='盤中零股每日收盤行情', totalCount=len(data))
    return dict(date=compact, stat='OK', tables=[table])


@pytest.fixture
def daily_capture(tmp_path, monkeypatch):
    selected = [dict(date=day, stock_id=stock, market=market) for day,stock,market in REQUESTS]
    request = tmp_path/'request'
    for name in ('request.json', 'stock_days.json', 'stock_days.csv', 'manifest.json'):
        write(request/name, {})
    code_files(tmp_path, ('scripts/audit_odd_lot_daily_gaps.py',
        'scripts/prepare_odd_lot_evidence_request.py', 'skills/replay_market_feeds.py'))
    monkeypatch.setattr(daily, 'ROOT', tmp_path)
    monkeypatch.setattr(daily, 'demand', lambda directory: ({'input_sha256':{}}, selected))
    stocks = defaultdict(list)
    for day,stock,market in REQUESTS:
        stocks[(day,market.lower())].append(stock)

    class Response:
        status_code = 200
        def __init__(self, payload):
            self.payload = payload
        def json(self):
            return deepcopy(self.payload)

    def exchange_get(url, params, timeout):
        market = 'twse' if 'twse.com.tw' in url else 'tpex'
        value = params['date'].replace('/', '')
        day = f'{value[:4]}-{value[4:6]}-{value[6:8]}'
        return Response(odd_payload(day, market, stocks[(day,market)]))

    directory = tmp_path/'capture'
    feeds = ReplayMarketFeeds(directory/'feeds', http_get=exchange_get, official_min_interval=0)
    for row in selected:
        feeds.get_odd(row['date'], row['stock_id'], row['market'])
    assert feeds.manifest()['request_counters']['official_http_requests'] == 8
    return directory, request


def test_daily_positive_zero_and_missing_are_three_distinct_evidence_states(daily_capture):
    directory, request = daily_capture
    report = daily.build(directory, request)
    assert report['statuses'] == {'official_daily_positive_trade':11,
        'official_daily_zero_trades':1, 'stock_row_absent_unproven':1}
    zero = next(r for r in report['rows'] if r['status']=='official_daily_zero_trades')
    missing = next(r for r in report['rows'] if r['status']=='stock_row_absent_unproven')
    assert zero['no_trades_confirmed_by_daily_record'] is True
    assert zero['daily_row_missing'] is False
    assert missing['official_daily_row'] is None
    assert missing['no_trades_confirmed_by_daily_record'] is False
    assert missing['daily_row_missing'] is True
    assert all(not r['accepted_for_strict_replay'] and not r['trading_suspension_proven']
               and not r['historical_auction_sequence_acquired'] for r in report['rows'])
    assert report['accepted_sequence_sessions'] == report['historical_auction_rows_acquired'] == 0


@pytest.mark.parametrize('mutation', ['date', 'market'])
def test_wrong_exchange_response_identity_is_rejected_even_with_updated_local_hashes(daily_capture, mutation):
    directory, request = daily_capture
    feeds = directory/'feeds'
    name = 'odd-tpex-2022-02-11.raw.json'
    raw = json.loads((feeds/name).read_text())
    if mutation == 'date':
        raw['payload']['date'] = '20220210'
    else:
        raw['provider'] = 'twse'
    write(feeds/name, raw)
    row_name = name.replace('.raw.json', '.rows.json')
    rows = json.loads((feeds/row_name).read_text())
    rows['raw_sha256'] = sha(feeds/name)
    write(feeds/row_name, rows)
    index = json.loads((feeds/'index.json').read_text())
    for key in (name, row_name):
        index['files_sha256'][key] = sha(feeds/key)
    write(feeds/'index.json', index)
    with pytest.raises(ReplayDataUnavailable):
        daily.build(directory, request)


def test_daily_verified_report_rejects_changed_source(daily_capture):
    directory, request = daily_capture
    path = directory/'report.json'
    write(path, daily.build(directory, request))
    path.with_suffix('.sha256').write_text(sha(path)+'\n')
    assert daily.verify_report(path)['required_stock_days'] == 13
    raw = directory/'feeds/odd-tpex-2022-02-11.raw.json'
    raw.write_bytes(raw.read_bytes()+b' ')
    with pytest.raises(ValueError, match='source changed'):
        daily.verify_report(path)


@pytest.fixture
def external_capture(tmp_path, monkeypatch):
    directory, demand, official = (tmp_path/name for name in ('publication', 'request', 'official'))
    daily_path = tmp_path/'daily/report.json'
    for name in ('request.json', 'manifest.json', 'stock_days.csv'):
        write(demand/name, {})
    write(demand/'stock_days.json', [dict(market='TPEX', date='2023-01-03')])
    write(daily_path, {})
    daily_path.with_suffix('.sha256').write_text(sha(daily_path))
    security = b'FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED'
    directory.mkdir()
    (directory/'entry.html').write_bytes(security)
    observed = '2026-09-25T05:45:48+00:00'
    write(directory/'entry.source.json', dict(url='https://mops.twse.com.tw/mops/',
        http_status=200, observed_at=observed, sha256=hashlib.sha256(security).hexdigest()))
    body = b'''<h1 class="entry-title">Walsin Technology global consolidated net sales for December 2022</h1>
        <div class="post-content">Historical revenue release body, observed today.</div>
        <div class="fusion-meta-info"><span>January 9th, 2023</span></div>'''
    (directory/'walsin-december-2022.html').write_bytes(body)
    write(directory/'walsin-december-2022.source.json', dict(http_status=200,
        path='walsin-december-2022.html', observed_at=observed, sha256=hashlib.sha256(body).hexdigest(),
        url='https://www.passivecomponent.com/2023/01/09/walsin-technology-global-consolidated-net-sales-for-december-2022/'))
    official.mkdir()
    sources = {'twse-h4.html':'data-price="1500" 上兩個月底',
        'tpex-mth.html':'2022/11/01 僅提供購買一年前 外部使用 data-price="10000"',
        'finmind-technical.html':'technical fixture', 'finmind-fundamental.html':'fundamental fixture'}
    for name, value in sources.items():
        (official/name).write_text(value)
    write(official/'manifest.json', dict(files={name:dict(http_status=200,sha256=sha(official/name))
                                               for name in sources}))
    code_files(tmp_path, ('scripts/audit_external_source_followup.py', 'skills/publication_versions.py'))
    for name, value in (('ROOT',tmp_path),('DIRECTORY',directory),('DEMAND',demand),
                        ('DAILY',daily_path),('OFFICIAL',official)):
        monkeypatch.setattr(external,name,value)
    monkeypatch.setattr(external,'verify_request',lambda path: dict(total_stock_days=1303,procurement_plan={}))
    monkeypatch.setattr(external,'verify_daily',lambda path: dict(input_sha256={},code_sha256={},
        required_stock_days=13,statuses={'official_daily_positive_trade':13},preparation_official_requests=8))
    return directory


def test_http_200_security_response_is_not_a_successful_announcement_archive(external_capture):
    report = external.build(external_capture)
    assert report['mops']['status'] == 'official_security_response'
    assert report['mops']['historical_records_received'] == 0
    assert report['mops']['bypass_attempted'] is False
    assert report['issuer']['current_observed_historical_documents_received'] == 1
    assert report['issuer']['point_in_time_documents_usable_on_20230110'] == 0
    assert report['issuer']['revision_history_received'] is False
    assert report['strict_data_ready'] is report['live_qualified'] is False


def test_http_200_without_the_recorded_security_page_cannot_replay_that_claim(external_capture):
    path = external_capture/'entry.html'
    path.write_text('<html>Unrelated successful page</html>')
    receipt = json.loads((external_capture/'entry.source.json').read_text())
    receipt['sha256'] = sha(path)
    write(external_capture/'entry.source.json', receipt)
    with pytest.raises(ValueError,match='security response'):
        external.build(external_capture)


def test_external_archive_verification_binds_source_metadata(external_capture):
    path = external_capture/'report.json'
    write(path, external.build(external_capture))
    path.with_suffix('.sha256').write_text(sha(path)+'\n')
    assert external.verify_report(path)['finmind_requests'] == 0
    receipt_path = external_capture/'walsin-december-2022.source.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['observed_at'] = '2023-01-09T05:45:48+00:00'
    write(receipt_path, receipt)
    with pytest.raises(ValueError,match='evidence changed'):
        external.verify_report(path)
