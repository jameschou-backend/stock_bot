"""Two primary-source trading exclusions; never manufacture quotes or quality.

Legal listing identity, strategy trading scope and input-quality evidence remain
separate. The managed-stock publication is retrospective historical evidence.
"""
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from skills.historical_universe_completion import validate_exclusions
from skills.market_input_validation import require
from skills.official_market_classification import sha


SCHEMA = 'historical_trading_repair_v1'
REVIEWED_SOURCES = {
    'managed_period': 'fc85ad93d5528d9248f973e6ac20183a4642d1d74feb43b30136e7eb143dc3d9',
    'managed_daily': '1f931b51bcc71b78117457f66148a2e11690c7905b406e34f8398fc364777a1f',
    'managed_daily_receipt': '48e4f1a1a4eeee02f7d5f6e8552cc4d35161f43dc7db0135246e0e85b69495ae',
    'face_announcement': '8d781df6511ef4f9b6ee43d192d5a329519d766a106c2dbeabbcc476c0327a9c',
    'face_announcement_receipt': 'f5dea20fd67aeba31a64f33c040e292564b378f5bc8679e9e2d1d7e11ae70cbb',
    'face_restore': 'd6439d0e1c14d7300dbe1b3af8a4a18f64b586f3d356d408eee7ea916bc28b90',
    'face_summary': 'a89c001553fb9410d7d197995abc8710b645227c71019aefa04e2dd52a74d521',
    'ordinary_gap_audit': 'c23ac39301495d826edeca8978dad99071b1d303ff68e1ef0d2c933b6caee827',
    'daily_scope_excerpt': '100ba9c9918751a13684cd15a330bc572c0a6bf910690fa43e3a95dc3d60417d',
}
PERIODS = {
    '4415': ('TPEx', 'managed_board', '2017-11-21', '2019-12-16'),
    '7780': ('TWSE', 'trading_suspension', '2026-01-09', '2026-01-19'),
}
TAIFEX_URL = 'https://www.taifex.com.tw/file/taifex/eng/eng11/%E6%96%B0%E8%81%9E%E7%A8%BF1081129doc.pdf'
MOPS_URL = 'https://mopsov.twse.com.tw/mops/web/t05st02'
DAILY_URL = 'https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes'


def _path(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'Relative trading source required')
    path = (root / name).resolve()
    require(path.is_relative_to(root) and path.is_file(), 'Trading source missing or escapes root')
    return path


def _payload_hash(value):
    clean = {k: v for k, v in value.items() if k != 'verified_manifest'}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class _HTMLText(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.parts = []
        self.feed(text)

    def handle_data(self, text):
        self.parts.append(text)


def _validate(value, root):
    require(value.get('schema') == SCHEMA, 'Unknown trading repair schema')
    for key in ('live_qualified', 'publication_time_archive_complete',
                'complete_historical_universe', 'performance_recomputed'):
        require(value.get(key) is False, 'Trading evidence cannot promote qualification')
    closure = value['source_sha256']
    require(isinstance(closure, dict) and closure, 'Trading source closure missing')
    for name, digest in closure.items():
        require(sha(_path(root, name)) == digest, 'Trading source hash changed')
    sources = value['sources']
    require(set(sources) == set(REVIEWED_SOURCES), 'Trading evidence sources changed')
    for key, spec in sources.items():
        require(spec['sha256'] == closure.get(spec['path']) == REVIEWED_SOURCES[key],
                'Trading source needs independent review')

    def read(key):
        return json.loads(_path(root, sources[key]['path']).read_text())

    text_source = read('managed_period')
    require(text_source['source_kind'] == 'official_pdf_web_extraction'
            and text_source['source_url'] == TAIFEX_URL
            and text_source['raw_pdf_bytes_obtained'] is False
            and text_source['pdf_pages_1based'] == [1]
            and text_source['publication_date'] == '2019-11-29',
            'Managed-stock publication provenance differs')
    text = re.sub(r'\s+', '', re.sub(r'L\d+@P\d+:', '', text_source['extraction']))
    require('4415台原藥公司' in text
            and '普通股股票自106年11月21日開始為櫃檯買賣管理股票' in text
            and '自108年12月16日起終止該公司之有價證券櫃檯買賣' in text,
            'Managed-stock interval is not explicit in primary publication')
    raw, receipt = read('managed_daily'), read('managed_daily_receipt')
    require(receipt['url'] == DAILY_URL and receipt['params'] == {'date': '2019/07/03', 'response': 'json'}
            and receipt['http_status'] == 200 and receipt['raw_sha256'] == sources['managed_daily']['sha256']
            and raw['date'] == '20190703' and raw['stat'] == 'ok', 'Managed daily source date differs')
    tables = [t for t in raw['tables'] if t['title'] == '管理股票']
    require(len(tables) == 1, 'Managed daily table ambiguous')
    table = tables[0]
    require(table['totalCount'] == len(table['data']), 'Managed daily table incomplete')
    rows = [dict(zip(table['fields'], r)) for r in table['data'] if r[0] == '4415']
    require(len(rows) == 1 and rows[0]['成交股數'] == '2,364'
            and all(rows[0][k] == '6.00' for k in ('開盤', '最高', '最低', '收盤')),
            'Managed daily source row changed')
    scope = read('daily_scope_excerpt')
    require(scope['source_kind'] == 'official_page_search_excerpt'
            and scope['raw_html_obtained'] is False and scope['original_tool_output_saved'] is False
            and scope['excerpt'] == '上櫃股票行情(含等價、零股、盤後、鉅額交易)',
            'Daily-report scope is an indexed excerpt, not raw HTML evidence')
    receipt = read('face_announcement_receipt')
    require(receipt['url'] == MOPS_URL and receipt['http_status'] == 200
            and receipt['raw_sha256'] == sources['face_announcement']['sha256']
            and all(receipt['params'].get(k) == v for k, v in
                    {'h311': '7780', 'h312': '20251222', 'h313': '175130', 'h315': '1'}.items()),
            'Issuer announcement receipt differs')
    html = _path(root, sources['face_announcement']['path']).read_bytes()
    text = re.sub(r'\s+', '', ''.join(_HTMLText(html.decode('utf-8')).parts))
    for literal in ('7780', '發言日期114/12/22', '發言時間17:51:30',
                    '舊股票最後交易日:民國115年1月8日',
                    '舊股票停止交易期間:民國115年1月9日至115年1月17日',
                    '新股票上市買賣日:民國115年1月19日'):
        require(literal in text, 'Issuer trading halt dates differ')
    restore = read('face_restore')
    require(any(r[:2] == ['115/01/19', '7780'] and r[-1] == '7780,20260109,20260119'
                for r in restore['data']), 'Exchange resumption corroboration differs')
    entries = value['entries']
    require(len(entries) == 2 and {e['stock_id'] for e in entries} == set(PERIODS),
            'Unsupported or duplicate trading exclusions')
    for e in entries:
        sid = e['stock_id']
        require(tuple(e[k] for k in ('market', 'kind', 'start', 'end')) == PERIODS[sid]
                and e['interval'] == 'start_inclusive_end_exclusive', 'Unproven trading interval')
        require(e['source_path'] == sources['managed_period' if sid == '4415' else 'face_announcement']['path'],
                'Exclusion evidence link differs')
        if sid == '4415':
            require(e['legal_security_type'] == 'ordinary_share'
                    and e['publication_date'] == '2019-11-29', 'Managed shares remain ordinary legal shares')
        else:
            require(e['announcement_at'] == '2025-12-22T17:51:30+08:00'
                    and e['announcement_date'] == '2025-12-22' and e['known_by'] == '2025-12-23'
                    and e['announced_last_old_trading_date'] == '2026-01-08'
                    and e['announced_suspension_last_date_inclusive'] == '2026-01-17'
                    and e['announced_new_trading_date'] == '2026-01-19', 'Halt timing differs')
    validate_exclusions(entries)
    # These diagnostics do not grant independent factor/price quality.
    gaps = value['quote_gaps_4415']
    expected = {'2019-06-14', '2019-06-25', '2019-06-27', '2019-07-01', '2019-07-02',
                '2019-07-03', '2019-07-04', '2019-07-08', '2019-07-09'}
    require(len(gaps) == 9 and {r['date'] for r in gaps} == expected, '4415 gap diagnostics differ')
    audit = {r['quote']['date']: r for r in read('ordinary_gap_audit')['missing_positive_quotes']['ordinary_quotes']
             if r['quote']['stock_id'] == '4415'}
    require(set(audit) == expected, 'Original quote-gap audit differs')
    for row in gaps:
        require(row['stock_id'] == '4415' and row['independent_adjusted_price_verified'] is False
                and row['quality_gate_passed'] is False
                and row['total_daily_shares'] == (2364 if row['date'] == '2019-07-03' else None),
                'Missing total volume or independent quality cannot be promoted')
        original = audit[row['date']]
        descriptor = original['source_descriptor']
        require(row['raw_ohlc'] == {k: original['quote'][k] for k in ('open', 'high', 'low', 'close')}
                and row['ordinary_session_shares'] == original['ordinary_session_shares']
                and row['ordinary_source'] == descriptor
                and closure.get(descriptor['path']) == descriptor['sha256']
                and closure.get(descriptor['receipt']) == descriptor['receipt_sha256']
                and row['ordinary_quote_not_promoted_to_total'] is True,
                'Original ordinary price/volume lineage differs')
        known = row['date'] == '2019-07-03'
        require(row['total_source'] == (sources['managed_daily'] if known else None)
                and row['total_volume_scope'] == ('dailyQuotes_reported_all_sessions' if known else 'unknown'),
                'Daily-report volume lineage differs')


def load_trading_repair(manifest_path, root, refs=None):
    """Verify reviewed primary sources and expose full archive closure."""
    root = Path(root).resolve()
    path = Path(manifest_path)
    path = (path if path.is_absolute() else root / path).resolve()
    require(path.is_relative_to(root) and path.is_file(), 'Trading manifest escapes root')
    digest = sha(path)
    require(path.with_suffix('.sha256').read_text().strip() == digest, 'Trading manifest hash changed')
    value = json.loads(path.read_text())
    _validate(value, root)
    closure = dict(value['source_sha256'], **{str(path.relative_to(root)): digest})
    if refs is not None:
        require(all(k not in refs or refs[k] == v for k, v in closure.items()), 'Trading closure conflict')
        refs.update(closure)
    value['verified_manifest'] = dict(path=str(path.relative_to(root)), sha256=digest,
                                      payload_sha256=_payload_hash(value))
    return value


def apply_trading_repair(base, repair):
    """Copy a listing report, preserving listing episodes and all quote gaps."""
    verified = repair.get('verified_manifest', {})
    require(verified.get('payload_sha256') == _payload_hash(repair), 'Use unchanged loaded trading repair')
    require('trading_repair' not in base, 'Trading repair already applied')
    result = deepcopy(base)
    old = result['trading_exclusions']
    for row in repair['entries']:
        sid, venue = row['stock_id'], row['market'].upper()
        episodes = [ep for ep in result['episodes'] if ep['stock_id'] == sid and ep['market'].upper() == venue
                    and ep['category'] == '股票' and ep['start'] is not None
                    and ep['start'] < row['end'] and (ep['end'] is None or ep['end'] > row['start'])]
        require(episodes, 'Trading repair needs overlapping verified ordinary listing')
        # The listing may already have been repaired by the independent IPO
        # overlay. It must span the complete suspension, not a current ISIN date.
        if sid == '7780':
            require(any(ep['start'] <= row['start'] and (ep['end'] is None or ep['end'] >= row['end'])
                        for ep in episodes), 'Repair 7780 IPO before applying its trading suspension')
        for existing in old:
            if existing['stock_id'] == sid and existing['market'].upper() == venue:
                require(not (existing['start'] < row['end'] and
                             (existing['end'] is None or existing['end'] > row['start'])),
                        'Overlapping existing trading exclusion requires review')
    result['trading_exclusions'] = validate_exclusions(old + deepcopy(repair['entries']))
    result['trading_repair'] = deepcopy(repair)
    result['source_sha256'] = dict(base.get('source_sha256', {}))
    closure = dict(repair['source_sha256'], **{verified['path']: verified['sha256']})
    require(all(k not in result['source_sha256'] or result['source_sha256'][k] == v
                for k, v in closure.items()), 'Trading closure conflict')
    result['source_sha256'].update(closure)
    result.update(live_qualified=False, complete_historical_universe=False,
                  continuous_eligibility_proven=False, publication_time_archive_complete=False,
                  performance_recomputed=False)
    return result
