"""Bind Yageo's announced four-for-one split and full-session suspension."""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest
from skills.all_entry_bank_distributions import normalized

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = 'docs/all_entry_2327_split_20261012.json'
KEY, DAY = '2327-2025-08-25', '2025-08-25'
URL = 'https://www.taifex.com.tw/file/taifex/CHINESE/11/attach/2327_20250825.pdf'


def load_split(root, document):
    root, document = Path(root).resolve(), Path(document).resolve()
    sources = Sources(root)
    doc = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    if (doc.get('schema') != 'all_entry_2327_split_halt_v1'
            or doc.get('strategy_parameters_changed') is not False
            or doc.get('live_qualified') is not False or doc.get('reviewed_pdf_pages') != [1, 3]):
        raise ValueError('Yageo split evidence scope differs')
    for name, sha in doc['source_sha256'].items(): sources.bind(name, sha)
    for key in ('notice_path', 'receipt_path', 'text_path', 'events_path'):
        if doc[key] not in sources.used: raise ValueError('Unbound split evidence')
    notice = root / doc['notice_path']; receipt = json.loads((root / doc['receipt_path']).read_text())
    if (receipt.get('url') != URL or receipt.get('final_url') != URL
            or receipt.get('http_status') != 200 or receipt.get('sha256') != digest(notice)
            or receipt.get('automatic_redirects_disabled') is not True
            or not notice.read_bytes().startswith(b'%PDF')):
        raise ValueError('Yageo primary notice receipt differs')
    text = normalized((root / doc['text_path']).read_text())
    required = ['中華民國114年8月6日', '1140014444', '2327',
                '每壹仟股換發新股票肆仟股', '調整生效日：114年8月25日',
                '停止交易期間：114年8月14日至114年8月22日', '恢復交易日：114年8月25日']
    if any(normalized(t) not in text for t in required):
        raise ValueError('Yageo primary notice does not support split terms')
    events = pd.read_parquet(root / doc['events_path'])
    rows = events[events.stock_id.eq('2327') & pd.to_datetime(events.event_date).eq(DAY)]
    if len(rows) != 1: raise ValueError('Yageo official event is not unique')
    event = rows.iloc[0]
    if (event['source'] != 'par_value_change' or event['market'] != 'TWSE'
            or event['prev_close'] != 546. or event['opening_ref'] != 136.5
            or event['ratio'] != .25 or event['ratio_opening'] != .25
            or json.loads(event['payload_json']) != {'detail': '2327,20250814,20250825'}):
        raise ValueError('Yageo official event disagrees with split and suspension')
    action = dict(kind='split', multiplier=4,
                  source=URL, announcement_date='2025-08-06',
                  use_scope='account_settlement_only_not_selection')
    halt = dict(stock_id='2327', market='TWSE', kind='trading_suspension',
                start='2025-08-14', end=DAY, announcement_date='2025-08-06',
                source_path=doc['notice_path'], source_sha256=digest(notice),
                source='TAIFEX primary notice citing TWSE 1140014444; sealed TWSE event dates')
    return action, halt, dict(sources.used)


def install_split(runner, provider, document, root):
    action, halt, refs = load_split(root, document)
    if KEY in provider.corporate_overrides:
        raise ValueError('Yageo split conflicts with prior corporate terms')
    provider.corporate_overrides = dict(provider.corporate_overrides, **{KEY: action})
    target = Path(root).resolve() / 'docs/stock_universe_2019_corporate_terms.json'
    original = runner.read
    def read(path):
        value = original(path)
        if Path(path).resolve() != target: return value
        value = deepcopy(value)
        if any(h['stock_id'] == '2327' and h['start'] < halt['end'] and halt['start'] < h['end']
               for h in value['verified_halts']):
            raise ValueError('Yageo suspension overlaps prior terms')
        value['verified_halts'].append(deepcopy(halt))
        for name, sha in refs.items():
            if name in value['source_sha256'] and value['source_sha256'][name] != sha:
                raise ValueError('Yageo evidence conflicts with prior source')
        value['source_sha256'].update(refs)
        return value
    runner.read = read
    return refs
