"""Replace one exhausted pending-delivery proof with final issuer settlement.

The previous action is hash-bound and its ratio and zero net fractional cash
cannot change. Ordinary shares become available only on the actual delivery
date, after this study's endpoint. This is retrospective account settlement
evidence, never an input into historical candidate selection.
"""
from copy import deepcopy
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from skills.all_entry_accounts import digest
from scripts.summarize_poc_broker_account import Sources


def terms_hash(row):
    return sha256(json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def load_completion(root, document):
    root = Path(root).resolve()
    document = Path(document).resolve()
    sources = Sources(root)
    value = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    key = value.get('key')
    if (value.get('schema') != 'all_entry_pending_delivery_completion_v1'
            or key != '6669-2026-09-02' or value.get('strategy_parameters_changed') is not False
            or value.get('live_qualified') is not False):
        raise ValueError('Unexpected pending-delivery completion scope')
    previous = value['previous_document']
    old = json.loads(sources.bind(previous['path'], previous['sha256']).read_text())['overrides'][key]
    row = value['replacement']
    if (terms_hash(old) != value['previous_terms_sha256'] or old.get('pending_only') is not True
            or old.get('pay_date') is not None or row.get('pending_only')
            or row.get('shares_per_share') != old['shares_per_share']
            or row.get('shares_per_share') != 1.9827946
            or row.get('fractional_cash_per_share') != old['fractional_cash_per_share']
            or row.get('fractional_cash_per_share') != 0
            or row.get('fractional_net_zero_verified') is not True
            or old.get('fractional_net_zero_verified') is not True
            or row.get('fractional_cash_rounding') != 'floor_ntd'
            or row.get('fractional_cash_pay_date') is not None
            or row.get('use_scope') != 'account_settlement_only_not_selection'
            or row.get('ordinary_share_available_date') != row.get('pay_date')
            or row.get('pay_date') != '2026-10-14'):
        raise ValueError('Completion changes old entitlement or invents settlement')
    expected = dict(entitlement='2026-08-19', delivery='2026-10-02', listing='2026-10-08')
    notices = value['sources']
    if sorted(n['purpose'] for n in notices) != sorted(expected):
        raise ValueError('All three final primary notices are required')
    for notice in notices:
        announced = date.fromisoformat(notice['announcement_date'])
        if (notice['announcement_date'] != expected[notice['purpose']]
                or row[notice['purpose'] + '_announcement_date'] != notice['announcement_date']):
            raise ValueError('Settlement notice chronology differs')
        path = sources.bind(notice['path'], notice['sha256'])
        receipt = json.loads(sources.bind(notice['receipt_path'], notice['receipt_sha256']).read_text())
        url = urlparse(notice['url'])
        announcement = receipt.get('announcement_row', [])
        if (url.scheme != 'https' or url.hostname != 'mopsov.twse.com.tw'
                or receipt.get('url') != notice['url'] or receipt.get('http_status') != 200
                or receipt.get('redirected') is not False or receipt.get('sha256') != notice['sha256']
                or len(announcement) < 3 or announcement[0] != '6669'
                or announcement[2] != f'{announced.year-1911:03d}/{announced.month:02d}/{announced.day:02d}'):
            raise ValueError('Final settlement primary receipt differs')
        text = ''.join(BeautifulSoup(path.read_bytes(), 'html.parser').get_text().split())
        if not notice.get('required_text') or any(''.join(t.split()) not in text for t in notice['required_text']):
            raise ValueError('Final notice does not prove settlement facts')
    if set(row['evidence_files']) != {n[k] for n in notices for k in ('path', 'receipt_path')}:
        raise ValueError('Final settlement evidence closure differs')
    return key, deepcopy(old), deepcopy(row), dict(sources.used)


def install_completion(runner, provider, document, root):
    key, previous, replacement, refs = load_completion(root, document)
    original = runner.merge_overrides

    def merge(base, additions):
        if key in additions and additions[key] == replacement and key in base and base[key] != replacement:
            if base[key] != previous:
                raise ValueError('Pending terms changed before final settlement completion')
            base = {k: v for k, v in base.items() if k != key}
        return original(base, additions)

    current = provider.corporate_overrides.get(key)
    if current is not None and current not in (previous, replacement):
        raise ValueError('Provider has conflicting settlement terms')
    provider.corporate_overrides = dict(provider.corporate_overrides, **{key: replacement})
    runner.merge_overrides = merge
    return refs
