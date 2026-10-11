"""Verified ordinary-share delivery additions, without changing execution rules."""
from copy import deepcopy
from datetime import date
from decimal import Decimal
import json
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from scripts.summarize_poc_broker_account import Sources


def load_terms(root, document):
    root = Path(root).resolve()
    document = Path(document).resolve()
    document.relative_to(root)
    sources = Sources(root)
    from skills.all_entry_accounts import digest
    value = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    if (value.get('schema') != 'all_entry_ordinary_delivery_v1'
            or value.get('strategy_parameters_changed') is not False
            or value.get('live_qualified') is not False
            or set(value.get('overrides', {})) != set(value.get('evidence', {}))):
        raise ValueError('Unexpected ordinary delivery supplement scope')
    for key, row in value['overrides'].items():
        sid, ex = key[:4], date.fromisoformat(key[5:])
        facts = value['evidence'][key]
        if (len(sid) != 4 or not sid.isdigit()
                or Decimal(str(row['shares_per_share'])) != Decimal(facts['shares_per_1000']) / 1000
                or Decimal(str(row['fractional_cash_per_share'])) != Decimal(facts['par_value'])
                or Decimal(facts['par_value']) <= 0 or Decimal(facts['shares_per_1000']) <= 0
                or facts.get('same_rights_as_ordinary') is not True
                or row.get('ordinary_share_available_date') != row.get('pay_date')):
            raise ValueError('Ordinary delivery rate or instrument differs')
        dates = [date.fromisoformat(row[k]) for k in ('entitlement_announcement_date',
                 'delivery_announcement_date', 'listing_announcement_date', 'pay_date')]
        if not dates[0] <= ex <= dates[1] <= dates[2] <= dates[3]:
            raise ValueError('Ordinary delivery chronology differs')
        if (row.get('use_scope') != 'account_settlement_only_not_selection'
                or row.get('fractional_cash_rounding') != 'floor_ntd'
                or row.get('fractional_cash_pay_date') is not None
                or any(row.get(k) is not False for k in ('fractional_cash_payment_date_verified',
                   'fractional_cash_net_amount_verified', 'fractional_cash_available_for_trading'))
                or row.get('fractional_cash_treatment') != 'gross_undated_receivable_upper_bound_net_unknown'):
            raise ValueError('Unverified fractional cash cannot be spent')
        notices = facts['sources']
        if sorted(s['purpose'] for s in notices) != ['delivery', 'entitlement', 'listing']:
            raise ValueError('Three primary ordinary delivery notices are required')
        for notice in notices:
            path = sources.bind(notice['path'], notice['sha256'])
            receipt = json.loads(sources.bind(notice['receipt_path'], notice['receipt_sha256']).read_text())
            announced = date.fromisoformat(notice['announcement_date'])
            index = {'entitlement': 0, 'delivery': 1, 'listing': 2}[notice['purpose']]
            url = urlparse(notice['url'])
            announcement = receipt.get('announcement_row', [])
            if (url.scheme != 'https' or url.hostname != 'mopsov.twse.com.tw'
                    or receipt.get('url') != notice['url'] or receipt.get('http_status') != 200
                    or receipt.get('redirected') is not False or receipt.get('sha256') != notice['sha256']
                    or len(announcement) < 3 or announcement[0] != sid
                    or announcement[2] != f'{announced.year-1911:03d}/{announced.month:02d}/{announced.day:02d}'
                    or announced != dates[index]):
                raise ValueError('Primary ordinary delivery receipt differs')
            text = ''.join(BeautifulSoup(path.read_bytes(), 'html.parser').get_text().split())
            if not notice.get('required_text') or any(''.join(t.split()) not in text for t in notice['required_text']):
                raise ValueError('Primary notice does not contain verified delivery facts')
        if set(row['evidence_files']) != {s[k] for s in notices for k in ('path', 'receipt_path')}:
            raise ValueError('Ordinary delivery evidence closure differs')
    return deepcopy(value['overrides']), dict(sources.used)
