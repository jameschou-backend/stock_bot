"""Two primary-source distributions, using the existing settlement-only rules."""
from copy import deepcopy
from datetime import date
from decimal import Decimal
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest

EXPECTED = {
    '1477-2024-09-12': (20, 4837132, '2024-06-28', '2024-10-04', '2024-10-09', '2024-10-18'),
    '2834-2026-08-04': (70, 680264330, '2026-06-18', '2026-08-28', '2026-09-07', '2026-09-10'),
}


def roc_day(value):
    match = re.fullmatch(r'(\d{3})年(\d{2})月(\d{2})日', value)
    if not match:
        raise ValueError('Unexpected exchange announcement date')
    year, month, day = map(int, match.groups())
    return date(year + 1911, month, day).isoformat()


def load_distributions(root, document):
    root, document = Path(root).resolve(), Path(document).resolve()
    sources = Sources(root)
    doc = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    if (doc.get('schema') != 'all_entry_distribution_completion_v1'
            or doc.get('strategy_parameters_changed') is not False
            or doc.get('live_qualified') is not False
            or doc.get('source_policy') != 'issuer_minutes_and_exchange_listing_primary_only'
            or set(doc.get('overrides', {})) != set(EXPECTED)
            or set(doc.get('evidence', {})) != set(EXPECTED)):
        raise ValueError('Distribution completion scope differs')
    for key, row in doc['overrides'].items():
        sid = key[:4]
        facts = doc['evidence'][key]
        per1000, quantity, entitlement, delivery, listing, pay = EXPECTED[key]
        if (facts.get('shares_per_1000') != per1000 or facts.get('par_value') != 10
                or Decimal(str(row['shares_per_share'])) != Decimal(per1000)/1000
                or [row.get(k) for k in ('entitlement_announcement_date', 'delivery_announcement_date',
                    'listing_announcement_date', 'pay_date')] != [entitlement, delivery, listing, pay]
                or row.get('ordinary_share_available_date') != pay
                or row.get('use_scope') != 'account_settlement_only_not_selection'
                or row.get('fractional_cash_rounding') != 'floor_ntd'
                or row.get('fractional_cash_pay_date') is not None
                or row.get('fractional_cash_payment_date_verified') is not False
                or row.get('fractional_cash_available_for_trading') is not False):
            raise ValueError('Distribution rate, dates or cash availability differ')
        refs = facts['source_sha256']
        if set(row['evidence_files']) != set(refs):
            raise ValueError('Distribution evidence inventory differs')
        for name, sha in refs.items():
            sources.bind(name, sha)
        issuer, exchange = facts['issuer'], facts['listing']
        for item, expected_host in ((issuer, 'www.makalot.com.tw' if sid == '1477' else 'ir.tbb.com.tw'),
                                    (exchange, 'www.twse.com.tw')):
            if any(item[k] not in refs for k in ('path', 'receipt_path')):
                raise ValueError('Distribution source escapes its evidence inventory')
            receipt = json.loads((root/item['receipt_path']).read_text())
            url = urlparse(receipt['url'])
            source_sha = receipt.get('sha256', receipt.get('raw_sha256'))
            if (url.scheme != 'https' or url.hostname != expected_host
                    or receipt.get('http_status') != 200 or source_sha != refs[item['path']]):
                raise ValueError('Distribution source is not the exact successful primary response')
            if item is issuer:
                if receipt.get('redirected') is not False or not (root/item['path']).read_bytes().startswith(b'%PDF'):
                    raise ValueError('Issuer minutes are not the verified PDF')
            elif (receipt.get('accepted') is not True or receipt.get('security_denied') is not False
                    or receipt.get('automatic_redirects_disabled') is not True
                    or url.path != '/rwd/zh/announcement/LT185'
                    or receipt.get('params', {}).get('stockNo') != sid):
                raise ValueError('Exchange listing receipt differs')
        if issuer['text_path'] not in refs or issuer['extraction_command'] != 'pdftotext -layout':
            raise ValueError('Reviewed PDF extraction is not bound')
        text = ''.join((root/issuer['text_path']).read_text().split())
        required = issuer['required_text']
        if not required or any(''.join(x.split()) not in text for x in required):
            raise ValueError('Reviewed issuer minutes do not contain the distribution facts')
        data = json.loads((root/exchange['path']).read_text())
        rows = data.get('data', [])
        ordinary = [r for r in rows if r[0] == sid and r[2].startswith('(股票上市)')]
        if (data.get('stat') != 'OK' or len(ordinary) != 1
                or roc_day(ordinary[0][5]) != listing or roc_day(ordinary[0][6]) != pay
                or int(ordinary[0][7].replace(',', '')) != quantity
                or exchange['ordinary_quantity'] != quantity
                or '與原已發行之股份相同' not in ordinary[0][9]
                or f'{int(delivery[:4])-1911}年{delivery[5:7]}月{delivery[8:]}日' not in ordinary[0][9]):
            raise ValueError('Confirmed ordinary-share listing differs from settlement')
        if sid == '1477':
            if (row.get('fractional_cash_per_share') != 0
                    or row.get('fractional_net_zero_verified') is not True
                    or row.get('fractional_cash_net_amount_verified') is not True
                    or row.get('fractional_cash_treatment') != 'net_zero_book_entry_fees_verified'
                    or '其未滿一股之畸零股款，將做為處理帳簿劃撥之費用' not in text
                    or 'certificate_delivery_date' in row):
                raise ValueError('Book-entry fractional fees must not become a cash asset')
        else:
            certificates = [r for r in rows if r[0] == sid and r[2] == '新股權利證書上市']
            if (len(certificates) != 1 or roc_day(certificates[0][6]) != '2026-09-02'
                    or int(certificates[0][7].replace(',', '')) != quantity
                    or row.get('certificate_delivery_date') != '2026-09-02'
                    or row.get('certificate_trading_modeled') is not False
                    or row.get('valuation_basis') != 'ordinary_share_close_proxy'
                    or facts.get('certificate_scope') != 'separate_nontradable_stock_receivable_until_ordinary_conversion'
                    or row.get('fractional_cash_per_share') != 10
                    or row.get('fractional_cash_net_amount_verified') is not False
                    or row.get('fractional_cash_treatment') != 'gross_undated_receivable_upper_bound_net_unknown'):
                raise ValueError('Certificate conversion or unverified fractional cash differs')
    return deepcopy(doc['overrides']), dict(sources.used)


def install_distributions(provider, document, root):
    overrides, refs = load_distributions(root, document)
    for key, row in overrides.items():
        if key in provider.corporate_overrides and provider.corporate_overrides[key] != row:
            raise ValueError('Distribution completion would rewrite an existing event')
    provider.corporate_overrides = dict(provider.corporate_overrides, **overrides)
    return refs
