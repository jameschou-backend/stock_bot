"""Keep Yulon's September certificates outside ordinary inventory at period end."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
from urllib.parse import urlparse
import unicodedata

import pandas as pd

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest

ROOT = Path(__file__).resolve().parents[1]
KEY = '9941-2026-09-01'
POLICY = 'all_entry_9941_nontraded_certificate_at_period_end_v1'
DOCUMENT = 'docs/all_entry_9941_certificate_20261011.json'
EX_DATE, CERTIFICATE, CONVERSION, END = '2026-09-01', '2026-09-23', '2026-10-16', '2026-10-02'


def load_completion(root, document):
    root, document = Path(root).resolve(), Path(document).resolve()
    sources = Sources(root)
    doc = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    if (doc.get('schema') != POLICY or doc.get('key') != KEY
            or doc.get('strategy_parameters_changed') is not False or doc.get('live_qualified') is not False):
        raise ValueError('Late certificate completion scope differs')
    refs = doc['source_sha256']
    for name, sha in refs.items(): sources.bind(name, sha)
    row = doc['override']
    required = dict(shares_per_share=.02, pay_date=CONVERSION, ordinary_share_available_date=CONVERSION,
        certificate_delivery_date=CERTIFICATE, certificate_trading_modeled=False,
        valuation_basis='ordinary_share_close_proxy', fractional_cash_per_share=10,
        fractional_cash_rounding='floor_ntd', fractional_cash_pay_date=None,
        fractional_cash_payment_date_verified=False, fractional_cash_net_amount_verified=False,
        fractional_cash_available_for_trading=False,
        fractional_cash_treatment='gross_undated_receivable_upper_bound_net_unknown',
        use_scope='account_settlement_only_not_selection')
    if any(row.get(k) != v for k,v in required.items()) or set(row['evidence_files']) != set(refs):
        raise ValueError('Late certificate terms change entitlement or cash availability')
    for kind, host in [('issuer', 'www.tac.com.tw'), ('exchange', 'www.twse.com.tw')]:
        item = doc[kind]
        if any(item[k] not in refs for k in ('path', 'receipt_path')):
            raise ValueError('Late certificate source escapes inventory')
        receipt = json.loads((root/item['receipt_path']).read_text()); url = urlparse(receipt['url'])
        if (url.scheme != 'https' or url.hostname != host or receipt.get('http_status') != 200
                or receipt.get('automatic_redirects_disabled') is not True
                or receipt.get('sha256', receipt.get('raw_sha256')) != refs[item['path']]):
            raise ValueError('Late certificate primary response identity differs')
        if kind == 'issuer':
            if (receipt.get('final_url') != receipt['url'] or not (root/item['path']).read_bytes().startswith(b'%PDF')
                    or item['text_path'] not in refs):
                raise ValueError('Late certificate issuer PDF is not bound')
            text = ''.join(unicodedata.normalize('NFKC', (root/item['text_path']).read_text()).split())
            if any(t not in text for t in ['每仟股無償配發20股', '每股面額10元',
                    '11,512,727股', '依公司法第240條規定改以現金分派之(至元為止)', '本案照案通過']):
                raise ValueError('Issuer minutes do not verify late certificate entitlement')
        elif (receipt.get('accepted') is not True or receipt.get('security_denied') is not False
                or receipt.get('redirect_statuses') != [] or url.path != '/rwd/zh/announcement/LT185'
                or receipt.get('params',{}).get('stockNo') != '9941'):
            raise ValueError('Late certificate exchange receipt differs')
    data = json.loads((root/doc['exchange']['path']).read_text())
    rows = [r for r in data.get('data',[]) if r[0] == '9941']
    certificates = [r for r in rows if r[2] == '新股權利證書上市']
    ordinary = [r for r in rows if r[2] == '(股票上市)新股權利證書終止上市，增資股股票上市']
    if (data.get('stat') != 'OK' or len(certificates) != 1 or len(ordinary) != 1
            or certificates[0][5:8] != ['115年09月15日','115年09月23日','11,512,727']
            or ordinary[0][5:8] != ['115年10月08日','115年10月16日','11,512,727']
            or '普通股11,512,727股' not in ordinary[0][9]
            or '新股權利證書終止上市日期' not in ordinary[0][9]):
        raise ValueError('Late certificate and ordinary listing differ')
    feed = doc['existing_dividend_feed']
    if feed not in refs: raise ValueError('Existing ex-date source is unbound')
    frame = pd.read_parquet(root/feed)
    events = frame.loc[frame.stock_id.eq('9941') & frame.StockExDividendTradingDate.eq(EX_DATE)]
    if (len(events) != 1 or float(events.iloc[0].StockEarningsDistribution) != .2
            or float(events.iloc[0].StockStatutorySurplus) != 0
            or float(events.iloc[0].ParticipateDistributionOfTotalShares) != 575636346):
        raise ValueError('Existing ex-date and final participation basis differ')
    return deepcopy(row), dict(sources.used)


def inventory_audit(case, terms, document_sha):
    if not case.get('completed'): raise ValueError('Partial account has no late certificate proof')
    account = case['account']
    issues = [r for r in account['corporate_actions'] if r.get('stock_id') == '9941'
              and r.get('date') == EX_DATE and r.get('kind') == 'stock_dividend']
    if not issues: return dict(schema=POLICY, document_sha256=document_sha, events=[])
    if len(issues) != 1: raise ValueError('Late certificate entitlement duplicated')
    issue = issues[0]
    restriction = dict(certificate_delivery_date=CERTIFICATE, ordinary_share_available_date=CONVERSION,
        valuation_basis='ordinary_share_close_proxy', certificate_trading_modeled=False)
    if (terms.get('shares_per_share') != .02 or terms.get('pay_date') != CONVERSION
            or terms.get('certificate_trading_modeled') is not False
            or type(issue.get('entitled_qty')) is not int or issue['entitled_qty'] <= 0):
        raise ValueError('Late certificate entitlement basis differs')
    whole = int(Decimal(issue['entitled_qty'])*Decimal('.02'))
    if (issue.get('certificate_restriction') != restriction or issue.get('pay_date') != CONVERSION
            or issue.get('whole_new_shares') != whole or issue.get('shares_per_share') != .02):
        raise ValueError('Late certificate was treated as ordinary shares')
    if any(r.get('action_id') == issue['action_id'] and r.get('kind') == 'share_delivery'
           for r in account['corporate_actions']):
        raise ValueError('Ordinary conversion is after this account period')
    claims = [r for r in account['receivables'] if r.get('action_id') == issue['action_id']]
    if ((whole and len(claims) != 1) or len(claims) > 1
            or any(r.get('kind') != 'shares' or r.get('stock_id') != '9941'
                   or r.get('qty') != whole or r.get('pay_date') != CONVERSION for r in claims)):
        raise ValueError('Late certificate units are not preserved as restricted receivables')
    if any(r.get('action_id') == issue['action_id']+'-fractional-cash' for r in account['cash_ledger']):
        raise ValueError('Unverified fractional cash was paid')
    days = [r['date'] for r in account['daily'] if EX_DATE <= r['date'] <= END]
    if not days or days[0] != EX_DATE or days[-1] != END or CERTIFICATE not in days:
        raise ValueError('Late certificate period is incomplete')
    if any(r.get('stock_id') == '9941' and EX_DATE <= r.get('date','') <= END
           and r.get('kind') in ('stock_dividend','share_delivery','split','capital_reduction')
           and r.get('action_id') != issue['action_id'] for r in account['corporate_actions']):
        raise ValueError('Overlapping unit change needs separate certificate review')
    ordinary, journal = issue['entitled_qty'], []
    for day in days:
        for t in account['trades']:
            if t['stock_id'] == '9941' and t['date'] == day:
                ordinary += t['qty'] if t['side'] == 'buy' else -t['qty']
                if ordinary < 0: raise ValueError('Late certificate units sold as ordinary shares')
        actual = sum(r['qty'] for r in account['holdings'] if r['date'] == day and r['stock_id'] == '9941')
        if actual != ordinary: raise ValueError('Late certificate units contaminate ordinary holdings')
        journal.append(dict(date=day, state='pending_entitlement' if day < CERTIFICATE else 'nontraded_certificate_inventory',
            restricted_units=whole, converted_units=0, ordinary_closing_units=ordinary,
            certificate_trades=0, certificate_cash_available=0))
    return dict(schema=POLICY, document_sha256=document_sha, events=[dict(stock_id='9941',
        action_id=issue['action_id'], certificate_delivery_date=CERTIFICATE, ordinary_conversion_date=CONVERSION,
        whole_units=whole, ending_restricted_units=whole, valuation_basis='ordinary_share_close_proxy',
        certificate_trading_modeled=False, daily=journal)])


def verify_inventory_proof(case, root=ROOT):
    terms, _ = load_completion(root, Path(root)/DOCUMENT)
    expected = inventory_audit(case, terms, digest(Path(root)/DOCUMENT))
    if not expected['events'] or case.get('late_certificate_inventory_audit') != expected:
        raise ValueError('Independent late certificate unit journal is absent or differs')
    return expected


def install_completion(runner, provider, document, root):
    terms, refs = load_completion(root, document)
    if KEY in provider.corporate_overrides and provider.corporate_overrides[KEY] != terms:
        raise ValueError('Late certificate completion conflicts with existing terms')
    provider.corporate_overrides = dict(provider.corporate_overrides, **{KEY:terms})
    writer = runner.write
    def write(path, value):
        if isinstance(value, dict) and value.get('completed') is True and 'account' in value:
            value = dict(value, late_certificate_inventory_audit=inventory_audit(value, terms, digest(document)))
        return writer(path, value)
    runner.write = write
    return refs
