"""Separate 5876's non-traded certificate units from ordinary inventory."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import unicodedata
from urllib.parse import urlparse, parse_qs, unquote

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest
from skills.all_entry_settlement_completion import terms_hash

KEY = '5876-2026-07-28'
POLICY = 'all_entry_5876_nontraded_certificate_inventory_v1'
DOCUMENT = 'docs/all_entry_certificate_completion_20261011.json'
ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = '2026-08-20'
CONVERSION = '2026-09-11'


def replacement(previous):
    row = deepcopy(previous)
    row.update(pay_date=CONVERSION, ordinary_share_available_date=CONVERSION,
        certificate_trading_modeled=False, same_code_fungible_trading_verified=False,
        valuation_basis='ordinary_share_close_proxy',
        source='Issuer entitlement and TWSE listing/conversion; separate non-traded certificate units until ordinary conversion')
    return row


def load_certificates(root, document):
    root, document = Path(root).resolve(), Path(document).resolve()
    sources = Sources(root)
    doc = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    if (doc.get('schema') != POLICY or doc.get('key') != KEY
            or doc.get('strategy_parameters_changed') is not False
            or doc.get('live_qualified') is not False):
        raise ValueError('Certificate completion scope differs')
    old = doc['previous_document']
    previous = json.loads(sources.bind(old['path'], old['sha256']).read_text())['overrides'][KEY]
    if (terms_hash(previous) != doc['previous_terms_sha256']
            or previous['shares_per_share'] != .01 or previous['fractional_cash_per_share'] != 10
            or previous['pay_date'] != CERTIFICATE or previous['ordinary_conversion_date'] != CONVERSION
            or doc['replacement'] != replacement(previous)):
        raise ValueError('Certificate correction changes entitlement or unrelated cash terms')
    refs = doc['source_sha256']
    if not set(previous['evidence_files']).issubset(refs):
        raise ValueError('Previous entitlement evidence is not preserved')
    for name, sha in refs.items(): sources.bind(name, sha)
    for kind, host in [('issuer', 'www.scsb.com.tw'), ('exchange', 'www.twse.com.tw')]:
        item = doc[kind]
        if any(item[k] not in refs for k in ('path', 'receipt_path')):
            raise ValueError('Certificate evidence escapes its inventory')
        receipt = json.loads((root/item['receipt_path']).read_text())
        url = urlparse(receipt['url'])
        if (url.scheme != 'https' or url.hostname != host or receipt.get('status_code') != 200
                or unquote(receipt['url']) != unquote(receipt.get('final_url', ''))
                or receipt['sha256'] != refs[item['path']]):
            raise ValueError('Certificate primary source identity differs')
        if kind == 'issuer':
            if not (root/item['path']).read_bytes().startswith(b'%PDF') or item['text_path'] not in refs:
                raise ValueError('Certificate issuer PDF is not bound')
            text = ''.join(unicodedata.normalize('NFKC', (root/item['text_path']).read_text()).split())
            needles = ['每壹仟股無償配發法定盈餘公積、資本公積轉增資10.00000000股',
                '增資新股權利證書發放日為115年08月20日', '元以下捨去']
            if any(x not in text for x in needles):
                raise ValueError('Certificate issuer entitlement differs')
        elif (url.path != '/rwd/zh/announcement/LT185'
                or parse_qs(url.query).get('stockNo') != ['5876']):
            raise ValueError('Certificate exchange query differs')
    data = json.loads((root/doc['exchange']['path']).read_text())
    if data.get('stat') != 'OK': raise ValueError('Certificate exchange response failed')
    rows = [r for r in data['data'] if r[0] == '5876']
    certificates = [r for r in rows if r[2] == '新股權利證書上市']
    ordinary = [r for r in rows if r[2] == '(股票上市)新股權利證書終止上市，增資股股票上市']
    if (len(certificates) != 1 or len(ordinary) != 1
            or certificates[0][5:8] != ['115年08月17日','115年08月20日','48,616,031']
            or ordinary[0][5:8] != ['115年09月08日','115年09月11日','48,616,031']
            or '普通股48,616,031股' not in ordinary[0][9]
            or '新股權利證書終止上市日期' not in ordinary[0][9]):
        raise ValueError('Certificate and ordinary conversion are not the same issue')
    return previous, deepcopy(doc['replacement']), dict(sources.used)


def inventory_audit(case, terms, document_sha):
    """Rebuild ordinary unit balances and a distinct non-traded rights ledger.

    No certificate price tape is borrowed. Rights remain non-spendable and are
    valued by the inherited ordinary-close proxy until their one conversion.
    """
    if not case.get('completed'): raise ValueError('A partial case has no certificate inventory proof')
    account = case['account']
    issues = [r for r in account['corporate_actions'] if r.get('stock_id') == '5876'
              and r.get('date') == '2026-07-28' and r.get('kind') == 'stock_dividend']
    if not issues: return dict(schema=POLICY, document_sha256=document_sha, events=[])
    if len(issues) != 1: raise ValueError('Certificate entitlement duplicated')
    issue = issues[0]
    if (terms.get('shares_per_share') != .01 or terms.get('pay_date') != CONVERSION
            or terms.get('certificate_trading_modeled') is not False
            or type(issue.get('entitled_qty')) is not int or issue['entitled_qty'] <= 0):
        raise ValueError('Certificate entitlement basis differs')
    restriction = dict(certificate_delivery_date=CERTIFICATE, ordinary_share_available_date=CONVERSION,
        valuation_basis='ordinary_share_close_proxy', certificate_trading_modeled=False)
    whole = int(Decimal(issue['entitled_qty'])*Decimal('.01'))
    if (issue.get('certificate_restriction') != restriction or issue.get('pay_date') != CONVERSION
            or issue.get('whole_new_shares') != whole or issue.get('shares_per_share') != .01):
        raise ValueError('Certificate units were credited as early ordinary shares')
    deliveries = [r for r in account['corporate_actions'] if r.get('action_id') == issue['action_id']
                  and r.get('kind') == 'share_delivery']
    if ((whole > 0 and len(deliveries) != 1) or len(deliveries) > 1
            or any(r['date'] != CONVERSION or r['qty'] != whole for r in deliveries)):
        raise ValueError('Certificate conversion must deliver the same units exactly once')
    if any(r.get('action_id') == issue['action_id']+'-fractional-cash' for r in account['cash_ledger']):
        raise ValueError('Undated certificate fractional cash was spent')
    days = [r['date'] for r in account['daily'] if issue['date'] <= r['date'] <= CONVERSION]
    if not days or days[0] != issue['date'] or days[-1] != CONVERSION or CERTIFICATE not in days:
        raise ValueError('Certificate calendar is incomplete')
    unsupported = [r for r in account['corporate_actions'] if r.get('stock_id') == '5876'
        and issue['date'] <= r.get('date','') <= CONVERSION
        and r.get('kind') in ('stock_dividend','share_delivery','split','capital_reduction')
        and r.get('action_id') != issue['action_id']]
    if unsupported: raise ValueError('Overlapping unit changes require separate certificate review')
    ordinary, journal = issue['entitled_qty'], []
    for day in days:
        if day == CONVERSION: ordinary += whole
        for trade in account['trades']:
            if trade['date'] == day and trade['stock_id'] == '5876':
                ordinary += trade['qty'] if trade['side'] == 'buy' else -trade['qty']
                if ordinary < 0: raise ValueError('Certificate units sold before ordinary conversion')
        actual = sum(r['qty'] for r in account['holdings'] if r['date'] == day and r['stock_id'] == '5876')
        if ordinary != actual: raise ValueError('Certificate units contaminate ordinary holdings')
        state = 'pending_entitlement' if day < CERTIFICATE else 'nontraded_certificate_inventory' if day < CONVERSION else 'converted_to_ordinary'
        journal.append(dict(date=day, state=state, restricted_units=whole if day < CONVERSION else 0,
            converted_units=whole if day == CONVERSION else 0, ordinary_closing_units=ordinary,
            certificate_trades=0, certificate_cash_available=0))
    return dict(schema=POLICY, document_sha256=document_sha, events=[dict(stock_id='5876',
        action_id=issue['action_id'], certificate_delivery_date=CERTIFICATE,
        ordinary_conversion_date=CONVERSION, whole_units=whole,
        valuation_basis='ordinary_share_close_proxy', certificate_trading_modeled=False, daily=journal)])


def verify_inventory_proof(case, root=ROOT):
    _, terms, _ = load_certificates(root, Path(root)/DOCUMENT)
    expected = inventory_audit(case, terms, digest(Path(root)/DOCUMENT))
    if not expected['events'] or case.get('certificate_inventory_audit') != expected:
        raise ValueError('Independent certificate unit journal is absent or differs')
    return expected


def install_certificates(runner, provider, document, root):
    previous, row, refs = load_certificates(root, document)
    original = runner.merge_overrides
    def merge(base, additions):
        # The frozen runner loads its range terms after provider completions.
        # Its exact, bound old event must not overwrite the corrected event.
        if base.get(KEY) == row and additions.get(KEY) == previous:
            additions = dict(additions, **{KEY:row})
        if additions.get(KEY) == row and KEY in base and base[KEY] != row:
            if base[KEY] != previous: raise ValueError('Inherited certificate terms changed')
            base = {k:v for k,v in base.items() if k != KEY}
        return original(base, additions)
    if KEY in provider.corporate_overrides and provider.corporate_overrides[KEY] not in (previous, row):
        raise ValueError('Provider certificate terms conflict')
    provider.corporate_overrides = dict(provider.corporate_overrides, **{KEY:row})
    runner.merge_overrides = merge
    writer = runner.write
    def write(path, value):
        if isinstance(value, dict) and value.get('completed') is True and 'account' in value:
            value = dict(value, certificate_inventory_audit=inventory_audit(value, row, digest(document)))
        return writer(path, value)
    runner.write = write
    return refs
