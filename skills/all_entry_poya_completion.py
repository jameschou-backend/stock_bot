"""5904 settlement evidence and a separate non-traded certificate inventory."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

from bs4 import BeautifulSoup
import pandas as pd

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest
from skills.all_entry_bank_distributions import normalized

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = 'docs/all_entry_5904_certificate_20261012.json'
KEY = '5904-2025-07-15'
CERTIFICATE, CONVERSION = '2025-08-15', '2025-09-26'
POLICY = 'all_entry_5904_nontraded_certificate_inventory_v1'
NOTE = '5904／2025 配股：公司議事錄原始 PDF 已核對；8/15 證書交付及 9/26 普通股轉換日期依公司與櫃買公告的媒體轉載交叉核對，尚缺公告原站回應。證書獨立記錄且不交易；帳簿劃撥畸零款依議事錄抵充費用，淨額為零。'
URLS = {
    'minutes': 'https://www.poya.com.tw/investor_template-detail/13/476/',
    'calendar': 'https://www.poya.com.tw/investor/50/8/',
    'conversion': 'https://www.moneydj.com/kmdj/news/newsviewer.aspx?a=89a32fe6-efd4-4fe5-9697-71c60f27f6d5',
    'listing': 'https://news.cnyes.com/news/id/6160591',
}
FACTS = {
    'minutes': ['1,049,362股,每股面額10元', '每仟股無償配發10股',
        '未滿一股之畸零股款將作為帳簿劃撥之費用', '83,621,946', '97.23%', '本案照原案表決通過'],
    'conversion': ['於114年8月15日起以新股權利證書上櫃買賣', '於114年9月26日起上櫃買賣',
        '普通股104,936,216股', '發行新股1,049,362股', '發行普通股458,248股',
        '普通股106,443,826股', '原新股權利證書轉換為普通股股份'],
    'listing': ['上櫃股票代號:5904', '普通股1,507,610股', '含員工酬勞轉增資458,248股',
        '上櫃股票開始買賣日期:民國114年09月26日', '新股權利證書終止上市日期',
        '上櫃股票權利義務:與原已發行之股份相同', '普通股股數餘額為106,443,826股'],
}
EXPECTED = dict(shares_per_share=.01, pay_date=CONVERSION, ordinary_share_available_date=CONVERSION,
    certificate_delivery_date=CERTIFICATE, certificate_listing_date=CERTIFICATE,
    ordinary_conversion_date=CONVERSION, ordinary_conversion_creates_new_entitlement=False,
    certificate_trading_modeled=False, same_code_fungible_trading_verified=False,
    valuation_basis='ordinary_share_close_proxy',
    fractional_cash_per_share=0, fractional_net_zero_verified=True,
    fractional_cash_rounding='floor_ntd', fractional_cash_pay_date=None,
    fractional_cash_payment_date_verified=False, fractional_cash_net_amount_verified=True,
    fractional_cash_available_for_trading=False, fractional_cash_treatment='net_zero_book_entry_fees_verified',
    use_scope='account_settlement_only_not_selection')


def load_completion(root, document):
    root, document = Path(root).resolve(), Path(document).resolve(); sources = Sources(root)
    doc = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    if (doc.get('schema') != POLICY or doc.get('key') != KEY or doc.get('evidence_note') != NOTE
            or doc.get('strategy_parameters_changed') is not False or doc.get('live_qualified') is not False
            or set(doc.get('evidence', {})) != set(URLS)):
        raise ValueError('Poya evidence scope differs')
    row, refs = doc['override'], doc['source_sha256']
    if (set(row) != set(EXPECTED) | {'source', 'evidence_files'}
            or any(row.get(k) != v for k,v in EXPECTED.items()) or set(row['evidence_files']) != set(refs)):
        raise ValueError('Poya settlement or fractional cash differs')
    for name,sha in refs.items(): sources.bind(name,sha)
    for kind,item in doc['evidence'].items():
        if any(item[k] not in refs for k in ('path','receipt_path')):
            raise ValueError('Poya source escapes inventory')
        path = root/item['path']; meta = json.loads((root/item['receipt_path']).read_text())
        if (meta.get('url') != URLS[kind] or meta.get('final_url') != meta['url']
                or meta.get('http_status') != 200 or meta.get('automatic_redirects_disabled') is not True
                or meta.get('sha256') != refs[item['path']]):
            raise ValueError('Poya source receipt differs')
        if kind == 'minutes':
            if (not path.read_bytes().startswith(b'%PDF') or item.get('text_path') not in refs
                    or item.get('reviewed_pdf_pages') != [4,5] or item.get('source_role') != 'primary_issuer_pdf'):
                raise ValueError('Poya original adopted minutes are unverified')
            text = (root/item['text_path']).read_text()
        else:
            role = 'primary_issuer_calendar' if kind == 'calendar' else 'secondary_republication_not_primary_response'
            if item.get('source_role') != role:
                raise ValueError('Poya secondary evidence cannot be promoted to primary')
            soup = BeautifulSoup(path.read_bytes(), 'html.parser'); text = soup.get_text()
            if kind == 'calendar':
                rows = [[x.get_text(' ',strip=True) for x in tr.select('th,td')] for tr in soup.select('tr')]
                if rows.count(['2024','23','0.1','23.1','2025/07/15','2025/07/15']) != 1:
                    raise ValueError('Poya final dividend calendar differs')
        if any(normalized(x) not in normalized(text) for x in FACTS.get(kind, [])):
            raise ValueError('Poya source facts differ: '+kind)
    feed = doc['existing_dividend_feed']
    if feed not in refs: raise ValueError('Poya dividend feed is unbound')
    df = pd.read_parquet(root/feed)
    selected = df.loc[df.stock_id.eq('5904') & df.StockExDividendTradingDate.eq('2025-07-15')]
    if len(selected) != 1: raise ValueError('Poya ex-date differs')
    event = selected.iloc[0]
    if (Decimal(str(event.StockEarningsDistribution)) != Decimal('.1') or event.StockStatutorySurplus != 0
            or event.ParticipateDistributionOfTotalShares != 104936216
            or int(Decimal('104936216')*Decimal('.01')) != 1049362):
        raise ValueError('Poya shareholder distribution or employee allocation differs')
    return deepcopy(row), dict(sources.used)


def settlement_notes(case):
    from skills.all_entry_context_distributions import settlement_notes as prior_notes
    notes = prior_notes(case)
    if any((r.get('stock_id'),r.get('date'),r.get('kind')) == ('5904','2025-07-15','stock_dividend')
           for r in case.get('account',{}).get('corporate_actions',[])):
        notes.append(NOTE)
    return notes


def install_completion(runner, provider, document, root):
    row, refs = load_completion(root,document)
    if KEY in provider.corporate_overrides and provider.corporate_overrides[KEY] != row:
        raise ValueError('Poya completion rewrites existing settlement')
    provider.corporate_overrides = dict(provider.corporate_overrides, **{KEY:row})
    writer = runner.write
    def write(path,value):
        if isinstance(value,dict) and value.get('completed') is True and 'account' in value:
            value = dict(value, settlement_evidence_notes=settlement_notes(value),
                poya_certificate_inventory_audit=inventory_audit(value,row,digest(document)))
        return writer(path,value)
    runner.write = write
    return refs


def inventory_audit(case, terms, document_sha):
    """Rebuild ordinary unit balances and a distinct non-traded rights ledger.

    No certificate price tape is borrowed. Rights remain non-spendable and are
    valued by the inherited ordinary-close proxy until their one conversion.
    """
    if not case.get('completed'): raise ValueError('A partial case has no certificate inventory proof')
    account = case['account']
    issues = [r for r in account['corporate_actions'] if r.get('stock_id') == '5904'
              and r.get('date') == '2025-07-15' and r.get('kind') == 'stock_dividend']
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
        raise ValueError('Net-zero certificate fractional cash was spent')
    days = [r['date'] for r in account['daily'] if issue['date'] <= r['date'] <= CONVERSION]
    if not days or days[0] != issue['date'] or days[-1] != CONVERSION or CERTIFICATE not in days:
        raise ValueError('Certificate calendar is incomplete')
    unsupported = [r for r in account['corporate_actions'] if r.get('stock_id') == '5904'
        and issue['date'] <= r.get('date','') <= CONVERSION
        and r.get('kind') in ('stock_dividend','share_delivery','split','capital_reduction')
        and r.get('action_id') != issue['action_id']]
    if unsupported: raise ValueError('Overlapping unit changes require separate certificate review')
    ordinary, journal = issue['entitled_qty'], []
    for day in days:
        if day == CONVERSION: ordinary += whole
        for trade in account['trades']:
            if trade['date'] == day and trade['stock_id'] == '5904':
                ordinary += trade['qty'] if trade['side'] == 'buy' else -trade['qty']
                if ordinary < 0: raise ValueError('Certificate units sold before ordinary conversion')
        actual = sum(r['qty'] for r in account['holdings'] if r['date'] == day and r['stock_id'] == '5904')
        if ordinary != actual: raise ValueError('Certificate units contaminate ordinary holdings')
        state = 'pending_entitlement' if day < CERTIFICATE else 'nontraded_certificate_inventory' if day < CONVERSION else 'converted_to_ordinary'
        journal.append(dict(date=day, state=state, restricted_units=whole if day < CONVERSION else 0,
            converted_units=whole if day == CONVERSION else 0, ordinary_closing_units=ordinary,
            certificate_trades=0, certificate_cash_available=0))
    return dict(schema=POLICY, document_sha256=document_sha, events=[dict(stock_id='5904',
        action_id=issue['action_id'], certificate_delivery_date=CERTIFICATE,
        ordinary_conversion_date=CONVERSION, whole_units=whole,
        valuation_basis='ordinary_share_close_proxy', certificate_trading_modeled=False, daily=journal)])



def verify_inventory_proof(case, root=ROOT):
    row, _ = load_completion(root, Path(root)/DOCUMENT)
    expected = inventory_audit(case, row, digest(Path(root)/DOCUMENT))
    if not expected['events'] or case.get('poya_certificate_inventory_audit') != expected:
        raise ValueError('Independent Poya certificate unit journal is absent or differs')
    return expected
