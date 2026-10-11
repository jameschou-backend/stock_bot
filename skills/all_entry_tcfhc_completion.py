"""5880 settlement with primary listing evidence and explicitly secondary terms."""
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
DOCUMENT = 'docs/all_entry_5880_certificate_20261012.json'
KEY = '5880-2026-08-11'
CERTIFICATE, CONVERSION = '2026-09-11', '2026-09-24'
POLICY = 'all_entry_5880_nontraded_certificate_inventory_v1'
NOTE = '5880／2026 配股：證書與普通股上市日已核對 TWSE 原始公告；配股率與畸零款條款採公司公告轉載交叉核對，公司議事錄原始 PDF 下載逾時、未留存且未完成視覺核對。證書獨立記錄且不交易；畸零款淨額與支付日未證實，不列可用現金。'
URLS = {
    'certificate': 'https://www.moneydj.com/KMDJ/news/newsviewer.aspx?a=258f8f59-19c4-4f1a-8e46-92a7cbf1dea7',
    'conversion': 'https://www.moneydj.com/kmdj/news/newsviewer.aspx?a=344ac8c7-07d5-4577-bc29-6de7795b4ce4',
    'exdate': 'https://www.moneydj.com/kmdj/news/newsviewer.aspx?a=c990177c-d484-41a4-b6c6-389ae0309185&c=MB06',
}
FACTS = {
    'certificate': ['普通股股東每仟股無償配發25股', '每股面額新臺幣10元',
        '發行新股3億9,202萬3,423股', '按面值折發現金(計算至元為止,元以下捨去)',
        '新股之權利義務與原有普通股相同'],
    'conversion': ['每股配發0.25元', '增資新股上市暨新股權利證書終止上市日期:115年9月24日',
        '於115年9月11日以新股權利證書上市買賣'],
    'exdate': ['現金股利每股配發0.80元', '股票股利每股配發0.25元', '除權(息)交易日:115/08/11'],
}
EXPECTED = dict(shares_per_share=.025, pay_date=CONVERSION, ordinary_share_available_date=CONVERSION,
    listing_announcement_date='2026-09-22', certificate_delivery_date=CERTIFICATE,
    certificate_listing_date=CERTIFICATE, ordinary_conversion_date=CONVERSION,
    ordinary_conversion_creates_new_entitlement=False, certificate_trading_modeled=False,
    same_code_fungible_trading_verified=False, valuation_basis='ordinary_share_close_proxy',
    fractional_cash_per_share=10, fractional_cash_rounding='floor_ntd', fractional_cash_pay_date=None,
    fractional_cash_payment_date_verified=False, fractional_cash_net_amount_verified=False,
    fractional_cash_available_for_trading=False,
    fractional_cash_treatment='gross_undated_receivable_upper_bound_net_unknown',
    use_scope='account_settlement_only_not_selection')


def load_completion(root, document):
    root, document = Path(root).resolve(), Path(document).resolve(); sources = Sources(root)
    doc = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    if (doc.get('schema') != POLICY or doc.get('key') != KEY or doc.get('evidence_note') != NOTE
            or doc.get('strategy_parameters_changed') is not False or doc.get('live_qualified') is not False
            or set(doc.get('evidence', {})) != set(URLS)):
        raise ValueError('TCFHC evidence scope differs')
    row, refs = doc['override'], doc['source_sha256']
    if (set(row) != set(EXPECTED) | {'source', 'evidence_files'}
            or any(row.get(k) != v for k,v in EXPECTED.items()) or set(row['evidence_files']) != set(refs)):
        raise ValueError('TCFHC settlement or fractional cash differs')
    for name,sha in refs.items(): sources.bind(name,sha)
    for kind,item in doc['evidence'].items():
        if any(item[k] not in refs for k in ('path','receipt_path')):
            raise ValueError('TCFHC source escapes inventory')
        meta = json.loads((root/item['receipt_path']).read_text())
        if (meta.get('url') != URLS[kind] or meta.get('final_url') != meta['url']
                or meta.get('http_status') != 200 or meta.get('automatic_redirects_disabled') is not True
                or meta.get('sha256') != refs[item['path']]
                or item.get('source_role') != 'secondary_republication_not_primary_response'):
            raise ValueError('TCFHC secondary receipt or source role differs')
        text = BeautifulSoup((root/item['path']).read_bytes(), 'html.parser').get_text()
        if any(normalized(x) not in normalized(text) for x in FACTS[kind]):
            raise ValueError('TCFHC source facts differ: '+kind)
    failed = doc['issuer_pdf_failure']
    if failed not in refs: raise ValueError('TCFHC PDF failure is unbound')
    failure = json.loads((root/failed).read_text())
    if (failure.get('error') != 'ReadTimeout' or failure.get('original_pdf_bytes_retained') is not False
            or failure.get('retries') != 0):
        raise ValueError('TCFHC unavailable PDF cannot become verified evidence')
    exchange = doc['exchange']
    if any(exchange[k] not in refs for k in ('path','receipt_path')):
        raise ValueError('TCFHC exchange evidence escapes inventory')
    meta = json.loads((root/exchange['receipt_path']).read_text())
    if (meta.get('url') != 'https://www.twse.com.tw/rwd/zh/announcement/LT185'
            or meta.get('http_status') != 200 or meta.get('accepted') is not True
            or meta.get('security_denied') is not False or meta.get('automatic_redirects_disabled') is not True
            or meta.get('redirect_statuses') != [] or meta.get('params',{}).get('stockNo') != '5880'
            or meta.get('raw_sha256') != refs[exchange['path']]
            or exchange.get('source_role') != 'primary_exchange_response'):
        raise ValueError('TCFHC exchange receipt differs')
    data = json.loads((root/exchange['path']).read_text()); rows = data.get('data', [])
    ordinary = [r for r in rows if r[0] == '5880' and r[2] == '(股票上市)新股權利證書終止上市，增資股股票上市']
    certificates = [r for r in rows if r[0] == '5880' and r[2] == '新股權利證書上市']
    if (data.get('stat') != 'OK' or len(rows) != 2 or len(ordinary) != 1 or len(certificates) != 1
            or ordinary[0][5:9] != ['115年09月22日','115年09月24日','392,023,423','16,072,960,337']
            or certificates[0][5:9] != ['115年09月08日','115年09月11日','392,023,423','16,072,960,337']
            or any('含員工酬勞轉增資0股' not in normalized(r[9]) or '與原已發行之股份相同' not in r[9] for r in rows)
            or '新股權利證書終止上市日期' not in ordinary[0][9]):
        raise ValueError('TCFHC certificate and ordinary dates or units differ')
    feed = doc['existing_dividend_feed']; events = doc['official_ex_events']
    if feed not in refs or events not in refs: raise ValueError('TCFHC dividend or ex-event feed is unbound')
    df = pd.read_parquet(root/feed)
    selected = df.loc[df.stock_id.eq('5880') & df.StockExDividendTradingDate.eq('2026-08-11')]
    if len(selected) != 1: raise ValueError('TCFHC ex-date differs')
    event = selected.iloc[0]
    if (Decimal(str(event.StockEarningsDistribution)) != Decimal('.25') or event.StockStatutorySurplus != 0
            or event.ParticipateDistributionOfTotalShares != 15680936914
            or event.CashEarningsDistribution != .8 or event.CashDividendPaymentDate != '2026-09-11'
            or abs(Decimal('15680936914')*Decimal('.025')-392023423) >= 1
            or 15680936914+392023423 != 16072960337):
        raise ValueError('TCFHC final distribution or employee allocation differs')
    df = pd.read_parquet(root/events)
    selected = df.loc[df.stock_id.eq('5880') & df.event_date.astype(str).eq('2026-08-11') & df.source.eq('ex_rights')]
    if (len(selected) != 1 or selected.iloc[0].prev_close != 26.1 or selected.iloc[0].ref_price != 24.68
            or round((26.1-.8)/1.025,2) != 24.68):
        raise ValueError('TCFHC official ex-right reference differs')
    return deepcopy(row), dict(sources.used)


def settlement_notes(case):
    from skills.all_entry_poya_completion import settlement_notes as prior_notes
    notes = prior_notes(case)
    if any((r.get('stock_id'),r.get('date'),r.get('kind')) == ('5880','2026-08-11','stock_dividend')
           for r in case.get('account',{}).get('corporate_actions',[])):
        notes.append(NOTE)
    return notes


def install_completion(runner, provider, document, root):
    row, refs = load_completion(root,document)
    if KEY in provider.corporate_overrides and provider.corporate_overrides[KEY] != row:
        raise ValueError('TCFHC completion rewrites existing settlement')
    provider.corporate_overrides = dict(provider.corporate_overrides, **{KEY:row})
    writer = runner.write
    def write(path,value):
        if isinstance(value,dict) and value.get('completed') is True and 'account' in value:
            value = dict(value, settlement_evidence_notes=settlement_notes(value),
                tcfhc_certificate_inventory_audit=inventory_audit(value,row,digest(document)))
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
    issues = [r for r in account['corporate_actions'] if r.get('stock_id') == '5880'
              and r.get('date') == '2026-08-11' and r.get('kind') == 'stock_dividend']
    if not issues: return dict(schema=POLICY, document_sha256=document_sha, events=[])
    if len(issues) != 1: raise ValueError('Certificate entitlement duplicated')
    issue = issues[0]
    if (terms.get('shares_per_share') != .025 or terms.get('pay_date') != CONVERSION
            or terms.get('certificate_trading_modeled') is not False
            or type(issue.get('entitled_qty')) is not int or issue['entitled_qty'] <= 0):
        raise ValueError('Certificate entitlement basis differs')
    restriction = dict(certificate_delivery_date=CERTIFICATE, ordinary_share_available_date=CONVERSION,
        valuation_basis='ordinary_share_close_proxy', certificate_trading_modeled=False)
    whole = int(Decimal(issue['entitled_qty'])*Decimal('.025'))
    if (issue.get('certificate_restriction') != restriction or issue.get('pay_date') != CONVERSION
            or issue.get('whole_new_shares') != whole or issue.get('shares_per_share') != .025):
        raise ValueError('Certificate units were credited as early ordinary shares')
    deliveries = [r for r in account['corporate_actions'] if r.get('action_id') == issue['action_id']
                  and r.get('kind') == 'share_delivery']
    if ((whole > 0 and len(deliveries) != 1) or len(deliveries) > 1
            or any(r['date'] != CONVERSION or r['qty'] != whole for r in deliveries)):
        raise ValueError('Certificate conversion must deliver the same units exactly once')
    if any(r.get('action_id') == issue['action_id']+'-fractional-cash' for r in account['cash_ledger']):
        raise ValueError('Undated certificate fractional cash was spent')
    fraction = Decimal(issue['entitled_qty'])*Decimal('.025')-whole
    gross = int(fraction*10)
    expected_fraction = dict(face_value=10.,fraction=float(fraction),gross_cash_amount=float(gross),
        payment_date=None,payment_date_verified=False,processing_or_remittance_fees_verified=False,rounding='floor_ntd')
    if (issue.get('fractional_right') != float(fraction)
            or issue.get('fractional_settlement') != expected_fraction):
        raise ValueError('TCFHC fractional entitlement differs')
    claims = [r for r in account['corporate_actions'] if r.get('action_id') == issue['action_id']+'-fractional-cash']
    if ((gross > 0 and len(claims) != 1) or len(claims) > 1
            or any(r.get('pay_date') is not None or r.get('entitlement_value') != gross
                or r.get('fractional_settlement') != expected_fraction for r in claims)):
        raise ValueError('TCFHC undated fractional receivable differs')
    days = [r['date'] for r in account['daily'] if issue['date'] <= r['date'] <= CONVERSION]
    if not days or days[0] != issue['date'] or days[-1] != CONVERSION or CERTIFICATE not in days:
        raise ValueError('Certificate calendar is incomplete')
    unsupported = [r for r in account['corporate_actions'] if r.get('stock_id') == '5880'
        and issue['date'] <= r.get('date','') <= CONVERSION
        and r.get('kind') in ('stock_dividend','share_delivery','split','capital_reduction')
        and r.get('action_id') != issue['action_id']]
    if unsupported: raise ValueError('Overlapping unit changes require separate certificate review')
    ordinary, journal = issue['entitled_qty'], []
    for day in days:
        if day == CONVERSION: ordinary += whole
        for trade in account['trades']:
            if trade['date'] == day and trade['stock_id'] == '5880':
                ordinary += trade['qty'] if trade['side'] == 'buy' else -trade['qty']
                if ordinary < 0: raise ValueError('Certificate units sold before ordinary conversion')
        actual = sum(r['qty'] for r in account['holdings'] if r['date'] == day and r['stock_id'] == '5880')
        if ordinary != actual: raise ValueError('Certificate units contaminate ordinary holdings')
        state = 'pending_entitlement' if day < CERTIFICATE else 'nontraded_certificate_inventory' if day < CONVERSION else 'converted_to_ordinary'
        journal.append(dict(date=day, state=state, restricted_units=whole if day < CONVERSION else 0,
            converted_units=whole if day == CONVERSION else 0, ordinary_closing_units=ordinary,
            certificate_trades=0, certificate_cash_available=0))
    return dict(schema=POLICY, document_sha256=document_sha, events=[dict(stock_id='5880',
        action_id=issue['action_id'], certificate_delivery_date=CERTIFICATE,
        ordinary_conversion_date=CONVERSION, whole_units=whole,
        valuation_basis='ordinary_share_close_proxy', certificate_trading_modeled=False, daily=journal)])



def verify_inventory_proof(case, root=ROOT):
    row, _ = load_completion(root, Path(root)/DOCUMENT)
    expected = inventory_audit(case, row, digest(Path(root)/DOCUMENT))
    if not expected['events'] or case.get('tcfhc_certificate_inventory_audit') != expected:
        raise ValueError('Independent TCFHC certificate unit journal is absent or differs')
    return expected
