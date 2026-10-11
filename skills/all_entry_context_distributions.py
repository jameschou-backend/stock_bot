"""Bound settlement evidence for the two context accounts; no signal changes."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
from urllib.parse import urlparse, unquote

from bs4 import BeautifulSoup
import pandas as pd

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest
from skills.all_entry_bank_distributions import normalized
from skills.all_entry_distribution_completion import roc_day

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = 'docs/all_entry_context_distributions_20261011.json'
SCHEMA = 'all_entry_context_distributions_v1'
WEB_TEXT_SHA = 'e64cf08d9e9e6fbb9a9c773fdff38d30fa47244eaf8587302dc52b05d6ad432f'
NOTE = '2887／2024 配股：比例與畸零款條款依官方議事錄工具擷取文字，普通股上市日另有 TWSE 原始公告；原始 PDF 下載失敗、未留存且未完成 PDF 視覺核對。畸零款淨額與支付日未證實，不列可用現金。'
EXPECTED = {
    '2801-2026-08-05': dict(rate='.025', feed='.25', total=11766046253, quantity=294151156,
        pay='2026-09-08', announced='2026-09-02', issuer='www.bankchb.com',
        required=['每壹仟股無償配發新股25股','每股面額10元',
            '按面值折發現金(計算至元,元以下捨去)','本案經表決照案通過']),
    '2887-2024-08-06': dict(rate='.04000000003', feed='.4000000003', total=12477061838, quantity=499082474,
        pay='2024-08-29', announced='2024-08-27', issuer='www.tsholdings.com.tw',
        required=['每壹仟股無償配發40.00股','每股面額10元','499,082,474股',
            '按面額折發現金(抵繳集保劃撥暨無實體登錄等費用)','元以下捨去','本案經表決照案通過']),
}


def load_distributions(root, document):
    root, document = Path(root).resolve(), Path(document).resolve()
    sources = Sources(root)
    doc = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    if (doc.get('schema') != SCHEMA or set(doc.get('events', {})) != set(EXPECTED)
            or doc.get('strategy_parameters_changed') is not False or doc.get('live_qualified') is not False):
        raise ValueError('Context settlement scope differs')
    overrides = {}
    for key, spec in EXPECTED.items():
        sid = key[:4]; event = doc['events'][key]; row = event['override']; refs = event['source_sha256']
        for name, sha in refs.items(): sources.bind(name, sha)
        required = dict(shares_per_share=float(spec['rate']), pay_date=spec['pay'],
            ordinary_share_available_date=spec['pay'], listing_announcement_date=spec['announced'],
            fractional_cash_per_share=10, fractional_cash_rounding='floor_ntd',
            fractional_cash_pay_date=None, fractional_cash_payment_date_verified=False,
            fractional_cash_net_amount_verified=False, fractional_cash_available_for_trading=False,
            fractional_cash_treatment='gross_undated_receivable_upper_bound_net_unknown',
            use_scope='account_settlement_only_not_selection')
        if any(row.get(k) != v for k,v in required.items()) or set(row['evidence_files']) != set(refs):
            raise ValueError('Context entitlement or cash treatment differs')
        if sid == '2801':
            restricted = dict(certificate_delivery_date=CERTIFICATE, certificate_listing_date=CERTIFICATE,
                certificate_trading_modeled=False, same_code_fungible_trading_verified=False,
                ordinary_conversion_date=CONVERSION, ordinary_conversion_creates_new_entitlement=False,
                valuation_basis='ordinary_share_close_proxy')
            if any(row.get(k) != v for k,v in restricted.items()):
                raise ValueError('Context certificate restriction differs')
        elif 'certificate_delivery_date' in row:
            raise ValueError('Ordinary distribution cannot become a certificate')
        issuer, exchange = event['issuer'], event['exchange']
        if any(item[k] not in refs for item in (issuer, exchange) for k in ('path','receipt_path')):
            raise ValueError('Context evidence escapes inventory')
        meta = json.loads((root/issuer['receipt_path']).read_text())
        if sid == '2801':
            url = urlparse(meta.get('url', ''))
            if (meta.get('http_status') != 200 or meta.get('automatic_redirects_disabled') is not True
                    or unquote(meta.get('final_url', '')) != unquote(meta['url']) or meta.get('sha256') != refs[issuer['path']]
                    or not (root/issuer['path']).read_bytes().startswith(b'%PDF')
                    or issuer.get('text_path') not in refs or issuer.get('reviewed_pdf_pages') != [9,10]):
                raise ValueError('Context issuer PDF identity differs')
            text = (root/issuer['text_path']).read_text()
            calendar = event['calendar']; receipt = json.loads((root/calendar['receipt_path']).read_text())
            if (calendar['path'] not in refs or calendar['receipt_path'] not in refs
                    or receipt.get('url') != 'https://www.bankchb.com/frontend/mashup.jsp?funcId=ede26b5415'
                    or receipt.get('url') != receipt.get('final_url') or receipt.get('http_status') != 200
                    or receipt.get('sha256') != refs[calendar['path']]):
                raise ValueError('Context final dividend calendar identity differs')
            soup = BeautifulSoup((root/calendar['path']).read_text(), 'html.parser')
            rows = [[x.get_text(' ',strip=True) for x in tr.select('th,td')] for tr in soup.select('tr')]
            if rows.count(['114','115-08-05','0.80','115-08-31','115-08-05','0.25','115-08-31']) != 1:
                raise ValueError('Context final dividend calendar differs')
        else:
            url = urlparse(meta.get('issuer_url', ''))
            expected = dict(schema='retained_web_tool_primary_observation_v1', tool='web.run',
                provider_reference='turn81view0', retained_tool_text_sha256=WEB_TEXT_SHA,
                original_pdf_bytes_retained=False, visual_pdf_verification=False,
                local_direct_download_failed=True, security_challenge_bypass=False)
            if (any(meta.get(k) != v for k,v in expected.items())
                    or digest(root/issuer['path']) != WEB_TEXT_SHA or event.get('evidence_note') != NOTE):
                raise ValueError('Context tool text cannot be promoted to raw PDF evidence')
            text = (root/issuer['path']).read_text()
        if (url.scheme != 'https' or url.hostname != spec['issuer']
                or any(normalized(x) not in normalized(text) for x in spec['required'])):
            raise ValueError('Context adopted entitlement or fractions differ')
        meta = json.loads((root/exchange['receipt_path']).read_text()); url = urlparse(meta['url'])
        if (url.scheme != 'https' or url.hostname != 'www.twse.com.tw' or url.path != '/rwd/zh/announcement/LT185'
                or meta.get('http_status') != 200 or meta.get('accepted') is not True
                or meta.get('security_denied') is not False or meta.get('automatic_redirects_disabled') is not True
                or meta.get('redirect_statuses') != [] or meta.get('params',{}).get('stockNo') != sid
                or meta.get('raw_sha256') != refs[exchange['path']]):
            raise ValueError('Context exchange receipt differs')
        data = json.loads((root/exchange['path']).read_text()); rows = data.get('data', [])
        kind = '(股票上市)新股權利證書終止上市，增資股股票上市' if sid == '2801' else '(股票上市)增資新股上市'
        ordinary = [r for r in rows if r[0] == sid and r[2] == kind]
        if (data.get('stat') != 'OK' or len(rows) != (2 if sid == '2801' else 1) or len(ordinary) != 1
                or roc_day(ordinary[0][5]) != spec['announced'] or roc_day(ordinary[0][6]) != spec['pay']
                or int(ordinary[0][7].replace(',','')) != spec['quantity']
                or '含員工酬勞轉增資0股' not in normalized(ordinary[0][9])
                or '與原已發行之股份相同' not in ordinary[0][9]):
            raise ValueError('Context ordinary issue dates or units differ')
        if sid == '2801':
            certificates = [r for r in rows if r[0] == sid and r[2] == '新股權利證書上市']
            if (len(certificates) != 1 or certificates[0][5:8] != ['115年08月24日','115年08月31日','294,151,156']
                    or '新股權利證書終止上市日期' not in ordinary[0][9]):
                raise ValueError('Context certificate and ordinary issue do not match')
        feed = event['existing_dividend_feed']
        if feed not in refs: raise ValueError('Context dividend feed is unbound')
        frame = pd.read_parquet(root/feed)
        selected = frame.loc[frame.stock_id.eq(sid) & frame.StockExDividendTradingDate.eq(key[5:])]
        if (len(selected) != 1 or Decimal(str(selected.iloc[0].StockEarningsDistribution)) != Decimal(spec['feed'])
                or float(selected.iloc[0].StockStatutorySurplus) != 0
                or float(selected.iloc[0].ParticipateDistributionOfTotalShares) != spec['total']
                or abs(Decimal(spec['rate'])*spec['total']-spec['quantity']) >= 1):
            raise ValueError('Context final rate or shareholder quantity differs')
        overrides[key] = deepcopy(row)
    return overrides, dict(sources.used)


def settlement_notes(case):
    from skills.all_entry_issuer_distributions import settlement_notes as prior_notes
    notes = prior_notes(case)
    if any((r.get('stock_id'),r.get('date'),r.get('kind')) == ('2887','2024-08-06','stock_dividend')
           for r in case.get('account',{}).get('corporate_actions',[])):
        notes.append(NOTE)
    return notes


def install_distributions(runner, provider, document, root):
    rows, refs = load_distributions(root, document)
    for key,row in rows.items():
        if key in provider.corporate_overrides and provider.corporate_overrides[key] != row:
            raise ValueError('Context distribution rewrites existing terms')
    provider.corporate_overrides = dict(provider.corporate_overrides, **rows)
    writer = runner.write
    def write(path, value):
        # Install this boundary before the older wrappers so their notes are
        # extended after they finish; all prior source limitations survive.
        if isinstance(value, dict) and value.get('completed') is True and 'account' in value:
            value = dict(value, settlement_evidence_notes=settlement_notes(value),
                context_certificate_inventory_audit=inventory_audit(value, rows[KEY], digest(document)))
        return writer(path, value)
    runner.write = write
    return refs


KEY = '2801-2026-08-05'
POLICY = 'all_entry_2801_nontraded_certificate_inventory_v1'
CERTIFICATE = '2026-08-31'
CONVERSION = '2026-09-08'


def inventory_audit(case, terms, document_sha):
    """Rebuild ordinary unit balances and a distinct non-traded rights ledger.

    No certificate price tape is borrowed. Rights remain non-spendable and are
    valued by the inherited ordinary-close proxy until their one conversion.
    """
    if not case.get('completed'): raise ValueError('A partial case has no certificate inventory proof')
    account = case['account']
    issues = [r for r in account['corporate_actions'] if r.get('stock_id') == '2801'
              and r.get('date') == '2026-08-05' and r.get('kind') == 'stock_dividend']
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
    days = [r['date'] for r in account['daily'] if issue['date'] <= r['date'] <= CONVERSION]
    if not days or days[0] != issue['date'] or days[-1] != CONVERSION or CERTIFICATE not in days:
        raise ValueError('Certificate calendar is incomplete')
    unsupported = [r for r in account['corporate_actions'] if r.get('stock_id') == '2801'
        and issue['date'] <= r.get('date','') <= CONVERSION
        and r.get('kind') in ('stock_dividend','share_delivery','split','capital_reduction')
        and r.get('action_id') != issue['action_id']]
    if unsupported: raise ValueError('Overlapping unit changes require separate certificate review')
    ordinary, journal = issue['entitled_qty'], []
    for day in days:
        if day == CONVERSION: ordinary += whole
        for trade in account['trades']:
            if trade['date'] == day and trade['stock_id'] == '2801':
                ordinary += trade['qty'] if trade['side'] == 'buy' else -trade['qty']
                if ordinary < 0: raise ValueError('Certificate units sold before ordinary conversion')
        actual = sum(r['qty'] for r in account['holdings'] if r['date'] == day and r['stock_id'] == '2801')
        if ordinary != actual: raise ValueError('Certificate units contaminate ordinary holdings')
        state = 'pending_entitlement' if day < CERTIFICATE else 'nontraded_certificate_inventory' if day < CONVERSION else 'converted_to_ordinary'
        journal.append(dict(date=day, state=state, restricted_units=whole if day < CONVERSION else 0,
            converted_units=whole if day == CONVERSION else 0, ordinary_closing_units=ordinary,
            certificate_trades=0, certificate_cash_available=0))
    return dict(schema=POLICY, document_sha256=document_sha, events=[dict(stock_id='2801',
        action_id=issue['action_id'], certificate_delivery_date=CERTIFICATE,
        ordinary_conversion_date=CONVERSION, whole_units=whole,
        valuation_basis='ordinary_share_close_proxy', certificate_trading_modeled=False, daily=journal)])



def verify_inventory_proof(case, root=ROOT):
    rows, _ = load_distributions(root, Path(root)/DOCUMENT)
    expected = inventory_audit(case, rows[KEY], digest(Path(root)/DOCUMENT))
    if not expected['events'] or case.get('context_certificate_inventory_audit') != expected:
        raise ValueError('Independent context certificate unit journal is absent or differs')
    return expected
