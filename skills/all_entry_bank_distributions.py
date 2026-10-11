"""Evidence-only ordinary distributions; no changes to account or selection rules."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import unicodedata
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import pandas as pd

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest
from skills.all_entry_distribution_completion import roc_day

SCHEMA = 'all_entry_verified_bank_distributions_v1'
EXPECTED = {
    '2886-2024-08-08': dict(rate='.03', quantity=432040144, employee=0,
        total=14401338138, announcement='2024-09-10', pay='2024-09-13', raw_ex='2024-08-08',
        issuer='www.megaholdings.com.tw', calendar='www.megaholdings.com.tw',
        calendar_row=['112','1.50','113/09/13','0.30','-','1.80','113/09/13','113/06/21','113/08/08','113/08/12','113/08/16'],
        required=['每仟股無償配發30股','每股面額10元','4億3,204萬144股',
                  '畸零股改發現金(計算至元,元以下捨去)','本案照原議案表決通過']),
    '2884-2024-07-26': dict(rate='.020033197', quantity=331800000, employee=18000000,
        total=15664000000, announcement='2024-08-26', pay='2024-08-29', raw_ex='2024-07-25',
        issuer='www.esunbank.com', calendar='www.esunfhc.com',
        calendar_row=['2024','2024.08.02','1.2','2024.08.29','2024.08.02','0.20033197','2024.08.29'],
        required=['股東股票股利3,138,000,000元','發行313,800,000股','發行新股18,000,000股',
                  '發行新股331,800,000股,每股面額10元','按面額折發現金(至元為止)',
                  '主席宣布本案表決通過']),
}


def normalized(value):
    return ''.join(unicodedata.normalize('NFKC', value).split())


def load_distributions(root, document):
    root, document = Path(root).resolve(), Path(document).resolve()
    sources = Sources(root)
    doc = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    if (doc.get('schema') != SCHEMA or doc.get('strategy_parameters_changed') is not False
            or doc.get('live_qualified') is not False or set(doc.get('events',{})) != set(EXPECTED)):
        raise ValueError('Verified bank distribution scope differs')
    overrides = {}
    for key, spec in EXPECTED.items():
        sid = key[:4]; event = doc['events'][key]; row = event['override']; refs = event['source_sha256']
        for name, sha in refs.items(): sources.bind(name, sha)
        required = dict(pay_date=spec['pay'], ordinary_share_available_date=spec['pay'],
            listing_announcement_date=spec['announcement'], fractional_cash_per_share=10,
            fractional_cash_rounding='floor_ntd', fractional_cash_pay_date=None,
            fractional_cash_payment_date_verified=False, fractional_cash_net_amount_verified=False,
            fractional_cash_available_for_trading=False,
            fractional_cash_treatment='gross_undated_receivable_upper_bound_net_unknown',
            use_scope='account_settlement_only_not_selection')
        if (any(row.get(k) != v for k,v in required.items())
                or Decimal(str(row.get('shares_per_share'))) != Decimal(spec['rate'])
                or set(row['evidence_files']) != set(refs) or 'certificate_delivery_date' in row):
            raise ValueError('Verified bank settlement terms differ')
        for kind, host in [('issuer',spec['issuer']),('calendar',spec['calendar']),('exchange','www.twse.com.tw')]:
            item = event[kind]
            if any(item[k] not in refs for k in ('path','receipt_path')):
                raise ValueError('Bank evidence escapes source inventory')
            receipt = json.loads((root/item['receipt_path']).read_text()); url = urlparse(receipt['url'])
            if (url.scheme != 'https' or url.hostname != host or receipt.get('http_status') != 200
                    or receipt.get('automatic_redirects_disabled') is not True
                    or receipt.get('sha256',receipt.get('raw_sha256')) != refs[item['path']]):
                raise ValueError('Bank distribution primary response identity differs')
            if kind != 'exchange' and receipt.get('final_url') != receipt['url']:
                raise ValueError('Bank issuer source redirected')
            if kind == 'exchange' and (receipt.get('accepted') is not True
                    or receipt.get('security_denied') is not False or receipt.get('redirect_statuses') != []
                    or url.path != '/rwd/zh/announcement/LT185'
                    or receipt.get('params',{}).get('stockNo') != sid):
                raise ValueError('Bank exchange response is not accepted LT185')
        issuer = event['issuer']
        if (issuer['text_path'] not in refs or issuer.get('extraction_command') != 'pdftotext -layout'
                or not (root/issuer['path']).read_bytes().startswith(b'%PDF')):
            raise ValueError('Bank adopted minutes are not bound')
        text = normalized((root/issuer['text_path']).read_text())
        if any(normalized(x) not in text for x in spec['required']):
            raise ValueError('Bank adopted minutes do not verify entitlement or fractions')
        soup = BeautifulSoup((root/event['calendar']['path']).read_text(),'html.parser')
        calendar_rows = [[x.get_text(' ',strip=True) for x in tr.select('th,td')] for tr in soup.select('tr')]
        if calendar_rows.count(spec['calendar_row']) != 1:
            raise ValueError('Bank final rate and delivery calendar differ')
        data = json.loads((root/event['exchange']['path']).read_text())
        rows = [x for x in data.get('data',[]) if x[0] == sid]
        if (data.get('stat') != 'OK' or len(rows) != 1 or rows[0][2] != '(股票上市)增資新股上市'
                or roc_day(rows[0][5]) != spec['announcement'] or roc_day(rows[0][6]) != spec['pay']
                or int(rows[0][7].replace(',','')) != spec['quantity']
                or f"含員工酬勞轉增資{spec['employee']:,}股" not in normalized(rows[0][9])
                or '與原已發行之股份相同' not in rows[0][9]):
            raise ValueError('Bank ordinary listing or employee share exclusion differs')
        feed = event['existing_dividend_feed']
        if feed not in refs: raise ValueError('Bank ex-date source is not bound')
        frame = pd.read_parquet(root/feed)
        selected = frame.loc[frame.stock_id.eq(sid) & frame.StockExDividendTradingDate.eq(spec['raw_ex'])]
        if (len(selected) != 1
                or Decimal(str(selected.iloc[0].StockEarningsDistribution))/10 != Decimal(spec['rate'])
                or float(selected.iloc[0].StockStatutorySurplus) != 0
                or float(selected.iloc[0].ParticipateDistributionOfTotalShares) != spec['total']):
            raise ValueError('Bank final distribution feed differs from primary evidence')
        # Shareholder units exclude employee compensation. Published rates are
        # rounded and aggregate issuance can differ by a few indivisible units.
        allocated = Decimal(spec['rate'])*spec['total']
        if abs(allocated-(spec['quantity']-spec['employee'])) > Decimal('10'):
            raise ValueError('Bank shareholder units disagree with final published rate')
        overrides[key] = deepcopy(row)
    return overrides, dict(sources.used)


def install_distributions(provider, document, root):
    overrides, refs = load_distributions(root, document)
    for key, row in overrides.items():
        if key in provider.corporate_overrides and provider.corporate_overrides[key] != row:
            raise ValueError('Verified bank distribution would rewrite an existing event')
    provider.corporate_overrides = dict(provider.corporate_overrides, **overrides)
    return refs
