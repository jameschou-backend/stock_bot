"""Two ordinary distributions with explicit provenance for a text-only source."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import pandas as pd

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest
from skills.all_entry_bank_distributions import normalized
from skills.all_entry_distribution_completion import roc_day

SCHEMA = 'all_entry_issuer_distributions_v1'
WEB_TEXT_SHA = '5e2a3497c382004fbc35a7be24270112e747bc90db320cd72aefe9f2419bdc1f'
TEXT_NOTE = '1519／2024 配股：比例與畸零款條款依先前取得的官方議事錄工具擷取文字，普通股上市日另有 TWSE 原始公告；公司網站下載受防護限制，原始 PDF 未留存且未完成 PDF 視覺核對。畸零款淨額與支付日未證實，不列可用現金。'
EXPECTED = {
    '1519-2024-07-26': dict(quantity=26105855,total=261058548,raw_ex='2024-07-25',
        announced='2024-08-21',pay='2024-08-30',kind='retained_primary_pdf_tool_text',
        host='www.fortune.com.tw',required=['每仟股約配發100股','每股面額新台幣壹拾元',
            '按面額折發現金至元為止(抵繳集保劃撥費用或無實體登錄費用)','本案照原案表決通過']),
    '6592-2024-07-16': dict(quantity=56650045,total=566500448,raw_ex='2024-07-16',
        announced='2024-08-09',pay='2024-08-16',kind='retained_primary_pdf',
        host='apihfcweb.hfcfinance.com.tw',required=['56,650,045股,每股面額10元',
            '每仟股無償配發l00股','按面額折付現金計算至元為止','照原案通過']),
}


def load_distributions(root, document):
    root,document=Path(root).resolve(),Path(document).resolve();sources=Sources(root)
    doc=json.loads(sources.bind(str(document.relative_to(root)),digest(document)).read_text())
    if (doc.get('schema')!=SCHEMA or set(doc.get('events',{}))!=set(EXPECTED)
            or doc.get('strategy_parameters_changed') is not False or doc.get('live_qualified') is not False):
        raise ValueError('Issuer distribution scope differs')
    overrides={}
    for key,spec in EXPECTED.items():
        sid=key[:4];event=doc['events'][key];refs=event['source_sha256'];row=event['override']
        for name,sha in refs.items():sources.bind(name,sha)
        required=dict(shares_per_share=.1,pay_date=spec['pay'],ordinary_share_available_date=spec['pay'],
            listing_announcement_date=spec['announced'],fractional_cash_per_share=10,
            fractional_cash_rounding='floor_ntd',fractional_cash_pay_date=None,
            fractional_cash_payment_date_verified=False,fractional_cash_net_amount_verified=False,
            fractional_cash_available_for_trading=False,
            fractional_cash_treatment='gross_undated_receivable_upper_bound_net_unknown',
            use_scope='account_settlement_only_not_selection')
        if (any(row.get(k)!=v for k,v in required.items()) or set(row['evidence_files'])!=set(refs)
                or 'certificate_delivery_date' in row or event.get('issuer_evidence_kind')!=spec['kind']):
            raise ValueError('Issuer distribution entitlement, cash or provenance differs')
        issuer=event['issuer'];exchange=event['exchange']
        if any(item[k] not in refs for item in (issuer,exchange) for k in ('path','receipt_path')):
            raise ValueError('Issuer distribution evidence escapes inventory')
        meta=json.loads((root/issuer['receipt_path']).read_text())
        if sid=='1519':
            # This is the actual retained web-tool observation, never a claim
            # that the challenge HTML was a successful PDF download.
            url=urlparse(meta.get('issuer_url',''))
            if (digest(root/issuer['path'])!=WEB_TEXT_SHA or meta.get('retained_tool_text_sha256')!=WEB_TEXT_SHA
                    or meta.get('schema')!='retained_preexisting_web_tool_primary_observation_v1'
                    or meta.get('tool')!='web.run' or meta.get('provider_reference')!='turn60view1'
                    or meta.get('original_pdf_bytes_retained') is not False
                    or meta.get('visual_pdf_verification') is not False
                    or meta.get('local_direct_download_blocked') is not True
                    or meta.get('security_challenge_bypass') is not False
                    or event.get('evidence_note')!=TEXT_NOTE):
                raise ValueError('Retained tool text must not be promoted to raw PDF evidence')
            text=re.sub(r'^L\d+(?:@P[\d-]+)?:\s*','',(root/issuer['path']).read_text(),flags=re.M)
        else:
            url=urlparse(meta.get('url',''))
            if (meta.get('http_status')!=200 or meta.get('automatic_redirects_disabled') is not True
                    or meta.get('final_url')!=meta['url'] or meta.get('sha256')!=refs[issuer['path']]
                    or not (root/issuer['path']).read_bytes().startswith(b'%PDF')
                    or issuer.get('text_path') not in refs or issuer.get('reviewed_pdf_pages')!=[3,4]
                    or event.get('evidence_note') is not None):
                raise ValueError('Issuer PDF is not retained and visually reviewed')
            text=(root/issuer['text_path']).read_text()
        if url.scheme!='https' or url.hostname!=spec['host'] or any(normalized(t) not in normalized(text) for t in spec['required']):
            raise ValueError('Issuer adopted distribution facts differ')
        meta=json.loads((root/exchange['receipt_path']).read_text());url=urlparse(meta['url'])
        if (url.scheme!='https' or url.hostname!='www.twse.com.tw' or url.path!='/rwd/zh/announcement/LT185'
                or meta.get('http_status')!=200 or meta.get('accepted') is not True
                or meta.get('security_denied') is not False or meta.get('automatic_redirects_disabled') is not True
                or meta.get('redirect_statuses')!=[] or meta.get('params',{}).get('stockNo')!=sid
                or meta.get('raw_sha256')!=refs[exchange['path']]):
            raise ValueError('Issuer ordinary listing receipt differs')
        data=json.loads((root/exchange['path']).read_text());rows=data.get('data',[])
        if (data.get('stat')!='OK' or len(rows)!=1 or rows[0][0]!=sid
                or rows[0][2]!='(股票上市)增資新股上市' or roc_day(rows[0][5])!=spec['announced']
                or roc_day(rows[0][6])!=spec['pay'] or int(rows[0][7].replace(',',''))!=spec['quantity']
                or '含員工酬勞轉增資0股' not in normalized(rows[0][9])
                or '與原已發行之股份相同' not in rows[0][9]):
            raise ValueError('Issuer ordinary-share dates or issue quantity differ')
        feed=event['existing_dividend_feed']
        if feed not in refs:raise ValueError('Issuer ex-date feed is unbound')
        frame=pd.read_parquet(root/feed)
        selected=frame.loc[frame.stock_id.eq(sid)&frame.StockExDividendTradingDate.eq(spec['raw_ex'])]
        if (len(selected)!=1 or float(selected.iloc[0].StockEarningsDistribution)!=1
                or float(selected.iloc[0].StockStatutorySurplus)!=0
                or float(selected.iloc[0].ParticipateDistributionOfTotalShares)!=spec['total']
                or abs(Decimal('.1')*spec['total']-spec['quantity'])>=1):
            raise ValueError('Issuer final ordinary distribution differs from feed and issue')
        overrides[key]=deepcopy(row)
    return overrides,dict(sources.used)


def settlement_notes(case):
    from skills.all_entry_northstar_completion import NOTE, guard_component_rounding
    guard_component_rounding(case)
    notes = []
    events = case.get('account', {}).get('corporate_actions', [])
    for sid, day, note in [('1519', '2024-07-26', TEXT_NOTE), ('8927', '2024-09-06', NOTE)]:
        if any((r.get('stock_id'), r.get('date'), r.get('kind')) == (sid, day, 'stock_dividend') for r in events):
            notes.append(note)
    return notes


def install_distributions(runner,provider,document,root):
    rows,refs=load_distributions(root,document)
    for key,row in rows.items():
        if key in provider.corporate_overrides and provider.corporate_overrides[key]!=row:
            raise ValueError('Issuer distribution would rewrite an existing event')
    provider.corporate_overrides=dict(provider.corporate_overrides,**rows)
    writer=runner.write
    def write(path,value):
        if isinstance(value,dict) and value.get('completed') is True and 'account' in value:
            value=dict(value,settlement_evidence_notes=settlement_notes(value))
        return writer(path,value)
    runner.write=write
    return refs
