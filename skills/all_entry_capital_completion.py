"""Evidence-bound 2352 cash reduction, with unspendable fractional proceeds.

This routes one newly evidenced event to the existing historical-close capital
settlement engine. Other events retain the frozen face-value/strict adapters.
"""
from copy import deepcopy
import json
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import pandas as pd

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest
from skills.face_value_capital import FaceValueCapitalActions
from skills.million_replay import UnresolvedAction
from skills.prepared_corporate_settlement import CapitalSettlementActions

KEY = '2352-2025-10-07'
TERMS = dict(kind='capital_reduction', multiplier=.82, cash_per_share=1.8,
    cash_rounding='floor_ntd', known_date='2025-08-04', pay_date='2025-10-07',
    fractional_reference_date='2025-09-24', fractional_policy='historical_close_gross_receivable',
    fractional_cash_rounding='floor_ntd', fractional_cash_pay_date=None,
    fractional_cash_net_amount_verified=False, fractional_cash_available_for_trading=False,
    use_scope='account_settlement_only_not_selection')
PDF_HASHES = {
    'meeting': '19f61a093d13532f15f3b54890bed1d75cd6a7926ff153ec4d37a3dcd64119a9',
    'annual': '87929c1b21e28c9e78c9b6041c7154c8897e033e21b94fed05709240e4375cde',
    'exchange': 'dc25043d74b811ce5f4355708affb1c97ae5011b6ddd54bf8311f552f853e4c5',
}
MOPS_TEXT = ('公司代號:2352', '公告日期:20250804', '每仟股換發820股',
    '退還股款新台幣1,800元', '民國114年9月24日', '民國114年10月7日',
    '最後交易日之收盤價', '抵繳集保劃撥費用或無實體登錄費用', '元以下捨去')


def load_capital_completion(root, document):
    root, document = Path(root).resolve(), Path(document).resolve()
    sources = Sources(root)
    value = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    row = value.get('terms', {})
    if (value.get('schema') != 'all_entry_2352_capital_completion_v1' or value.get('key') != KEY
            or value.get('strategy_parameters_changed') is not False
            or value.get('live_qualified') is not False
            or any(row.get(k) != v for k, v in TERMS.items())
            or set(value.get('sources', {})) != {'meeting', 'annual', 'exchange', 'mops_republication'}):
        raise ValueError('Capital completion changes the reviewed entitlement or scope')
    evidence = []
    for purpose, notice in value['sources'].items():
        path = sources.bind(notice['path'], notice['sha256'])
        receipt = json.loads(sources.bind(notice['receipt_path'], notice['receipt_sha256']).read_text())
        host = urlparse(notice['url']).hostname
        expected_host = ('image-prod.qisda.com' if purpose in ('meeting', 'annual')
                         else 'www.pfcf.com.tw' if purpose == 'exchange' else 'tw.tradingview.com')
        if (not notice['url'].startswith('https://') or host != expected_host
                or receipt['url'] != notice['url'] or receipt.get('http_status', receipt.get('status')) != 200
                or receipt.get('redirected', False) is not False
                or receipt['sha256'] != notice['sha256']):
            raise ValueError('Capital source receipt differs')
        if purpose in PDF_HASHES:
            if notice['sha256'] != PDF_HASHES[purpose] or not path.read_bytes().startswith(b'%PDF'):
                raise ValueError('Visually reviewed capital PDF changed')
        else:
            body = ''.join(BeautifulSoup(path.read_bytes(), 'html.parser').get_text().split())
            if any(text not in body for text in MOPS_TEXT):
                raise ValueError('Reproduced MOPS announcement lacks final capital terms')
        evidence.extend((notice['path'], notice['receipt_path']))
    if set(row.get('evidence_files', [])) != set(evidence):
        raise ValueError('Capital evidence closure differs')
    # Require the existing sealed raw close and exchange event; never infer the
    # conversion or cash rate from a price discontinuity.
    quote = value['raw_close']; event = value['official_event']
    qpath = sources.bind(quote['path'], quote['sha256'])
    epath = sources.bind(event['path'], event['sha256'])
    if (quote['sha256'] != '449edc32c17904cf105a8e19497b8a502cbe3c375a4b02eca61bde9e680e97b8'
            or event['sha256'] != '783a2d9e800004e51c0a343cd02409099a3d646db5770335d59db4015200ed74'):
        raise ValueError('Capital comparison uses another market-data revision')
    q = pd.read_parquet(qpath, columns=['date', '2352'])
    valid = q.loc[q.date.lt('2025-10-07') & q['2352'].gt(0)]
    e = pd.read_parquet(epath)
    match = e.loc[e.stock_id.eq('2352') & e.event_date.astype(str).eq('2025-10-07')]
    if (str(valid.iloc[-1].date.date()) != TERMS['fractional_reference_date']
            or valid.iloc[-1]['2352'] != 30.15 or len(match) != 1
            or match.iloc[0]['source'] != 'capital_reduction' or match.iloc[0].prev_close != 30.15
            or match.iloc[0].opening_ref != 34.55):
        raise ValueError('Capital exchange date or historical quote differs')
    return KEY, deepcopy(row), dict(sources.used)


class CapitalCompletionRoute:
    def __init__(self, original, account):
        if not isinstance(original, FaceValueCapitalActions):
            raise ValueError('Unexpected frozen corporate adapter')
        self.original = original
        self.historical = CapitalSettlementActions(original.provider, account)
        self.account = account

    def __getattr__(self, name):
        return getattr(self.original, name)

    def on_date(self, sid, day):
        if (sid, day) != ('2352', '2025-10-07'):
            return self.original.on_date(sid, day)
        rows = self.original.provider.on_date(sid, day)
        if (len(rows) != 1 or any(rows[0].get(k) != v for k, v in TERMS.items())
                or self.account.raw(pd.Timestamp('2025-09-24'), sid) != 30.15):
            raise UnresolvedAction('2352 capital terms or fractional close differs')
        return self.historical.on_date(sid, day)


class CompletedCapitalAccount:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corporate = CapitalCompletionRoute(self.corporate, self)


def install_capital_completion(runner, provider, document, root):
    key, row, refs = load_capital_completion(root, document)
    if key in provider.corporate_overrides and provider.corporate_overrides[key] != row:
        raise ValueError('Capital completion conflicts with an existing entitlement')
    provider.corporate_overrides = dict(provider.corporate_overrides, **{key: row})
    original = runner.engine_types
    def engine_types(*args, **kwargs):
        account, benchmark = original(*args, **kwargs)
        return type('CompletedCapital'+account.__name__, (CompletedCapitalAccount, account), {}), benchmark
    runner.engine_types = engine_types
    return refs
