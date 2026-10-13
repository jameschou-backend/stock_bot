"""Verified 2236 distribution encountered by the new holding path only."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest
from skills.all_entry_bank_distributions import normalized
from skills.poc_followup_data import FollowupData

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT/'docs/poc_followup_2236_delivery_20261014.json'
KEY = '2236-2024-09-25'
EXPECTED = dict(shares_per_share=.021397789, pay_date='2024-10-23',
    ordinary_share_available_date='2024-10-23', entitlement_announcement_date='2024-09-09',
    delivery_announcement_date='2024-09-09', listing_announcement_date='2024-10-15',
    fractional_cash_per_share=0, fractional_net_zero_verified=True,
    fractional_cash_rounding='floor_ntd', fractional_cash_pay_date=None,
    fractional_cash_payment_date_verified=False, fractional_cash_net_amount_verified=True,
    fractional_cash_available_for_trading=False, fractional_cash_treatment='net_zero_book_entry_fees_verified',
    use_scope='account_settlement_only_not_selection')
FACTS = {
    'entitlement': ['2236', '21.39778900', '0.21397789', '51,809,419', '113年09月25日'],
    'delivery': ['1,108,607股', '每股股票面額新台幣10元', '113年10月23日起開始發放'],
    'listing': ['普通股1,108,607股', '113年10月23日', '與原已發行之股份相同'],
    'minutes': ['發行新股1,108,607股', '其未滿一股之畸零股款，將做為處理帳簿劃撥之費用', '本案照案通過'],
}


def load(root=ROOT, document=DOCUMENT):
    root, document = Path(root), Path(document)
    sources = Sources(root)
    doc = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    if (doc.get('schema') != 'poc_followup_2236_delivery_v1' or doc.get('event') != KEY
            or doc.get('strategy_parameters_changed') is not False or doc.get('live_qualified') is not False
            or any(doc['override'].get(k) != v for k, v in EXPECTED.items())):
        raise ValueError('2236 distribution scope or settlement differs')
    refs = doc['source_sha256']
    if set(doc['override']['evidence_files']) != set(refs):
        raise ValueError('2236 evidence inventory differs')
    for name, sha in refs.items(): sources.bind(name, sha)
    for kind, required in FACTS.items():
        item = doc['evidence'][kind]
        if any(item[k] not in refs for k in ('path', 'receipt_path')):
            raise ValueError('2236 source escapes inventory')
        meta = json.loads((root/item['receipt_path']).read_text())
        host = urlparse(meta['url']).hostname
        if (meta.get('http_status') != 200 or meta.get('redirected') is not False
                or meta.get('sha256') != refs[item['path']]
                or urlparse(meta['url']).scheme != 'https'
                or host != ('doc.twse.com.tw' if kind == 'minutes' else 'mopsov.twse.com.tw')):
            raise ValueError('2236 primary receipt differs')
        if kind == 'minutes':
            if (not (root/item['path']).read_bytes().startswith(b'%PDF')
                    or item.get('reviewed_pdf_pages') != [2,3] or item['text_path'] not in refs):
                raise ValueError('2236 adopted PDF has not been reviewed')
            text = (root/item['text_path']).read_text()
        else:
            if meta.get('security_denied') is not False or meta.get('announcement_row', [])[0] != '2236':
                raise ValueError('2236 announcement identity differs')
            expected_date = '113/10/15' if kind == 'listing' else '113/09/09'
            if meta['announcement_row'][2] != expected_date:
                raise ValueError('2236 announcement date differs')
            text = BeautifulSoup((root/item['path']).read_bytes(), 'html.parser').get_text()
        if any(normalized(x) not in normalized(text) for x in required):
            raise ValueError('2236 primary facts differ: '+kind)
    if abs(Decimal('51809419')*Decimal('.021397789')-Decimal('1108607')) >= 1:
        raise ValueError('2236 final rate and issued quantity differ')
    return deepcopy(doc['override']), dict(sources.used)


class CompletedFollowupData(FollowupData):
    def __init__(self, *, online=False):
        super().__init__(online=online)
        row, refs = load()
        if KEY in self.corporate_overrides:
            raise ValueError('2236 completion cannot rewrite an existing event')
        self.corporate_overrides[KEY] = row
        for name, sha in refs.items(): self._bind(ROOT/name, sha)
        self._bind(Path(__file__))
