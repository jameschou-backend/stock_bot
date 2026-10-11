"""One retrospectively verified ordinary-share delivery; no signal inputs."""
from copy import deepcopy
import json
from pathlib import Path

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest

KEY = '5386-2024-08-22'
URLS = {
    'minutes': 'https://www.albatron.com.tw/db/upload/webdata15/5albatron_20248221642682725.pdf',
    'annual': 'https://www.albatron.com.tw/db/upload/webdata15/5albatron_202582814392155215.pdf',
}
EXPECTED = dict(pay_date='2024-09-27', ordinary_share_available_date='2024-09-27',
    shares_per_share=.09, fractional_cash_per_share=0, fractional_net_zero_verified=True,
    fractional_cash_rounding='floor_ntd', fractional_cash_pay_date=None,
    fractional_cash_payment_date_verified=False, fractional_cash_net_amount_verified=True,
    fractional_cash_available_for_trading=False,
    fractional_cash_treatment='net_zero_book_entry_fees_verified',
    use_scope='account_settlement_only_not_selection', entitlement_announcement_date='2024-06-07')


def load_albatron(root, document):
    root, document = Path(root).resolve(), Path(document).resolve()
    sources = Sources(root)
    doc = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    if (doc.get('schema') != 'all_entry_albatron_completion_v1' or doc.get('key') != KEY
            or doc.get('strategy_parameters_changed') is not False or doc.get('live_qualified') is not False
            or doc.get('source_policy') != 'issuer_minutes_and_retrospective_annual_execution'):
        raise ValueError('Albatron settlement scope differs')
    row, evidence = doc['replacement'], doc['evidence']
    if (set(row) != set(EXPECTED) | {'evidence_files', 'source'}
            or any(row[k] != v for k, v in EXPECTED.items())
            or set(evidence) != {'minutes', 'annual'}):
        raise ValueError('Albatron delivery or fractional cash differs')
    inventory = set()
    for kind, item in evidence.items():
        path = sources.bind(item['path'], item['sha256'])
        receipt = json.loads(sources.bind(item['receipt_path'], item['receipt_sha256']).read_text())
        if (receipt.get('url') != URLS[kind] or receipt.get('http_status') != 200
                or receipt.get('redirected') is not False or receipt.get('sha256') != item['sha256']
                or not path.read_bytes().startswith(b'%PDF')):
            raise ValueError('Albatron issuer PDF source differs')
        inventory.update([item['path'], item['receipt_path']])
    minutes, annual = evidence['minutes'], evidence['annual']
    if (minutes.get('reviewed_pdf_pages') != [1, 3, 4]
            or minutes.get('review_method') != 'visually_read_scanned_original'
            or minutes.get('reviewed_facts') != dict(approved=True, shares_per_1000=90,
                new_ordinary_shares=2577080, par_value=10, same_rights_as_ordinary=True,
                book_entry_fractional_cash_used_for_processing_fees=True)
            or annual.get('reviewed_pdf_pages') != [37]
            or annual.get('extraction_command') != 'pdftotext -layout'):
        raise ValueError('Albatron reviewed primary facts differ')
    text = ''.join(sources.bind(annual['text_path'], annual['text_sha256']).read_text().split())
    inventory.add(annual['text_path'])
    # This completed-event disclosure is only for settlement reconstruction.
    # It is published later and must never be exposed to historical selection.
    needle = '盈餘轉增資發行新股案。執行情形：決議通過。訂定113年8月30日為除權基準日，113年9月27日為股票發放日。'
    if needle not in text or set(row['evidence_files']) != inventory:
        raise ValueError('Albatron annual execution date is not proved')
    return deepcopy(row), dict(sources.used)


def install_albatron(provider, document, root):
    row, refs = load_albatron(root, document)
    if KEY in provider.corporate_overrides and provider.corporate_overrides[KEY] != row:
        raise ValueError('Albatron completion would rewrite existing settlement')
    provider.corporate_overrides = dict(provider.corporate_overrides, **{KEY: row})
    return refs
