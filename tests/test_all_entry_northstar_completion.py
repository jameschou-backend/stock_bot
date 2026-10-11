import json
from pathlib import Path

import pandas as pd
import pytest

from skills.all_entry_accounts import digest
from skills import all_entry_northstar_completion as module
from skills.all_entry_issuer_distributions import settlement_notes

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = 'docs/all_entry_8927_delivery_20261011.json'

@pytest.fixture
def package(tmp_path):
    doc = json.loads((ROOT / DOCUMENT).read_text())
    for name in doc['source_sha256']: (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
    for kind, item in doc['evidence'].items():
        p = tmp_path / item['path']
        if kind in ('minutes','financial'):
            p.write_bytes(b'%PDF synthetic unit test fixture')
            (tmp_path / item['text_path']).write_text('\n'.join(module.FACTS[kind]))
        else: p.write_text('<p>' + ' '.join(module.FACTS[kind]) + '</p>')
        (tmp_path / item['receipt_path']).write_text(json.dumps(dict(url=module.URLS[kind],
            final_url=module.URLS[kind], http_status=200, automatic_redirects_disabled=True, sha256=digest(p))))
    p = tmp_path / doc['existing_dividend_feed']
    pd.DataFrame([dict(stock_id='8927', StockExDividendTradingDate='2024-09-06',
        StockEarningsDistribution=.2996241524, StockStatutorySurplus=.9987471341,
        ParticipateDistributionOfTotalShares=326851421)]).to_parquet(p)
    p.with_suffix('.json').write_text('{}')
    doc['source_sha256'] = {name: digest(tmp_path / name) for name in doc['source_sha256']}
    path = tmp_path / DOCUMENT; path.parent.mkdir(exist_ok=True)
    def save(): path.write_text(json.dumps(doc))
    save(); return tmp_path, path, doc, save


def test_final_combined_rate_and_secondary_source_limitation(package):
    root, path, _, _ = package
    row, refs = module.load_completion(root, path)
    assert len(refs) == 13 and row['shares_per_share'] == .12983712865
    assert row['pay_date'] == '2024-10-31' and row['fractional_cash_available_for_trading'] is False
    case = dict(account=dict(corporate_actions=[dict(stock_id='8927', date='2024-09-06',
        kind='stock_dividend', entitled_qty=2147, whole_new_shares=278)]))
    module.guard_component_rounding(case)
    assert settlement_notes(case) == [module.NOTE]
    case['account']['corporate_actions'][0]['entitled_qty'] = 10
    with pytest.raises(ValueError, match='component rounding'): module.guard_component_rounding(case)


@pytest.mark.parametrize('change', ['early_delivery','rounded_rate','cash','primary_claim','source_bytes','issue_quantity'])
def test_no_guessed_settlement_or_upgraded_provenance(package, change):
    root, path, doc, save = package
    if change == 'early_delivery': doc['override']['pay_date'] = '2024-09-14'
    elif change == 'rounded_rate': doc['override']['shares_per_share'] = .13
    elif change == 'cash': doc['override']['fractional_cash_available_for_trading'] = True
    elif change == 'primary_claim': doc['evidence']['listing']['source_role'] = 'primary_exchange_response'
    elif change == 'source_bytes': (root / doc['evidence']['minutes']['path']).write_bytes(b'not the source')
    else:
        item = doc['evidence']['listing']; f = root / item['path']
        f.write_text(f.read_text().replace('42,437,450', '42,437,451'))
        doc['source_sha256'][item['path']] = digest(f)
        receipt = root / item['receipt_path']; meta = json.loads(receipt.read_text()); meta['sha256'] = digest(f)
        receipt.write_text(json.dumps(meta)); doc['source_sha256'][item['receipt_path']] = digest(receipt)
    save()
    with pytest.raises(ValueError): module.load_completion(root, path)


def test_actual_primary_and_secondary_package_when_available():
    doc = json.loads((ROOT / DOCUMENT).read_text())
    if not (ROOT / doc['evidence']['minutes']['path']).exists(): pytest.skip('Source archive is local')
    row, refs = module.load_completion(ROOT, ROOT / DOCUMENT)
    assert len(refs) == 13
