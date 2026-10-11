import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from skills.all_entry_accounts import digest
from skills import all_entry_issuer_distributions as module

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = 'docs/all_entry_issuer_distributions_20261011.json'


@pytest.fixture
def package(tmp_path, monkeypatch):
    doc = json.loads((ROOT / DOCUMENT).read_text())
    for key, event in doc['events'].items():
        spec = module.EXPECTED[key]; sid = key[:4]
        for name in event['source_sha256']:
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        issuer = event['issuer']; path = tmp_path / issuer['path']
        url = 'https://' + spec['host'] + '/fixture.pdf'
        if sid == '1519':
            path.write_text('\n'.join('L10@P3: ' + t for t in spec['required']))
            monkeypatch.setattr(module, 'WEB_TEXT_SHA', digest(path))
            meta = dict(issuer_url=url, retained_tool_text_sha256=digest(path),
                schema='retained_preexisting_web_tool_primary_observation_v1',
                tool='web.run', provider_reference='turn60view1', original_pdf_bytes_retained=False,
                visual_pdf_verification=False, local_direct_download_blocked=True, security_challenge_bypass=False)
        else:
            path.write_bytes(b'%PDF synthetic unit test fixture')
            (tmp_path / issuer['text_path']).write_text('\n'.join(spec['required']))
            meta = dict(url=url, final_url=url, http_status=200,
                automatic_redirects_disabled=True, sha256=digest(path))
        (tmp_path / issuer['receipt_path']).write_text(json.dumps(meta))
        item = event['exchange']; path = tmp_path / item['path']
        roc = lambda d: str(int(d[:4])-1911)+'年'+d[5:7]+'月'+d[8:]+'日'
        path.write_text(json.dumps(dict(stat='OK', data=[[sid, 'fixture', '(股票上市)增資新股上市',
            1, 'fixture', roc(spec['announced']), roc(spec['pay']), str(spec['quantity']), '1',
            '與原已發行之股份相同 含員工酬勞轉增資0股']])))
        (tmp_path / item['receipt_path']).write_text(json.dumps(dict(
            url='https://www.twse.com.tw/rwd/zh/announcement/LT185', http_status=200,
            accepted=True, security_denied=False, automatic_redirects_disabled=True,
            redirect_statuses=[], params={'stockNo': sid}, raw_sha256=digest(path))))
        feed = tmp_path / event['existing_dividend_feed']
        pd.DataFrame([dict(stock_id=sid, StockExDividendTradingDate=spec['raw_ex'],
            StockEarningsDistribution=1, StockStatutorySurplus=0,
            ParticipateDistributionOfTotalShares=spec['total'])]).to_parquet(feed)
        feed.with_suffix('.json').write_text('{}')
        event['source_sha256'] = {name: digest(tmp_path / name) for name in event['source_sha256']}
    path = tmp_path / DOCUMENT; path.parent.mkdir(exist_ok=True)
    def save(): path.write_text(json.dumps(doc))
    save()
    return tmp_path, path, doc, save


def test_entitlements_do_not_make_unknown_fractional_cash_spendable(package):
    root, path, _, _ = package
    rows, refs = module.load_distributions(root, path)
    assert len(rows) == 2 and len(refs) == 14
    assert rows['1519-2024-07-26']['pay_date'] == '2024-08-30'
    assert rows['6592-2024-07-16']['pay_date'] == '2024-08-16'
    assert all(r['shares_per_share'] == .1 and r['fractional_cash_available_for_trading'] is False for r in rows.values())


@pytest.mark.parametrize('field,value', [('shares_per_share', .01), ('pay_date', '2024-08-02'),
    ('fractional_cash_available_for_trading', True), ('fractional_cash_net_amount_verified', True),
    ('fractional_cash_payment_date_verified', True), ('certificate_delivery_date', '2024-08-30')])
def test_no_wrong_ratio_record_date_or_unverified_cash(package, field, value):
    root, path, doc, save = package
    doc['events']['1519-2024-07-26']['override'][field] = value; save()
    with pytest.raises(ValueError): module.load_distributions(root, path)


@pytest.mark.parametrize('change', ['bytes', 'raw_pdf_claim', 'visual_claim', 'employee', 'certificate', 'feed'])
def test_primary_facts_and_provenance_cannot_be_promoted(package, change):
    root, path, doc, save = package; event = doc['events']['1519-2024-07-26']
    if change == 'bytes':
        (root / event['issuer']['path']).write_text('changed')
    elif change in ('raw_pdf_claim', 'visual_claim'):
        f = root / event['issuer']['receipt_path']; meta = json.loads(f.read_text())
        meta['original_pdf_bytes_retained' if change == 'raw_pdf_claim' else 'visual_pdf_verification'] = True
        f.write_text(json.dumps(meta)); event['source_sha256'][str(f.relative_to(root))] = digest(f)
    elif change == 'feed':
        f = root / event['existing_dividend_feed']; frame = pd.read_parquet(f)
        frame['StockEarningsDistribution'] = .1; frame.to_parquet(f)
        event['source_sha256'][str(f.relative_to(root))] = digest(f)
    else:
        item = event['exchange']; f = root / item['path']; data = json.loads(f.read_text())
        if change == 'employee': data['data'][0][9] = data['data'][0][9].replace('0股', '100股')
        else: data['data'][0][2] = '新股權利證書上市'
        f.write_text(json.dumps(data)); event['source_sha256'][item['path']] = digest(f)
        f = root / item['receipt_path']; meta = json.loads(f.read_text())
        meta['raw_sha256'] = event['source_sha256'][item['path']]
        f.write_text(json.dumps(meta)); event['source_sha256'][item['receipt_path']] = digest(f)
    save()
    with pytest.raises(ValueError): module.load_distributions(root, path)


def test_conflicts_rejected_and_text_only_notes_follow_actual_exposure(package):
    root, path, _, _ = package
    with pytest.raises(ValueError, match='rewrite'):
        module.install_distributions(SimpleNamespace(), SimpleNamespace(corporate_overrides={'1519-2024-07-26': {}}), path, root)
    saved = []; runner = SimpleNamespace(write=lambda p, v: saved.append(v))
    provider = SimpleNamespace(corporate_overrides={})
    module.install_distributions(runner, provider, path, root)
    case = dict(completed=True, account=dict(corporate_actions=[dict(
        stock_id='1519', date='2024-07-26', kind='stock_dividend')]))
    runner.write('case.json', case)
    assert saved[-1]['settlement_evidence_notes'] == [module.TEXT_NOTE]
    assert 'settlement_evidence_notes' not in case
    case['account']['corporate_actions'][0]['kind'] = 'cash_dividend'
    runner.write('case.json', case)
    assert saved[-1]['settlement_evidence_notes'] == []


def test_actual_primary_package_when_available():
    doc = json.loads((ROOT / DOCUMENT).read_text())
    if not (ROOT / doc['events']['1519-2024-07-26']['issuer']['path']).exists():
        pytest.skip('Primary archive is local')
    rows, refs = module.load_distributions(ROOT, ROOT / DOCUMENT)
    assert len(rows) == 2 and len(refs) == 14
