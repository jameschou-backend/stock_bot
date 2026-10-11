from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skills.all_entry_accounts import digest
import skills.all_entry_poya_completion as subject


def terms():
    return dict(shares_per_share=.01, pay_date=subject.CONVERSION, certificate_trading_modeled=False)


def account():
    action = subject.KEY+'-stock'
    days = ['2025-07-15','2025-08-14',subject.CERTIFICATE,'2025-09-25',subject.CONVERSION]
    return dict(completed=True,account=dict(corporate_actions=[
        dict(kind='stock_dividend',stock_id='5904',date='2025-07-15',action_id=action,
            entitled_qty=1000,whole_new_shares=10,shares_per_share=.01,pay_date=subject.CONVERSION,
            certificate_restriction=dict(certificate_delivery_date=subject.CERTIFICATE,
                ordinary_share_available_date=subject.CONVERSION,
                valuation_basis='ordinary_share_close_proxy',certificate_trading_modeled=False)),
        dict(kind='share_delivery',stock_id='5904',date=subject.CONVERSION,action_id=action,qty=10)],
        daily=[dict(date=d) for d in days],cash_ledger=[],
        trades=[dict(date='2025-08-14',stock_id='5904',side='sell',qty=1000),
                dict(date=subject.CONVERSION,stock_id='5904',side='sell',qty=10)],
        holdings=[dict(date='2025-07-15',stock_id='5904',qty=1000)]))


def test_certificate_inventory_stays_separate_until_one_conversion():
    proof=subject.inventory_audit(account(),terms(),'fixture')
    rows=proof['events'][0]['daily']
    assert [r['restricted_units'] for r in rows]==[10,10,10,10,0]
    assert [r['ordinary_closing_units'] for r in rows]==[1000,0,0,0,0]
    assert sum(r['converted_units'] for r in rows)==10
    assert all(r['certificate_trades']==r['certificate_cash_available']==0 for r in rows)


@pytest.mark.parametrize('change',['early_delivery','early_sale','early_holding','duplicate',
    'inflated','cash_payment','flag_only','calendar_gap'])
def test_certificate_cannot_be_misrepresented_as_spendable(change):
    case=account();a=case['account']
    if change=='early_delivery':a['corporate_actions'][1]['date']=subject.CERTIFICATE
    elif change=='early_sale':a['trades'][1]['date']=subject.CERTIFICATE
    elif change=='early_holding':a['holdings'].append(dict(date=subject.CERTIFICATE,stock_id='5904',qty=10))
    elif change=='duplicate':a['corporate_actions'].append(deepcopy(a['corporate_actions'][1]))
    elif change=='inflated':a['corporate_actions'][0]['whole_new_shares']=26
    elif change=='cash_payment':a['cash_ledger'].append(dict(action_id=subject.KEY+'-stock-fractional-cash'))
    elif change=='flag_only':a['corporate_actions'][0]['certificate_restriction']['certificate_trading_modeled']=True
    else:a['daily']=[r for r in a['daily'] if r['date']!=subject.CERTIFICATE]
    with pytest.raises(ValueError):subject.inventory_audit(case,terms(),'fixture')


def test_writer_preserves_prior_source_notes(monkeypatch,tmp_path):
    from skills.all_entry_context_distributions import NOTE as prior_note
    path=tmp_path/'doc.json';path.write_text('{}');saved=[]
    monkeypatch.setattr(subject,'load_completion',lambda *_:(terms(),{}))
    runner=SimpleNamespace(write=lambda p,v:saved.append(v));provider=SimpleNamespace(corporate_overrides={})
    subject.install_completion(runner,provider,path,tmp_path)
    inner=runner.write
    runner.write=lambda p,v:inner(p,dict(v,settlement_evidence_notes=[prior_note]))
    case=account();case['account']['corporate_actions'].append(dict(stock_id='2887',date='2024-08-06',kind='stock_dividend'))
    runner.write('case.json',case)
    assert saved[0]['settlement_evidence_notes']==[prior_note,subject.NOTE]
    assert saved[0]['poya_certificate_inventory_audit']['events'][0]['whole_units']==10


@pytest.fixture
def archived_package(tmp_path):
    doc=json.loads((subject.ROOT/subject.DOCUMENT).read_text())
    if not all((subject.ROOT/p).exists() for p in doc['source_sha256']):
        pytest.skip('Archived local evidence is unavailable')
    for name in doc['source_sha256']:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((subject.ROOT/name).read_bytes())
    p=tmp_path/subject.DOCUMENT;p.parent.mkdir(parents=True);p.write_text(json.dumps(doc))
    return tmp_path,doc,p


def test_archived_evidence_credits_only_shareholder_shares_and_zero_net_fraction(archived_package):
    root,_,p=archived_package;row,refs=subject.load_completion(root,p)
    assert len(refs)==14 and row['shares_per_share']==.01
    assert row['pay_date']==subject.CONVERSION and row['fractional_cash_per_share']==0
    assert 'per_share' not in row  # preserve cash dividend feed


@pytest.mark.parametrize('change',['early_delivery','spend_fraction','promote_secondary','employee_shares'])
def test_changed_evidence_cannot_be_rehashed_as_valid(archived_package,change):
    root,doc,p=archived_package
    if change=='early_delivery':doc['override']['pay_date']=subject.CERTIFICATE
    elif change=='spend_fraction':doc['override']['fractional_cash_per_share']=10
    elif change=='promote_secondary':doc['evidence']['listing']['source_role']='primary_exchange'
    else:
        item=doc['evidence']['listing'];path=root/item['path']
        path.write_bytes(path.read_bytes().replace(b'458,248',b'0'))
        doc['source_sha256'][item['path']]=digest(path)
        receipt=root/item['receipt_path'];meta=json.loads(receipt.read_text());meta['sha256']=digest(path)
        receipt.write_text(json.dumps(meta));doc['source_sha256'][item['receipt_path']]=digest(receipt)
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):subject.load_completion(root,p)


def test_publication_rejects_certificate_without_independent_inventory():
    from scripts.summarize_all_entry_accounts import guard_certificate_inventory,UnverifiedCertificateInventory
    with pytest.raises(UnverifiedCertificateInventory,match='5904'):
        guard_certificate_inventory(account())


