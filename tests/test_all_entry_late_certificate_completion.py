from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skills.all_entry_accounts import digest
from skills.all_entry_late_certificate_completion import (
    DOCUMENT, KEY, ROOT, POLICY, EX_DATE, CERTIFICATE, CONVERSION, END,
    inventory_audit, load_completion, verify_inventory_proof, install_completion,
)


def terms():
    return json.loads((ROOT/DOCUMENT).read_text())['override']


def account():
    action=KEY+'-stock'
    issue=dict(action_id=action,kind='stock_dividend',stock_id='9941',date=EX_DATE,
        entitled_qty=3362,whole_new_shares=67,shares_per_share=.02,pay_date=CONVERSION,
        certificate_restriction=dict(certificate_delivery_date=CERTIFICATE,
            ordinary_share_available_date=CONVERSION,valuation_basis='ordinary_share_close_proxy',certificate_trading_modeled=False))
    days=[EX_DATE,'2026-09-22',CERTIFICATE,END]
    return dict(completed=True,account=dict(corporate_actions=[issue],cash_ledger=[],
        daily=[dict(date=d) for d in days],
        trades=[dict(date='2026-09-22',stock_id='9941',side='sell',qty=3362)],
        holdings=[dict(date=EX_DATE,stock_id='9941',qty=3362)],
        receivables=[dict(action_id=action,kind='shares',stock_id='9941',qty=67,pay_date=CONVERSION)]))


def test_late_conversion_preserves_certificate_units_and_no_ordinary_sale():
    proof=inventory_audit(account(),terms(),'bound-doc')
    event=proof['events'][0];assert event['ending_restricted_units']==67
    assert [r['state'] for r in event['daily']]==['pending_entitlement']*2+['nontraded_certificate_inventory']*2
    assert [r['ordinary_closing_units'] for r in event['daily']]==[3362,0,0,0]
    assert all(r['restricted_units']==67 and r['converted_units']==0 and r['certificate_cash_available']==0 for r in event['daily'])


@pytest.mark.parametrize('change',['early_delivery','early_sale','early_inventory','lost_claim','inflated_claim','cash_paid','duplicate_issue','missing_end','flag_only'])
def test_certificate_violations_block_publication(change):
    case=account();a=case['account']
    if change=='early_delivery':a['corporate_actions'].append(dict(kind='share_delivery',action_id=KEY+'-stock',date=CERTIFICATE,qty=67))
    elif change=='early_sale':a['trades'].append(dict(date=CERTIFICATE,stock_id='9941',side='sell',qty=67))
    elif change=='early_inventory':a['holdings'].append(dict(date=CERTIFICATE,stock_id='9941',qty=67))
    elif change=='lost_claim':a['receivables']=[]
    elif change=='inflated_claim':a['receivables'][0]['qty']=68
    elif change=='cash_paid':a['cash_ledger'].append(dict(action_id=KEY+'-stock-fractional-cash',cash_change=2))
    elif change=='duplicate_issue':a['corporate_actions'].append(deepcopy(a['corporate_actions'][0]))
    elif change=='missing_end':a['daily'].pop()
    else:a['corporate_actions'][0]['certificate_restriction']['certificate_trading_modeled']=True
    with pytest.raises(ValueError):inventory_audit(case,terms(),'bound-doc')


@pytest.fixture
def primary_package(tmp_path):
    doc=json.loads((ROOT/DOCUMENT).read_text())
    if not (ROOT/doc['issuer']['path']).exists():pytest.skip('Archived primary evidence is in the local research cache')
    for name in doc['source_sha256']:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    p=tmp_path/DOCUMENT;p.parent.mkdir(parents=True);p.write_text(json.dumps(doc))
    return tmp_path,p,doc


def test_bound_primary_proof_recomputed_before_ranking(primary_package):
    root,path,doc=primary_package;row,refs=load_completion(root,path);assert len(refs)==9
    saved=[];runner=SimpleNamespace(write=lambda p,v:saved.append(v));provider=SimpleNamespace(corporate_overrides={})
    install_completion(runner,provider,path,root);assert provider.corporate_overrides[KEY]==row
    case=account();runner.write('case.json',case);assert 'late_certificate_inventory_audit' not in case
    assert verify_inventory_proof(saved[0],root)['events'][0]['whole_units']==67
    saved[0]['late_certificate_inventory_audit']['events'][0]['ending_restricted_units']=0
    with pytest.raises(ValueError,match='journal'):verify_inventory_proof(saved[0],root)


@pytest.mark.parametrize('change',['early_terms','cash_available','primary_bytes','rehashed_conversion'])
def test_primary_delivery_scope_cannot_be_rewritten(primary_package,change):
    root,path,doc=primary_package
    if change=='early_terms':doc['override']['pay_date']=CERTIFICATE
    elif change=='cash_available':doc['override']['fractional_cash_available_for_trading']=True
    elif change=='primary_bytes':(root/doc['issuer']['path']).write_bytes(b'%PDF tampered')
    else:
        item=doc['exchange'];p=root/item['path'];v=json.loads(p.read_text());v['data'][1][6]='115年09月23日';p.write_text(json.dumps(v));doc['source_sha256'][item['path']]=digest(p)
        p=root/item['receipt_path'];v=json.loads(p.read_text());v['raw_sha256']=doc['source_sha256'][item['path']];p.write_text(json.dumps(v));doc['source_sha256'][item['receipt_path']]=digest(p)
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):load_completion(root,path)


def test_missing_unit_proof_cannot_pass_guard():
    from scripts.summarize_all_entry_accounts import guard_certificate_inventory,UnverifiedCertificateInventory
    with pytest.raises(UnverifiedCertificateInventory,match='9941'):guard_certificate_inventory(account())
