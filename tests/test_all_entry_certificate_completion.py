from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skills.all_entry_accounts import digest
from skills.all_entry_certificate_completion import (
    KEY, POLICY, DOCUMENT, CERTIFICATE, CONVERSION, load_certificates,
    inventory_audit, verify_inventory_proof, install_certificates,
)
from skills.all_entry_settlement_completion import terms_hash
from scripts.research_poc_latest_account import merge_overrides

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def package(tmp_path):
    doc=json.loads((ROOT/DOCUMENT).read_text())
    oldpath=tmp_path/doc['previous_document']['path'];oldpath.parent.mkdir(parents=True)
    previous=json.loads((ROOT/doc['previous_document']['path']).read_text())['overrides'][KEY]
    oldpath.write_text(json.dumps({'overrides':{KEY:previous}}))
    doc['previous_document']['sha256']=digest(oldpath);doc['previous_terms_sha256']=terms_hash(previous)
    for name in doc['source_sha256']:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic fixture')
    issuer=doc['issuer'];p=tmp_path/issuer['path'];p.write_bytes(b'%PDF synthetic issuer fixture')
    (tmp_path/issuer['text_path']).write_text('每壹仟股無償配發法定盈餘公積、資本公積轉增資10.00000000股 增資新股權利證書發放日為115年08月20日 元以下捨去')
    exchange=doc['exchange'];(tmp_path/exchange['path']).write_text(json.dumps(dict(stat='OK',data=[
        ['5876','fixture','新股權利證書上市',1,'fixture','115年08月17日','115年08月20日','48,616,031','0','fixture'],
        ['5876','fixture','(股票上市)新股權利證書終止上市，增資股股票上市',2,'fixture','115年09月08日','115年09月11日','48,616,031','0','普通股48,616,031股 新股權利證書終止上市日期']
    ])))
    for item,url in [(issuer,'https://www.scsb.com.tw/fixture.pdf'),(exchange,'https://www.twse.com.tw/rwd/zh/announcement/LT185?stockNo=5876')]:
        (tmp_path/item['receipt_path']).write_text(json.dumps(dict(url=url,final_url=url,status_code=200,sha256=digest(tmp_path/item['path']))))
    doc['source_sha256']={name:digest(tmp_path/name) for name in doc['source_sha256']}
    path=tmp_path/DOCUMENT
    def save():path.write_text(json.dumps(doc))
    save();return tmp_path,doc,save


def account():
    action='5876-2026-07-28-stock'
    issue=dict(kind='stock_dividend',stock_id='5876',date='2026-07-28',action_id=action,
        entitled_qty=1000,whole_new_shares=10,shares_per_share=.01,pay_date=CONVERSION,
        certificate_restriction=dict(certificate_delivery_date=CERTIFICATE,
            ordinary_share_available_date=CONVERSION,valuation_basis='ordinary_share_close_proxy',certificate_trading_modeled=False))
    days=['2026-07-28','2026-08-19',CERTIFICATE,'2026-09-10',CONVERSION]
    return dict(completed=True,account=dict(corporate_actions=[issue,
        dict(kind='share_delivery',stock_id='5876',date=CONVERSION,action_id=action,qty=10)],
        daily=[dict(date=d) for d in days],cash_ledger=[],
        trades=[dict(date='2026-08-19',stock_id='5876',side='sell',qty=1000),
                dict(date=CONVERSION,stock_id='5876',side='sell',qty=10)],
        holdings=[dict(date='2026-07-28',stock_id='5876',qty=1000)]))


def test_independent_units_cannot_be_spent_before_one_conversion(package):
    root,doc,_=package;_,terms,_=load_certificates(root,root/DOCUMENT)
    case=account();proof=inventory_audit(case,terms,digest(root/DOCUMENT))
    rows=proof['events'][0]['daily']
    assert [r['state'] for r in rows]==['pending_entitlement','pending_entitlement','nontraded_certificate_inventory','nontraded_certificate_inventory','converted_to_ordinary']
    assert [r['restricted_units'] for r in rows]==[10,10,10,10,0]
    assert sum(r['converted_units'] for r in rows)==10
    assert all(r['certificate_cash_available']==0 and r['certificate_trades']==0 for r in rows)
    case['certificate_inventory_audit']=proof
    assert verify_inventory_proof(case,root)==proof
    case['certificate_inventory_audit']['events'][0]['daily'][2]['restricted_units']=0
    with pytest.raises(ValueError,match='journal'):verify_inventory_proof(case,root)


@pytest.mark.parametrize('change',['early_delivery','early_sale','early_holding','duplicate','inflated','cash_payment','flag_only'])
def test_old_fungible_inventory_and_flags_cannot_pass(package,change):
    root,doc,_=package;_,terms,_=load_certificates(root,root/DOCUMENT);case=account();a=case['account']
    if change=='early_delivery':a['corporate_actions'][1]['date']=CERTIFICATE
    elif change=='early_sale':a['trades'][1]['date']=CERTIFICATE
    elif change=='early_holding':a['holdings'].append(dict(date=CERTIFICATE,stock_id='5876',qty=10))
    elif change=='duplicate':a['corporate_actions'].append(deepcopy(a['corporate_actions'][1]))
    elif change=='inflated':a['corporate_actions'][0]['whole_new_shares']=11
    elif change=='cash_payment':a['cash_ledger'].append(dict(action_id=KEY+'-stock-fractional-cash',cash_change=5))
    else:a['corporate_actions'][0]['certificate_restriction']['certificate_trading_modeled']=True
    with pytest.raises(ValueError):inventory_audit(case,terms,digest(root/DOCUMENT))


@pytest.mark.parametrize('change',['early_terms','cash_change','primary_bytes','rehashed_conversion'])
def test_exact_primary_sources_and_unchanged_entitlement(package,change):
    root,doc,save=package
    if change=='early_terms':doc['replacement']['pay_date']=CERTIFICATE
    elif change=='cash_change':doc['replacement']['fractional_cash_per_share']=0
    elif change=='primary_bytes':(root/doc['issuer']['path']).write_bytes(b'%PDF changed')
    else:
        item=doc['exchange'];p=root/item['path'];v=json.loads(p.read_text());v['data'][1][6]='115年09月10日';p.write_text(json.dumps(v))
        doc['source_sha256'][item['path']]=digest(p)
        p=root/item['receipt_path'];v=json.loads(p.read_text());v['sha256']=doc['source_sha256'][item['path']];p.write_text(json.dumps(v));doc['source_sha256'][item['receipt_path']]=digest(p)
    save()
    with pytest.raises(ValueError):load_certificates(root,root/DOCUMENT)


def test_completion_replaces_only_exact_old_event_and_seals_unit_journal(package):
    root,doc,_=package;old,terms,_=load_certificates(root,root/DOCUMENT);saved=[]
    runner=SimpleNamespace(merge_overrides=merge_overrides,write=lambda p,v:saved.append(v))
    provider=SimpleNamespace(corporate_overrides={})
    install_certificates(runner,provider,root/DOCUMENT,root)
    assert runner.merge_overrides({KEY:old,'unrelated':42},provider.corporate_overrides)=={KEY:terms,'unrelated':42}
    assert runner.merge_overrides({KEY:terms,'unrelated':42},{KEY:old})=={KEY:terms,'unrelated':42}
    with pytest.raises(ValueError):runner.merge_overrides({KEY:terms},{KEY:dict(old,pay_date='2026-08-21')})
    with pytest.raises(ValueError,match='changed'):runner.merge_overrides({KEY:{}},provider.corporate_overrides)
    case=account();runner.write('case.json',case)
    assert 'certificate_inventory_audit' not in case
    assert verify_inventory_proof(saved[0],root)['events'][0]['whole_units']==10


def test_real_archived_primary_sources():
    if not (ROOT/'.cache/poc-range-first-20261005/corporate-repair/5876/twse-LT185-5876-2026.json').exists():
        pytest.skip('Primary evidence retained in local research cache')
    old,terms,refs=load_certificates(ROOT,ROOT/DOCUMENT)
    assert old['pay_date']==CERTIFICATE and terms['pay_date']==CONVERSION and len(refs)==11
