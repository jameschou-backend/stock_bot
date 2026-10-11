from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skills.all_entry_accounts import digest
import skills.all_entry_context_distributions as subject


def terms():
    return dict(shares_per_share=.025, pay_date=subject.CONVERSION, certificate_trading_modeled=False)


def account():
    action = subject.KEY+'-stock'
    days = ['2026-08-05','2026-08-28',subject.CERTIFICATE,'2026-09-07',subject.CONVERSION]
    return dict(completed=True,account=dict(corporate_actions=[
        dict(kind='stock_dividend',stock_id='2801',date='2026-08-05',action_id=action,
            entitled_qty=1000,whole_new_shares=25,shares_per_share=.025,pay_date=subject.CONVERSION,
            certificate_restriction=dict(certificate_delivery_date=subject.CERTIFICATE,
                ordinary_share_available_date=subject.CONVERSION,
                valuation_basis='ordinary_share_close_proxy',certificate_trading_modeled=False)),
        dict(kind='share_delivery',stock_id='2801',date=subject.CONVERSION,action_id=action,qty=25)],
        daily=[dict(date=d) for d in days],cash_ledger=[],
        trades=[dict(date='2026-08-28',stock_id='2801',side='sell',qty=1000),
                dict(date=subject.CONVERSION,stock_id='2801',side='sell',qty=25)],
        holdings=[dict(date='2026-08-05',stock_id='2801',qty=1000)]))


def test_certificate_inventory_stays_separate_until_one_conversion():
    proof=subject.inventory_audit(account(),terms(),'fixture')
    rows=proof['events'][0]['daily']
    assert [r['restricted_units'] for r in rows]==[25,25,25,25,0]
    assert [r['ordinary_closing_units'] for r in rows]==[1000,0,0,0,0]
    assert sum(r['converted_units'] for r in rows)==25
    assert all(r['certificate_trades']==r['certificate_cash_available']==0 for r in rows)


@pytest.mark.parametrize('change',['early_delivery','early_sale','early_holding','duplicate',
    'inflated','cash_payment','flag_only','calendar_gap'])
def test_certificate_cannot_be_misrepresented_as_spendable(change):
    case=account();a=case['account']
    if change=='early_delivery':a['corporate_actions'][1]['date']=subject.CERTIFICATE
    elif change=='early_sale':a['trades'][1]['date']=subject.CERTIFICATE
    elif change=='early_holding':a['holdings'].append(dict(date=subject.CERTIFICATE,stock_id='2801',qty=25))
    elif change=='duplicate':a['corporate_actions'].append(deepcopy(a['corporate_actions'][1]))
    elif change=='inflated':a['corporate_actions'][0]['whole_new_shares']=26
    elif change=='cash_payment':a['cash_ledger'].append(dict(action_id=subject.KEY+'-stock-fractional-cash'))
    elif change=='flag_only':a['corporate_actions'][0]['certificate_restriction']['certificate_trading_modeled']=True
    else:a['daily']=[r for r in a['daily'] if r['date']!=subject.CERTIFICATE]
    with pytest.raises(ValueError):subject.inventory_audit(case,terms(),'fixture')


def test_writer_extends_prior_wrappers_without_losing_notes(monkeypatch,tmp_path):
    from skills.all_entry_issuer_distributions import TEXT_NOTE
    path=tmp_path/'doc.json';path.write_text('{}');saved=[]
    monkeypatch.setattr(subject,'load_distributions',lambda *_:({subject.KEY:terms()},{}))
    runner=SimpleNamespace(write=lambda p,v:saved.append(v));provider=SimpleNamespace(corporate_overrides={})
    subject.install_distributions(runner,provider,path,tmp_path)
    inner=runner.write
    # The older writer runs first and supplies only its own source note.
    runner.write=lambda p,v:inner(p,dict(v,settlement_evidence_notes=[TEXT_NOTE]))
    case=account()
    for sid,day in [('1519','2024-07-26'),('2887','2024-08-06')]:
        case['account']['corporate_actions'].append(dict(stock_id=sid,date=day,kind='stock_dividend'))
    runner.write('case.json',case)
    assert saved[0]['settlement_evidence_notes']==[TEXT_NOTE,subject.NOTE]
    assert saved[0]['context_certificate_inventory_audit']['events'][0]['whole_units']==25
    assert 'settlement_evidence_notes' not in case


@pytest.fixture
def archived_package(tmp_path):
    doc=json.loads((subject.ROOT/subject.DOCUMENT).read_text())
    refs={p:sha for e in doc['events'].values() for p,sha in e['source_sha256'].items()}
    if not all((subject.ROOT/p).exists() for p in refs):pytest.skip('Archived local evidence is unavailable')
    for name in refs:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((subject.ROOT/name).read_bytes())
    p=tmp_path/subject.DOCUMENT;p.parent.mkdir(parents=True);p.write_text(json.dumps(doc))
    return tmp_path,doc,p


def test_archived_primary_evidence_and_exact_feed_precision(archived_package):
    root,_,p=archived_package;rows,refs=subject.load_distributions(root,p)
    assert len(refs)==20
    assert rows['2887-2024-08-06']['shares_per_share']==.04000000003
    assert rows[subject.KEY]['pay_date']=='2026-09-08'
    assert 'per_share' not in rows[subject.KEY]  # preserve existing cash dividends


@pytest.mark.parametrize('change',['early_terms','spend_fraction','promote_text','exchange_date','source_bytes'])
def test_changed_provenance_or_settlement_cannot_be_rehashed_as_valid(archived_package,change):
    root,doc,p=archived_package
    event=doc['events'][subject.KEY]
    if change=='early_terms':event['override']['pay_date']=subject.CERTIFICATE
    elif change=='spend_fraction':event['override']['fractional_cash_available_for_trading']=True
    else:
        if change=='promote_text':
            event=doc['events']['2887-2024-08-06'];f=root/event['issuer']['receipt_path']
            value=json.loads(f.read_text());value['original_pdf_bytes_retained']=True
        elif change=='exchange_date':
            f=root/event['exchange']['path'];value=json.loads(f.read_text());value['data'][1][6]='115年08月31日'
        else:
            (root/event['issuer']['path']).write_bytes(b'%PDF forged');value=None
        if value:
            f.write_text(json.dumps(value));event['source_sha256'][str(f.relative_to(root))]=digest(f)
            if change=='exchange_date':
                receipt=root/event['exchange']['receipt_path'];meta=json.loads(receipt.read_text())
                meta['raw_sha256']=digest(f);receipt.write_text(json.dumps(meta))
                event['source_sha256'][str(receipt.relative_to(root))]=digest(receipt)
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):subject.load_distributions(root,p)
