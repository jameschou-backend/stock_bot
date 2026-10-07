"""The comparison adds delivery evidence without inventing spendable proceeds."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from skills import strategy_comparison_corporate as mod
from skills.account_cohort_attribution import cohort_outcomes
from skills.million_replay import Replay
from skills.scenario_exit_replay import FractionalCashActions


def save(root, name, value):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return sha256(path.read_bytes()).hexdigest()


def evidence(root, sid='3086'):
    ex, pay, rate, face, entitlement, delivery, listing = (
        ('2024-09-24', '2024-10-29', '0.10000000423', '10', '2024-09-05', '2024-10-08', '2024-10-17')
        if sid == '3086' else
        ('2025-09-08', '2025-10-30', '0.08414122310', '5', '2025-08-20', '2025-10-23', '2025-10-28'))
    key = sid+'-'+ex
    facts = dict(ex_date=ex, stock_pay_date=pay, ordinary_share_available_date=pay,
        shares_per_share_decimal=rate, shares_per_1000='100.00000423' if sid=='3086' else '84.14122310',
        par_value_ntd=face, entitlement_announcement_date=entitlement,
        delivery_confirmation_announcement_date=delivery, listing_confirmation_announcement_date=listing,
        same_rights_as_existing_ordinary=True, fractional_cash_rounding='floor_ntd',
        fractional_cash_pay_date=None, fractional_cash_net_amount=None, spendable_fractional_cash_proven=False,
        fractional_cash_purpose='offset_book_entry_and_dematerialized_registration_fees' if sid=='3086' else 'offset_book_entry_fees')
    sources, official = {}, []
    for purpose, day in [('issuer_entitlement_and_tentative_delivery', entitlement),
                         ('issuer_confirmed_delivery', delivery), ('official_listing_confirmation', listing)]:
        name=f'.cache/source/{sid}-{day}.html'; receipt=name+'.source.json'
        url='https://mopsov.twse.com.tw/'+sid+'/'+day
        sources[name]=save(root, name, 'synthetic original '+key+' '+purpose)
        stamp=pd.Timestamp(day)
        sources[receipt]=save(root, receipt, dict(url=url, http_status=200, redirected=False, sha256=sources[name],
            announcement_row=[sid, 'synthetic', f'{stamp.year-1911:03d}/{stamp.month:02d}/{stamp.day:02d}']))
        official.append(dict(announcement_date=day, purpose=purpose, url=url, path=name,
            sha256=sources[name], receipt_path=receipt, receipt_sha256=sources[receipt], http_status=200))
    failure=f'.cache/failed/{key}.json'; sources[failure]=save(root, failure, dict(completed=False, error='pay_date missing'))
    manifest=dict(schema='strategy_comparison_corporate_evidence_v1', stock_id=sid, action_id=key,
        use_scope='account_settlement_only_not_selection', strategy_parameters_changed=False,
        live_qualified=False, verified_terms=facts, official_sources=official, source_sha256=sources)
    manifest_name=f'docs/evidence-{key}.json'
    row=dict(pay_date=pay, ordinary_share_available_date=pay, shares_per_share=float(rate),
        fractional_cash_per_share=float(face), entitlement_announcement_date=entitlement,
        delivery_announcement_date=delivery, listing_announcement_date=listing,
        use_scope='account_settlement_only_not_selection', fractional_cash_rounding='floor_ntd',
        fractional_cash_pay_date=None, fractional_cash_payment_date_verified=False,
        fractional_cash_net_amount_verified=False, fractional_cash_available_for_trading=False,
        fractional_cash_treatment='gross_undated_receivable_upper_bound_net_unknown',
        evidence_files=list(sources), evidence_manifest=dict(path=manifest_name, sha256=save(root,manifest_name,manifest)))
    document=dict(schema='strategy_comparison_corporate_completion_v1', strategy_parameters_changed=False,
        cash_supplements=[], live_qualified=False, overrides={key:row})
    save(root,mod.TERMS,document)
    return document, manifest


def rewrite(root, document, manifest):
    row=next(iter(document['overrides'].values()))
    row['evidence_manifest']['sha256']=save(root,row['evidence_manifest']['path'],manifest)
    save(root,mod.TERMS,document)


@pytest.mark.parametrize('sid,rate,face',[('3086',.10000000423,10),('8932',.08414122310,5)])
def test_verified_delivery_preserves_historical_par_and_all_source_receipts(tmp_path,sid,rate,face):
    document,manifest=evidence(tmp_path,sid)
    rows,refs,qualification=mod.load_comparison_corporate_terms(tmp_path)
    key=next(iter(rows)); row=rows[key]
    assert row['shares_per_share']==rate and row['fractional_cash_per_share']==face
    assert refs[mod.TERMS]==sha256((tmp_path/mod.TERMS).read_bytes()).hexdigest()
    assert set(manifest['source_sha256']) <= set(refs)
    assert any('failed/' in name for name in refs)
    assert qualification[key]['fractional_cash_available_for_trading'] is False
    assert qualification[key]['fractional_net_cash_amount'] is None
    assert 'upper_bound' in qualification[key]['reported_nav_basis']
    rows[key]['pay_date']='2099-01-01'
    assert mod.load_comparison_corporate_terms(tmp_path)[0]==document['overrides']


@pytest.mark.parametrize('field,bad',[
    ('shares_per_share',.1),('fractional_cash_per_share',5),('pay_date','2024-09-24'),
    ('ordinary_share_available_date','2024-10-28'),('fractional_cash_pay_date','2024-10-29'),
    ('fractional_cash_available_for_trading',True),('fractional_cash_payment_date_verified',True),
    ('fractional_cash_net_amount_verified',True),('fractional_cash_treatment','paid'),
    ('listing_announcement_date','2024-09-01'),('use_scope','selection'),
])
def test_no_guessed_settlement_term_is_accepted(tmp_path,field,bad):
    document,_=evidence(tmp_path); next(iter(document['overrides'].values()))[field]=bad
    save(tmp_path,mod.TERMS,document)
    with pytest.raises(ValueError): mod.load_comparison_corporate_terms(tmp_path)


@pytest.mark.parametrize('bad',[None,'',True,'not-a-sha','0'*64])
def test_manifest_requires_exact_saved_sha(tmp_path,bad):
    document,_=evidence(tmp_path); next(iter(document['overrides'].values()))['evidence_manifest']['sha256']=bad
    save(tmp_path,mod.TERMS,document)
    with pytest.raises(ValueError): mod.load_comparison_corporate_terms(tmp_path)


@pytest.mark.parametrize('change',['raw','receipt','manifest','failure'])
def test_all_evidence_including_failed_case_is_immutable(tmp_path,change):
    document,manifest=evidence(tmp_path)
    name=({'raw':manifest['official_sources'][0]['path'],'receipt':manifest['official_sources'][0]['receipt_path'],
        'manifest':next(iter(document['overrides'].values()))['evidence_manifest']['path'],
        'failure':next(k for k in manifest['source_sha256'] if 'failed/' in k)})[change]
    (tmp_path/name).write_text('changed')
    with pytest.raises(ValueError,match='hash changed'): mod.load_comparison_corporate_terms(tmp_path)


@pytest.mark.parametrize('change',['sid','date','status','redirect','url','only_tentative','unknown_face'])
def test_self_consistent_but_wrong_primary_evidence_is_rejected(tmp_path,change):
    document,manifest=evidence(tmp_path); source=manifest['official_sources'][0]
    receipt=json.loads((tmp_path/source['receipt_path']).read_text())
    if change=='sid': receipt['announcement_row'][0]='2330'
    elif change=='date': receipt['announcement_row'][2]='113/09/06'
    elif change=='status': receipt['http_status']=403
    elif change=='redirect': receipt['redirected']=True
    elif change=='url': receipt['url']=source['url']='https://example.com/fake'
    elif change=='only_tentative': manifest['official_sources']=manifest['official_sources'][:1]
    elif change=='unknown_face': manifest['verified_terms']['par_value_ntd']=None
    source['receipt_sha256']=save(tmp_path,source['receipt_path'],receipt)
    manifest['source_sha256'][source['receipt_path']]=source['receipt_sha256']
    rewrite(tmp_path,document,manifest)
    with pytest.raises(ValueError): mod.load_comparison_corporate_terms(tmp_path)


@pytest.mark.parametrize('path',['../escape.json','/tmp/escape.json','docs/../evidence.json'])
def test_manifest_path_cannot_escape_source_root(tmp_path,path):
    document,_=evidence(tmp_path); next(iter(document['overrides'].values()))['evidence_manifest']['path']=path
    save(tmp_path,mod.TERMS,document)
    with pytest.raises(ValueError,match='path is unsafe'): mod.load_comparison_corporate_terms(tmp_path)


def test_symlink_cannot_hide_relocated_evidence(tmp_path):
    _,manifest=evidence(tmp_path); name=manifest['official_sources'][0]['path']
    path=tmp_path/name; replacement=path.with_suffix('.original'); path.rename(replacement); path.symlink_to(replacement)
    with pytest.raises(ValueError,match='path is unavailable'): mod.load_comparison_corporate_terms(tmp_path)


@pytest.mark.parametrize('sid,qty,whole,gross',[('3086',1147,114,7),('8932',3215,270,2)])
def test_actual_share_counts_deliver_only_whole_shares_and_leave_unknown_gross_unsettled(tmp_path,sid,qty,whole,gross):
    document,_=evidence(tmp_path,sid); overrides,_,_=mod.load_comparison_corporate_terms(tmp_path)
    key,terms=next(iter(overrides.items())); ex=key[5:]; pay=terms['pay_date']; eid='actual-'+key
    row=dict(action_id=key+'-stock',stock_id=sid,kind='stock_dividend',shares_per_share=terms['shares_per_share'],
        pay_date=pay,fractional_cash_per_share=terms['fractional_cash_per_share'],source=terms['evidence_manifest'])
    provider=SimpleNamespace(overrides=overrides,on_date=lambda stock,day:[row] if stock==sid and day==ex else [])
    account=Replay.__new__(Replay)
    account.holdings={sid:dict(qty=qty,event_id=eid,due_index=9)}
    account.cohorts=[dict(event_id=eid,stock_id=sid,name='synthetic cash path',entry_date='2024-01-02',due_index=9)]
    account.receivables=[]; account.actions=[]; account.cash_ledger=[]; account.cash=1000.; account.marks={sid:dict(price=50.)}
    account.raw=lambda day,stock:dict(close=50.)
    account.corporate=FractionalCashActions(provider,account)
    assert account.corporate_day(pd.Timestamp(ex))==gross
    shares,=[r for r in account.receivables if r['kind']=='shares']
    assert shares['qty']==whole and shares['pay_date']==pay
    # The original sale may precede whole-share delivery. Delivery preserves the
    # original cohort and due date; it does not turn the pending fraction into cash.
    account.holdings={}
    account.corporate_day(pd.Timestamp(pay))
    assert account.holdings[sid]==dict(qty=whole,event_id=eid,due_index=9)
    account.holdings={}
    account.corporate_day(pd.Timestamp('2026-10-02'))
    pending,=account.receivables
    assert pending['kind']=='cash' and pending['amount']==gross and pending['pay_date'] is None
    assert pending['action_id']==key+'-stock-fractional-cash'
    assert account.receivable_value()==gross and account.cash==1000.
    assert not [r for r in account.cash_ledger if r['cash_change']>0]
    # A closed stock trade still has unsettled rights and cannot count as a fully
    # settled win. This minimal cash path verifies the shared attribution seam.
    journal=dict(cohorts=account.cohorts,corporate_actions=account.actions,receivables=account.receivables,
        cash_ledger=[dict(kind='buy',event_id=eid,stock_id=sid,cash_change=-1000.),
                     dict(kind='sell',event_id=eid,stock_id=sid,cash_change=1000.)])
    summary=dict(final_holdings=[],final_receivables=account.receivables,receivable=gross,market_value=0,
        initial_cash=1000.,cash=1000.,profit=gross)
    result=cohort_outcomes(dict(account=journal,summary=summary))[eid]
    assert result['closed'] and not result['settled'] and result['receivable']==gross


def test_adapter_binds_new_terms_helper_and_unknown_payment_source_closure(tmp_path):
    from skills.strategy_comparison_data import StrategyComparisonData
    evidence(tmp_path)
    provider=StrategyComparisonData.__new__(StrategyComparisonData)
    provider.root=tmp_path; provider.corporate_overrides={'unchanged-old-action':{'pay_date':'2020-01-01'}}
    bound={}
    provider._bind=lambda path,expected=None: bound.setdefault(Path(path),expected)
    provider._load_corporate_terms()
    key='3086-2024-09-24'
    assert provider.corporate_overrides['unchanged-old-action']=={'pay_date':'2020-01-01'}
    assert provider.corporate_overrides[key]['fractional_cash_pay_date'] is None
    assert tmp_path/mod.TERMS in bound and Path(mod.__file__) in bound
    assert any(str(path).endswith('.html.source.json') for path in bound)
    assert provider.comparison_corporate_qualifications[key]['fees_and_net_settlement_verified'] is False
    for path,expected in bound.items():
        if expected is not None: assert expected==sha256(path.read_bytes()).hexdigest()


def test_adapter_never_replaces_an_existing_corporate_action(tmp_path):
    from skills.strategy_comparison_data import StrategyComparisonData
    evidence(tmp_path)
    provider=StrategyComparisonData.__new__(StrategyComparisonData); provider.root=tmp_path
    provider.corporate_overrides={'3086-2024-09-24':{'pay_date':'2024-09-24'}}
    provider._bind=lambda *args: None
    before=deepcopy(provider.corporate_overrides)
    with pytest.raises(ValueError,match='conflicts with existing terms'): provider._load_corporate_terms()
    assert provider.corporate_overrides==before
