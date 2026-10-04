"""Explicit unknown exclusions must not change known POC or live reservations."""
from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import pandas as pd
import pytest

from skills import poc_gap_selection as gap
from skills.volume_profile_selection import select_candidates, UnresolvedProfile
from skills.volume_profile_account_adapter import AccountCandidateHook, ReservationPlanner, AccountReservationAudit
from skills.five_axis_replay import FiveAxisReplay
from test_volume_profile_selection import Planner, events, Provider


def run(source, flags, arm='poc_priority', **settings):
    planner = Planner(**settings); before = deepcopy(planner.__dict__)
    provider = Provider(flags)
    result = gap.select_with_profile_gaps(source,arm=arm,provider=provider,planner=planner,
        planner_factory=lambda _:Planner(**settings))
    assert planner.__dict__ == before
    return result,provider


def real_engine(count=6):
    day=pd.Timestamp('2024-01-03')
    value=SimpleNamespace(opening_limit=1_000_000.,cash=1_000_000.,opening_members=set(),
        released_residuals=set(),holdings={},slots=3,previous_nav=1_000_000.,
        residual_budget=1_000_000/3,residual_block=False,prior=lambda d,s:100.,
        amount20=pd.DataFrame([[100e6]*count],index=[day],columns=[str(1000+i) for i in range(count)]))
    return value,day


def test_real_planner_restarts_from_opening_and_requests_each_event_once():
    live,day=real_engine();planner=ReservationPlanner(live,day);before=planner.snapshot()
    provider=Provider(dict(e0=True,e1=None,e2=True,e3=False,e4=None,e5=True))
    result=gap.select_with_profile_gaps(events(6),arm='poc_priority',provider=provider,planner=planner)
    cert=result['certificate']
    assert [r['event_id'] for r in result['events']]==['e0','e2','e5']
    assert provider.requested==['e0','e1','e2','e3','e4','e5']
    assert planner.snapshot()==before and live.cash==1_000_000 and live.holdings=={}
    assert cert['reserved_event_ids']==['e0','e2','e5'] and cert['selection_passes']==3
    assert cert['initial_state']==before and cert['final_state']['occupied']==['1000','1002','1005']
    assert cert['original_ordered_ids']==[f'e{i}' for i in range(6)]
    assert cert['retained_ordered_ids']==['e0','e2','e3','e5']
    assert [r['event_id'] for r in cert['profile_data_exclusions']]==['e1','e4']
    assert all(r['available'] is False and r['poc_up'] is None and r['applied'] for r in cert['profile_data_exclusions'])
    assert cert['decisions'][1]['profile_status']=='unknown'
    assert cert['decisions'][1]['selection_status']=='profile_data_excluded'
    assert [r['original_rank'] for r in cert['decisions']]==list(range(1,7))
    assert cert['complete'] and not cert['profile_ranking_complete']


def test_small_profiles_equal_known_only_oracle_without_requerying():
    for size in range(1,5):
        source=events(size)
        for flags in product([False,True,None],repeat=size):
            values=dict(zip([r['event_id'] for r in source],flags))
            for arm in ('poc_priority','poc_filter'):
                actual,provider=run(source,values,arm,slots=2)
                # Full known-only ordering is an independent selection oracle;
                # lazy exhausted suffixes need not be queried or excluded.
                oracle=Planner(slots=2)
                ordered=[r for r in source if values[r['event_id']] is True]
                if arm=='poc_priority':ordered += [r for r in source if values[r['event_id']] is False]
                expected=[r['event_id'] for r in ordered if oracle.consider(r)['attempted']]
                assert [r['event_id'] for r in actual['events']]==expected
                assert len(provider.requested)==len(set(provider.requested))


def test_no_gap_preserves_every_original_certificate_field():
    source=events(4);flags=dict(e0=False,e1=True,e2=True,e3=False)
    old=select_candidates(source,arm='poc_priority',provider=Provider(flags),planner=Planner())
    new,_=run(source,flags)
    assert old['events']==new['events']
    assert all(new['certificate'][k]==v for k,v in old['certificate'].items())
    assert new['certificate']['profile_data_exclusions']==[]


def test_false_fallback_preserved_and_unknown_not_promoted_to_false():
    source=events(4);flags=dict(e0=False,e1=True,e2=None,e3=True)
    ranked,_=run(source,flags)
    filtered,_=run(source,flags,'poc_filter')
    assert [r['event_id'] for r in ranked['events']]==['e1','e3','e0']
    assert [r['event_id'] for r in filtered['events']]==['e1','e3']
    assert ranked['certificate']['decisions'][2]['profile_status']=='unknown'
    assert ranked['certificate']['requested_profiles'][2]['poc_up'] is None


@pytest.mark.parametrize('reason,recoverable',[('missing_raw',True),('not_requested',True),
    ('raw_tape_unavailable',True),('request_budget_paused',True),('quota_paused',True),
    ('provider_error',True),('unclassified_data_gap',False),('ordinary_tape_conflict',None)])
def test_other_structured_unknowns_are_excluded_with_exact_reason(reason,recoverable):
    p=Planner();result=gap.select_with_profile_gaps(events(1),arm='poc_filter',
        provider=lambda _:dict(available=False,poc_up=None,reason=reason,recoverable=recoverable),
        planner=p,planner_factory=lambda _:Planner())
    assert result['events']==[]
    row=result['certificate']['profile_data_exclusions'][0]
    assert (row['reason'],row['recoverable'],row['available'],row['poc_up'])==(reason,recoverable,False,None)
    assert result['certificate']['final_state']==result['certificate']['initial_state']


@pytest.mark.parametrize('reason',sorted(gap.QUALITY_REASONS))
def test_existing_permanent_quality_exception_propagates_unchanged(monkeypatch,reason):
    observed={};original=gap.select_candidates
    def capture(*args,**kwargs):
        try:return original(*args,**kwargs)
        except UnresolvedProfile as e:
            observed['exception']=e;observed['certificate']=deepcopy(e.certificate);raise
    monkeypatch.setattr(gap,'select_candidates',capture)
    with pytest.raises(UnresolvedProfile) as caught:
        gap.select_with_profile_gaps(events(1),arm='poc_filter',planner=Planner(),
            planner_factory=lambda _:Planner(),
            provider=lambda _:dict(available=False,reason=reason,recoverable=False))
    assert caught.value is observed['exception']
    assert caught.value.reason==reason and caught.value.recoverable is False
    assert all(caught.value.certificate[k]==v for k,v in observed['certificate'].items())
    assert caught.value.certificate['profile_data_exclusions_applied'] is False


def test_original_available_hook_restores_full_day_even_after_provisional_gap(monkeypatch):
    monkeypatch.setattr(FiveAxisReplay,'corporate_day',lambda self,day:0.)
    live,day=real_engine(3);x=object.__new__(AccountCandidateHook);x.__dict__.update(live.__dict__)
    source=events(3);x.events={day:deepcopy(source)};x.selection_arm='poc_priority_available'
    x.selection_decisions=[];x.candidate_selector=gap.select_with_profile_gaps
    x.profile_provider=lambda e:dict(available=False,reason='missing_raw',recoverable=True) if e['event_id']=='e0' else dict(
        available=False,reason='ordinary_tape_conflict',recoverable=False)
    x.corporate_day(day)
    c=x.selection_decisions[0]
    assert x.events[day]==source and c['fallback_to_original'] and c['fallback_reason']=='ordinary_tape_conflict'
    assert c['reset_state']==c['initial_state'] and x.cash==1_000_000.
    assert c['certificate']['profile_data_exclusions'][0]['event_id']=='e0'
    assert c['certificate']['profile_data_exclusions'][0]['applied'] is False


def test_reservation_audit_accepts_certificate_after_gap_rebuild(monkeypatch):
    live,day=real_engine(3);p=ReservationPlanner(live,day)
    result=gap.select_with_profile_gaps(events(3),arm='poc_priority',planner=p,
        provider=Provider(dict(e0=True,e1=None,e2=True)))
    cert=result['certificate']
    class Base:
        def corporate_day(self,day):return 0.
    class Audit(AccountReservationAudit,Base):pass
    account=Audit();account.selection_decisions=[dict(date=str(day.date()),arm='poc_priority_available',certificate=cert)]
    account.day_plans={(r['event_id'],'buy'):dict(event_id=r['event_id'],side='buy',
        reserved_cash=next(d['reservation']['allocation'] for d in cert['decisions'] if d['event_id']==r['event_id']))
        for r in result['events']}
    account.corporate_day(day)
    assert account.selection_decisions[0]['actual_reservations_verified'] is True


def test_exhausted_or_unreservable_candidates_never_query_or_claim_exclusion():
    result,provider=run(events(3),dict(e0=True),slots=1)
    assert provider.requested==['e0'] and result['certificate']['profile_data_exclusions']==[]
    source=events(2);source[0]['amount']=1.
    result,provider=run(source,dict(e1=None))
    assert provider.requested==['e1']
    assert result['certificate']['decisions'][0]['profile_status']=='not_needed_unreservable'
    assert [r['event_id'] for r in result['certificate']['profile_data_exclusions']]==['e1']


def test_original_does_not_require_factory_or_provider():
    p=Planner();before=p.snapshot()
    result=gap.select_with_profile_gaps(events(),arm='original',planner=p,provider=None)
    assert result['events']==events() and p.snapshot()==before
    assert result['certificate']['profile_provider_event_ids']==[]


def test_retry_cannot_reuse_reservations_or_mutate_caller():
    p=Planner()
    with pytest.raises(ValueError,match='fresh'):
        gap.select_with_profile_gaps(events(1),arm='poc_priority',planner=p,
            provider=Provider(dict(e0=None)),planner_factory=lambda _:p)
    reused=Planner()
    with pytest.raises(ValueError,match='fresh'):
        gap.select_with_profile_gaps(events(1),arm='poc_priority',planner=p,
            provider=Provider(dict(e0=None)),planner_factory=lambda _:reused)
    def mutating(event):
        p.cash-=1
        return dict(available=False,reason='missing_raw',recoverable=True)
    with pytest.raises(ValueError,match='caller reservation'):
        gap.select_with_profile_gaps(events(1),arm='poc_priority',planner=p,
            provider=mutating,planner_factory=lambda _:Planner())


def test_arbitrary_errors_and_identity_corruption_are_not_silently_skipped():
    def failure(_):raise RuntimeError('Unexpected program failure')
    with pytest.raises(RuntimeError):
        gap.select_with_profile_gaps(events(1),arm='poc_priority',provider=failure,
            planner=Planner(),planner_factory=lambda _:Planner())
    with pytest.raises(ValueError,match='stock differs'):
        gap.select_with_profile_gaps(events(1),arm='poc_priority',
            provider=lambda _:dict(available=False,stock_id='9999'),planner=Planner(),planner_factory=lambda _:Planner())


def test_input_and_provider_values_are_copied_and_future_outcomes_unused():
    source=events(3);original=deepcopy(source);calls=[]
    def provider(event):
        eid=event['event_id'];calls.append(eid);event['priority']=-999
        return dict(available=False,reason='missing_raw',recoverable=True) if eid=='e1' else dict(available=True,poc_up=True)
    a=gap.select_with_profile_gaps(source,arm='poc_priority',provider=provider,planner=Planner(),planner_factory=lambda _:Planner())
    assert source==original and calls==['e0','e1','e2']
    for row in source:row['future_return']=-99999
    b=gap.select_with_profile_gaps(source,arm='poc_priority',provider=provider,planner=Planner(),planner_factory=lambda _:Planner())
    assert a['certificate']==b['certificate']
