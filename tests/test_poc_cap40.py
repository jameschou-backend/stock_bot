from copy import deepcopy
from types import SimpleNamespace
import pandas as pd
import pytest
from skills.poc_cap40 import Cap40, audit_cap40
from skills.partial_risk import CoreSlots
from skills.midpoint_exit_replay import MidpointExitReplay

class PartialRisk(Cap40, MidpointExitReplay, CoreSlots):
    def __init__(self,*args,stop_events,risk_arm,**kwargs):
        super().__init__(*args,**kwargs)
from skills.partial_risk_audit import audit_partial_risk, audit_core_resources
from skills.scenario_exit_replay import ExitSignals
from skills.midpoint_exit_audit import audit_midpoint_exit
from test_cash_allocation_replay import fixture, ENTRY, Corporate
from test_historical_selector_replay import identities
from test_mixed_odd_replay import Odds


def simulate(arm='cap40', future_high=60., delayed=False, hard=False, cap=True, core=False, split=False):
    def mutate(q,days):
        bars = [(51,50),(90,89),(90,74),(future_high,74),(80,75),(80,74),(81,76)] if cap else [(51,50),(60,59),(60,50),(future_high,50),(60,51),(60,50),(60,51)]
        if hard: bars[2]=(60,43)
        if core: bars[4:] = [(50,47),(50,47),(50,47)]
        for n,(h,c) in enumerate(bars):
            h=max(h,c)
            q.loc[q.stock_id.eq('1101')&q.date.eq(days[ENTRY+n]),['open','high','low','close']]=[c,h,c-1,c]
        if split:
            q.loc[q.stock_id.eq('1101')&q.date.ge(days[ENTRY+4]),['open','high','low','close']] /= 2
    days,adjusted,args,kw=fixture(entries=[ENTRY],end=ENTRY+6,mutate=mutate)
    adjusted['1101']=args[0].loc[args[0].stock_id.eq('1101')].set_index('date')['close']*2
    args[4].get_limits=lambda sid:{str(d.date()):dict(lower=10.,upper=60. if cap and d==days[ENTRY] else 200.) for d in days}
    class OddFeed(Odds):
        def get_odd(self,day,sid,market):
            r=super().get_odd(day,sid,market)
            if delayed and str(days[ENTRY+2].date())<=day<=str(days[ENTRY+3].date()):r['odd_shares']=0
            if split and day>=str(days[ENTRY+4].date()):
                r['odd_high']/=2;r['odd_low']/=2
            return r
    events=pd.DataFrame(columns=['stock_id','event_date','ratio'])
    if split:
        stamp=str(days[ENTRY+4].date())
        adjusted.loc[days[ENTRY+4]:,'1101'] *= 2
        actions=Corporate({('1101',stamp):[dict(kind='split',stock_id='1101',action_id='split-test',multiplier=2)]})
        actions.reference_price=lambda sid,day,price:price/2 if day==stamp else price
        args=(*args[:5],actions)
        events=pd.DataFrame([dict(stock_id='1101',event_date=stamp,ratio=.5)])
    if not hasattr(args[5],'reference_price'): args[5].reference_price=lambda sid,day,prior:prior
    features=ExitSignals(adjusted,days); odds=OddFeed(high=54.,low=48.)
    e=PartialRisk(*args,**kw,stop_events=events,risk_arm=arm,ordering='original',position_count=3,
        factor_mask=0,residual_policy='release',identity_report=identities(),liquidity_identity=identities(),
        exit_signals=features,ticks=None,odd_feeds=odds)
    a=e.run();data=SimpleNamespace(days=days,quotes=args[0],features=features,events=events,end=kw['end'])
    audit_core_resources(a,e,args[0])
    audit_midpoint_exit(a,None,odds,{'1101':'TWSE'},args[0],days,e.corporate,args[4])
    return e,a,days,data



def test_cap_reduces_to_fixed_units_next_day_and_keeps_slot():
    e,a,days,data=simulate(future_high=80.)
    rows=[r for r in a['cap40_records'] if r['triggered']]
    assert rows and rows[0]['date']==str(days[ENTRY+2].date())
    r=rows[0]
    assert r['prior_weight']>.4
    assert r['after']['keep']==int(r['prior_nav']/3/r['reference'])
    assert a['cohorts'][0]['exit_date'] is None
    assert e.residual_days[-1]['opening_active']==['1101']
    assert a['cap40_audit']['passed']


def test_original_hard_stop_has_priority():
    _,a,_,_=simulate(hard=True)
    row=next(r for r in a['cap40_records'] if r['original_reason']=='loss12')
    assert row['after'] is None and row['requested_qty']==0
    assert any(t['reason']=='loss12' for t in a['trades'])


def test_decision_ignores_execution_day_high():
    _,a,_,_=simulate(future_high=80.)
    _,b,_,_=simulate(future_high=100.)
    assert a['cap40_records']==b['cap40_records']


@pytest.mark.parametrize('field,value',[('requested_qty',999999),('signal_date','2099-01-01'),('prior_weight',.2)])
def test_cap_auditor_rejects_tampering(field,value):
    _,a,_,data=simulate(future_high=80.)
    r=next(r for r in a['cap40_records'] if r['triggered'])
    r[field]=value
    with pytest.raises(ValueError): audit_cap40(a,data.features.adjusted_close)


def test_unfilled_reduction_keeps_target_and_split_adjusts_units():
    _,a,days,_=simulate(delayed=True,split=True,future_high=80.)
    rows=[r for r in a['cap40_records'] if r['requested_qty']]
    assert len(rows)>=3
    before=next(r for r in rows if r['date']==str(days[ENTRY+3].date()))
    after=next(r for r in rows if r['date']==str(days[ENTRY+4].date()))
    assert after['before']['keep']==before['after']['keep']*2
    assert after['after']['signal_date']==before['after']['signal_date']
    assert a['cohorts'][0]['exit_date'] is None


@pytest.mark.parametrize('weight,want',[(.4,False),(.400001,True),(.399999,False)])
def test_strict_40_percent_boundary(weight,want):
    class Parent:
        def __init__(self):
            self.days=pd.bdate_range('2026-01-01',periods=2)
            self.positions={d:i for i,d in enumerate(self.days)}
            self.holdings={'1101':dict(qty=2000,event_id='e',due_index=63)}
            self.exit_states={'e':dict(trigger_reason=None)}
            self.holding_rows=[dict(date=str(self.days[0].date()),event_id='e',market_value=weight*300000)]
            self.previous_nav=300000
            self.actions=[];self.opening_limit=100000
            self.day_plans={};self.tick_plans=[]
        def corporate_day(self,day):return 0
        def prior(self,day,sid):return 60
        def _plan(self,day,sid,side,eid,*args):
            self.day_plans[(eid,side)]={};self.tick_plans.append({})
    engine=type('Boundary',(Cap40,Parent),{})()
    engine.corporate_day(engine.days[1])
    assert engine.cap_records[0]['triggered'] is want
