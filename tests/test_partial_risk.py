from copy import deepcopy
from types import SimpleNamespace
import pandas as pd
import pytest
from skills.partial_risk import PartialRisk
from skills.partial_risk_audit import audit_partial_risk, audit_core_resources
from skills.scenario_exit_replay import ExitSignals
from skills.midpoint_exit_audit import audit_midpoint_exit
from test_cash_allocation_replay import fixture, ENTRY, Corporate
from test_historical_selector_replay import identities
from test_mixed_odd_replay import Odds


def simulate(arm='half15', future_high=60., delayed=False, hard=False, cap=False, core=False, split=False):
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
            if delayed and day==str(days[ENTRY+3].date()):r['odd_shares']=0
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
    features=ExitSignals(adjusted,days); odds=OddFeed(high=54.,low=48.)
    e=PartialRisk(*args,**kw,stop_events=events,risk_arm=arm,ordering='original',position_count=3,
        factor_mask=0,residual_policy='release',identity_report=identities(),liquidity_identity=identities(),
        exit_signals=features,ticks=None,odd_feeds=odds)
    a=e.run();data=SimpleNamespace(days=days,quotes=args[0],features=features,events=events,end=kw['end'])
    audit_core_resources(a,e,args[0]);audit_partial_risk(a,data,e)
    audit_midpoint_exit(a,None,odds,{'1101':'TWSE'},args[0],days,e.corporate,args[4])
    return e,a,days,data


def test_half_is_once_and_core_remains_an_occupied_slot():
    e,a,days,_=simulate()
    buys=sum(t['qty'] for t in a['trades'] if t['side']=='buy')
    sales=[t for t in a['trades'] if t['side']=='sell']
    assert sum(t['qty'] for t in sales)==buys//2
    assert {t['date'] for t in sales}=={str(days[ENTRY+3].date())}
    assert a['cohorts'][0]['exit_date'] is None
    assert e.residual_days[-1]['opening_active']==['1101']
    assert not e.residual_days[-1]['released']
    assert a['holdings'][-1]['qty']<1000


def test_pending_half_does_not_repeatedly_halve_and_exec_day_high_is_not_signal():
    _,a,_,_=simulate(delayed=True)
    _,b,_,_=simulate(delayed=True,future_high=80.)
    assert a['risk_decisions']==b['risk_decisions']
    requests=[r['requested_qty'] for r in a['risk_decisions']]
    assert len(requests)==2 and requests[1]<=999
    assert a['risk_decisions'][0]['keep']==a['risk_decisions'][1]['keep']


def test_hard_loss_overrides_half():
    _,a,_,_=simulate(hard=True)
    assert not a['risk_decisions']
    assert {t['reason'] for t in a['trades'] if t['side']=='sell'}=={'loss12'}
    assert a['cohorts'][0]['exit_date']


def test_core_exit_uses_two_closes_after_half_signal():
    _,a,days,_=simulate(core=True)
    exit_row=next(r for r in a['risk_decisions'] if r['reason']=='core_ma60_two')
    assert exit_row['date']==str(days[ENTRY+5].date())
    assert exit_row['signal_date']==str(days[ENTRY+4].date())


def test_cap_is_prior_nav_sized_and_combination_does_not_double_sell():
    _,a,days,_=simulate(arm='cap40',cap=True,future_high=80.)
    _,b,_,_=simulate(arm='half15_cap40',cap=True,future_high=80.)
    cap=next(r for r in a['risk_decisions'] if r['reason']=='reduce_cap40')
    assert cap['date']==str(days[ENTRY+2].date())
    assert len({(p['date'],p['event_id'],p['side']) for p in b['tick_plans']})==len(b['tick_plans'])


def test_pending_reduction_survives_split_without_selling_the_retained_core():
    _,a,_,_=simulate(delayed=True,split=True)
    before,after=a['risk_decisions'][:2]
    assert after['keep']==before['keep']*2
    assert a['holdings'][-1]['qty']==after['keep']
    assert not a['cohorts'][0]['exit_date']


@pytest.mark.parametrize('field,value',[('requested_qty',999999),('signal_date','2099-01-01'),('keep',0)])
def test_independent_audit_rejects_tampered_instructions(field,value):
    e,a,_,data=simulate();bad=deepcopy(a);bad['risk_decisions'][0][field]=value
    with pytest.raises(ValueError):audit_partial_risk(bad,data,e)
