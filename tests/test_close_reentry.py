from copy import deepcopy
from types import SimpleNamespace
import pandas as pd
import pytest

from skills.close_reentry import CloseReentry, recovered
from skills.close_reentry_audit import audit_reentries
from skills.close_confirmed_audit import audit_close_stops
from skills.scenario_exit_replay import ExitSignals
from skills.high_return_audit import audit_high_return_resources
from test_cash_allocation_replay import fixture, ENTRY
from test_historical_selector_replay import identities
from test_mixed_odd_replay import Odds


def simulate(arm='reclaim',future_high=66.,delayed_odd=False,cycles=False):
    def mutate(q,days):
        bars = [(51,50),(60,59),(60,50),(60,59),(66,65),(future_high,65),(67,66),(68,67)]
        if cycles:
            bars[7:] = [(90,89),(85,75),(80,77),(89,88),(89,88),(90,89),(120,119),(115,100),(106,103),(119,118),(120,119),(121,120)]
        for n,(h,c) in enumerate(bars):
            q.loc[q.stock_id.eq('1101') & q.date.eq(days[ENTRY+n]),['open','high','low','close']] = [c,h,c-1,c]
    days,adjusted,args,kwargs = fixture(entries=[ENTRY],end=ENTRY+(18 if cycles else 7),mutate=mutate)
    adjusted['1101'] = args[0].loc[args[0].stock_id.eq('1101')].set_index('date')['close']*2
    args[4].get_limits = lambda sid:{str(d.date()):dict(lower=40.,upper=150.) for d in days}
    class OddFeed(Odds):
        def get_odd(self,day,sid,market):
            r = super().get_odd(day,sid,market)
            if delayed_odd and str(days[ENTRY+3].date())<=day<=str(days[ENTRY+5].date()):r['odd_shares']=0
            return r
    events = pd.DataFrame(columns=['stock_id','event_date','ratio'])
    features = ExitSignals(adjusted,days)
    e = CloseReentry(*args,**kwargs,stop_events=events,reentry_arm=arm,ordering='original',position_count=3,
        factor_mask=0,residual_policy='release',identity_report=identities(),liquidity_identity=identities(),
        exit_signals=features,ticks=None,odd_feeds=OddFeed(high=54.,low=48.))
    a = e.run()
    data = SimpleNamespace(days=days,quotes=args[0],features=features,events=events,end=kwargs['end'],entries=args[3])
    audit_high_return_resources(a,e.resource_plans,e.slot_decisions,e.board_decisions,e.residual_days,args[0])
    audit_close_stops(a,e.exit_states,data,e)
    if arm!='control':audit_reentries(a,data)
    return e,a,days,data


def test_reclaim_is_t_plus_one_after_full_exit_and_confirmation_adds_one_day():
    e,a,days,_ = simulate()
    assert len(e.reentry_log)==1
    assert e.reentry_log[0]['signal_date']==str(days[ENTRY+4].date())
    buys = [t for t in a['trades'] if t['side']=='buy' and '-reclaim-' in t['event_id']]
    assert buys and {t['date'] for t in buys}=={str(days[ENTRY+5].date())}
    c,b,_,_ = simulate('confirm2')
    assert c.reentry_log[0]['date']==str(days[ENTRY+6].date())
    child = next(c for c in a['cohorts'] if c.get('reentry_parent'))
    assert child['due_index']==ENTRY+5+63
    _,control,_,_ = simulate('control')
    assert not any(c.get('reentry_parent') for c in control['cohorts'])


def test_execution_day_high_cannot_change_reentry_signal_or_sizing():
    e,a,_,_ = simulate()
    f,b,_,_ = simulate(future_high=80.)
    assert e.reentry_log==f.reentry_log
    assert [(t['date'],t['qty']) for t in a['trades'] if t['side']=='buy']==[(t['date'],t['qty']) for t in b['trades'] if t['side']=='buy']


def test_pending_odd_lots_prevent_same_wave_reentry():
    e,a,_,_ = simulate(delayed_odd=True)
    assert not e.reentry_log


def test_repeated_recovery_stops_after_two_reentry_cycles():
    e,a,_,_ = simulate(cycles=True)
    assert [r['attempt'] for r in e.reentry_log]==[1,2]
    assert len([c for c in a['cohorts'] if c.get('reentry_parent')])==2
    assert all(c['exit_date'] for c in a['cohorts'])


def test_missing_recovery_inputs_are_not_permission_to_buy():
    assert recovered(105.,100.,.01,102.)
    assert not recovered(101.,100.,.01,102.)
    assert not recovered(105.,100.,None,102.)
    assert not recovered(105.,100.,-.01,102.)


@pytest.mark.parametrize('field,value',[('signal_date','2099-01-01'),('attempt',3)])
def test_audit_rejects_future_signal_and_excess_attempt(field,value):
    _,a,_,data = simulate()
    bad = deepcopy(a);bad['reentry_log'][0][field]=value
    with pytest.raises(ValueError):audit_reentries(bad,data)
