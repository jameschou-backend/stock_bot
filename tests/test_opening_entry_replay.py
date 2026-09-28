from copy import deepcopy
import pandas as pd
import pytest

from skills.opening_entry_replay import factory,audit_opening,opening_match
from skills.strict_tick_inputs import StrictResidualReplay
from skills.residual_slot_replay import audit_residual_slots
from skills.replay_market_feeds import ReplayDataUnavailable
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture,ENTRY
from test_historical_selector_replay import identities


class Ticks:
    def get(self,sid,day,market):
        return pd.DataFrame(dict(time=pd.to_timedelta(['09:00:04','09:00:04','10:00:00']),
            price=[50.,50.,51.],shares=[100_000,400_000,1_000_000])), 'test-digest'


def run(**options):
    days,adjusted,args,kwargs=fixture(entries=[ENTRY],end=ENTRY+65,**options)
    feeds=args[4]
    feeds.get_limits=lambda sid:{str(day.date()):dict(upper=55.,lower=45.) for day in days}
    engine=factory(StrictResidualReplay)(*args,**kwargs,residual_policy='release',factor_mask=0,
        identity_report=identities(),liquidity_identity=identities(),exit_signals=ExitSignals(adjusted,days),ticks=Ticks())
    account=engine.run()
    audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,engine.residual_days,args[0])
    audit_opening(account,engine.ticks,engine.markets,args[0],days,engine.corporate,feeds=feeds)
    return engine,account,args[0],days


def test_opening_price_not_limit_and_original_exit_account_reconciles():
    _,account,_,days=run()
    buy,sell=account['trades']
    assert buy['reference_price']==50. and buy['limit_price']==55.
    assert buy['date']==str(days[ENTRY].date()) and buy['qty']==3000
    assert sell['reference_price']==50. and sell['order_time']=='09:01:00'
    assert account['settings']['cancellation_latency_verified'] is False
    assert account['daily'][-1]['nav']<1_000_000


def test_future_open_cannot_change_preopen_order():
    _,before,_,_=run()
    # Every future opening price is deliberately inconsistent with tape. The
    # plan must already exist and remain identical when execution rejects it.
    days,adjusted,args,kwargs=fixture(entries=[ENTRY])
    args[0].loc[args[0].date.ge(days[ENTRY]),'open']=51.
    args[4].get_limits=lambda sid:{str(day.date()):dict(upper=55.,lower=45.) for day in days}
    engine=factory(StrictResidualReplay)(*args,**kwargs,residual_policy='release',factor_mask=0,
        identity_report=identities(),liquidity_identity=identities(),exit_signals=ExitSignals(adjusted,days),ticks=Ticks())
    with pytest.raises(ReplayDataUnavailable,match='raw daily open'):engine.run()
    assert engine.tick_plans[0]==before['tick_plans'][0]


def test_later_volume_does_not_fund_opening_and_upper_limit_has_no_fill():
    tape,_=Ticks().get('1101','2021-01-01','twse')
    tape.loc[tape.time.eq(pd.Timedelta('09:00:04')),'shares']=[40_000,50_000]
    assert opening_match(tape,50.,55.,3000,2e6,.01)['filled_qty']==0
    tape.loc[tape.time.eq(pd.Timedelta('09:00:04')),'shares']=[60_000,90_000]
    row=opening_match(tape,50.,55.,3000,2e6,.01)
    assert row['filled_qty']==1000 and row['opening_shares']==150_000
    assert opening_match(tape,50.,50.,3000,2e6,.01)['filled_qty']==0
    with pytest.raises(ReplayDataUnavailable,match='raw daily open'):
        opening_match(tape,51.,55.,3000,2e6,.01)


@pytest.mark.parametrize('target',['plan','fill','cost'])
def test_audit_rejects_mutated_plan_fill_and_cost(target):
    engine,account,quotes,days=run()
    bad=deepcopy(account)
    if target=='plan':bad['tick_plans'][0]['planned_qty']+=1000
    elif target=='fill':bad['orders'][0]['opening_shares']+=1000
    else:bad['trades'][0]['cash_change']+=1
    with pytest.raises(ValueError):
        audit_opening(bad,engine.ticks,engine.markets,quotes,days,engine.corporate,feeds=engine.feeds)
