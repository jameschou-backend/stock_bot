from copy import deepcopy
import pandas as pd
import pytest
from skills.mixed_odd_replay import factory,odd_match
from skills.mixed_odd_audit import audit_mixed_execution,audit_mixed_resources
from skills.strict_tick_inputs import StrictResidualReplay
from skills.scenario_exit_replay import ExitSignals
from skills.replay_market_feeds import ReplayDataUnavailable
from test_cash_allocation_replay import fixture,ENTRY
from test_historical_selector_replay import identities
from test_opening_entry_replay import Ticks


class Odds:
    def __init__(self,high=51.,low=49.,volume=100_000):self.high,self.low,self.volume=high,low,volume
    def get_odd(self,day,sid,market):return dict(odd_high=self.high,odd_low=self.low,odd_shares=self.volume)


def run(high_price=False,odds=None):
    def mutate(q,days):
        if high_price:
            for col in ('open','close','high','low'):q.loc[q.stock_id.eq('1101'),col]*=10
    days,adjusted,args,kwargs=fixture(entries=[ENTRY],end=ENTRY+65,mutate=mutate)
    upper,lower=(550.,450.) if high_price else (55.,45.)
    args[4].get_limits=lambda sid:{str(d.date()):dict(upper=upper,lower=lower) for d in days}
    engine=factory(StrictResidualReplay)(*args,**kwargs,residual_policy='release',factor_mask=0,
        identity_report=identities(),liquidity_identity=identities(),exit_signals=ExitSignals(adjusted,days),
        ticks=Ticks(),odd_feeds=odds or Odds(high=510.,low=490.) if high_price else odds or Odds())
    account=engine.run()
    audit_mixed_resources(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,engine.residual_days,args[0])
    audit_mixed_execution(account,engine.ticks,engine.odd_feeds,engine.markets,args[0],days,engine.corporate,engine.feeds)
    return engine,account,args[0],days


def test_buy_expensive_stock_below_one_lot_and_sell_all_shares():
    _,a,_,days=run(high_price=True)
    buys=[t for t in a['trades'] if t['side']=='buy'];sells=[t for t in a['trades'] if t['side']=='sell']
    assert len(buys)==len(sells)==1
    assert buys[0]['channel']=='odd' and 0<buys[0]['qty']<1000
    assert buys[0]['reference_price']==510 and sells[0]['reference_price']==490
    assert buys[0]['qty']==sells[0]['qty'] and buys[0]['date']==str(days[ENTRY].date())
    assert a['settings']['odd_tick_verified'] is False


def test_board_and_remainder_share_one_budget_with_separate_fees():
    _,a,_,_=run()
    buys=[t for t in a['trades'] if t['side']=='buy']
    assert {t['channel'] for t in buys}=={'board','odd'}
    assert sum(-t['cash_change'] for t in buys)<=a['tick_plans'][0]['sizing_budget']
    assert all(t['commission']>=20 for t in buys)
    assert all(t['qty']<1000 for t in a['trades'] if t['channel']=='odd')


def test_current_odd_prices_do_not_change_frozen_quantity():
    _,a,_,_=run(odds=Odds(high=51))
    _,b,_,_=run(odds=Odds(high=54))
    assert a['tick_plans'][0]==b['tick_plans'][0]
    assert a['trades'][1]['cash_change']!=b['trades'][1]['cash_change']


def test_own_odd_volume_and_daily_boundary():
    row=dict(odd_high=51.,odd_low=49.,odd_shares=250)
    assert odd_match(row,'buy',55,999,.01)['filled_qty']==2
    assert odd_match(row,'buy',51,999,.01)['filled_qty']==0
    with pytest.raises(ReplayDataUnavailable):odd_match(None,'buy',55,999,.01)
    with pytest.raises(ValueError):odd_match(row,'buy',55,1000,.01)


def test_mutated_price_or_mislabeling_rejected():
    e,a,q,days=run()
    bad=deepcopy(a);bad['trades'][1]['reference_price']=50.
    with pytest.raises(ValueError):audit_mixed_execution(bad,e.ticks,e.odd_feeds,e.markets,q,days,e.corporate,e.feeds)
    bad=deepcopy(a);bad['settings']['odd_tick_verified']=True
    with pytest.raises(ValueError):audit_mixed_execution(bad,e.ticks,e.odd_feeds,e.markets,q,days,e.corporate,e.feeds)


def test_explicit_official_zero_volume_is_known_no_fill():
    row=odd_match(dict(odd_shares=0,odd_high=None,odd_low=None),'buy',55,123,.01)
    assert row['filled_qty']==0 and row['failure']=='official_odd_zero_volume'
