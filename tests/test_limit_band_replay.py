from copy import deepcopy
import pytest

from skills.strict_tick_inputs import StrictResidualReplay
from skills.limit_band_replay import factory,audit_bands,band_limit
from skills.residual_slot_replay import audit_residual_slots
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture,ENTRY
from test_historical_selector_replay import identities
from test_residual_tick_replay import Ticks


def run(mask,price=50.,**options):
    days,adjusted,args,kwargs=fixture(entries=[ENTRY],end=ENTRY+65,**options)
    engine=factory(mask)(StrictResidualReplay)(*args,**kwargs,residual_policy='release',factor_mask=0,
        identity_report=identities(),liquidity_identity=identities(),
        exit_signals=ExitSignals(adjusted,days),ticks=Ticks(price))
    account=engine.run()
    audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,
        engine.board_decisions,engine.residual_days,args[0])
    audit_bands(account,engine.ticks,engine.markets,args[0],days,engine.corporate)
    return engine,account,args[0],days


def test_fixed_band_buy_sell_and_cost_change_without_extra_day():
    _,control,_,_=run(0)
    _,buy,_,_=run(1)
    _,both,_,days=run(3)
    assert not control['trades']
    assert len(buy['trades'])==1 and len(both['trades'])==2
    assert both['trades'][0]['date']==str(days[ENTRY].date())
    assert both['trades'][0]['reference_price']==51.
    assert both['trades'][1]['reference_price']==49.
    assert both['daily'][-1]['nav']<1_000_000


def test_future_quotes_cannot_change_opening_buy_plan():
    _,before,_,_=run(3)
    def mutate(quotes,days):
        quotes.loc[quotes.stock_id.eq('1101') & quotes.date.ge(days[ENTRY]),'close']=50.8
    _,after,_,_=run(3,mutate=mutate)
    assert before['tick_plans'][0]==after['tick_plans'][0]


def test_mutated_limit_or_volume_cannot_pass_audit():
    engine,account,quotes,days=run(3)
    bad=deepcopy(account);bad['tick_plans'][0]['limit_price']=52.
    with pytest.raises(ValueError,match='fixed pre-tick rule'):
        audit_bands(bad,engine.ticks,engine.markets,quotes,days,engine.corporate)
    bad=deepcopy(account);bad['orders'][0]['eligible_shares']+=1000
    with pytest.raises(ValueError,match='independently reproduce'):
        audit_bands(bad,engine.ticks,engine.markets,quotes,days,engine.corporate)


def test_price_grid_is_rounded_within_limit_band():
    assert band_limit(49.9,'1101','buy',1)==50.8
    assert band_limit(50.1,'1101','sell',2)==49.1
