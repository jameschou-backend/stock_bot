from copy import deepcopy

import pandas as pd
import pytest

from skills.execution_resources import audit_resources
from skills.board_only_verified_replay import audit_verified_board_only
from skills.residual_slot_replay import audit_residual_slots
from skills.residual_tick_replay import ResidualTickReplay, ResidualTickBenchmark, audit_tick_plans
from skills.replay_market_feeds import ReplayDataUnavailable
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture, ENTRY
from test_historical_selector_replay import identities
from test_residual_slot_replay import six_stocks


class Ticks:
    def __init__(self, price=49., shares=1_000_000):
        self.price, self.shares, self.calls = price, shares, []

    def get(self, sid, day, market):
        self.calls.append((sid,day,market))
        return pd.DataFrame(dict(time=pd.to_timedelta(['09:02:00','10:00:00']),
            price=[self.price,self.price],shares=[self.shares//2,self.shares//2])), 'fixture-digest'


def run(ticks=None, **options):
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], **options)
    ticks = ticks or Ticks()
    engine = ResidualTickReplay(*args, **kwargs, residual_policy='release', factor_mask=0,
        identity_report=identities(), exit_signals=ExitSignals(adjusted,days), ticks=ticks)
    account = engine.run()
    audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,
        engine.board_decisions,engine.residual_days,args[0])
    audit_tick_plans(account,ticks,engine.markets,args[0],days,engine.corporate)
    return engine,account,days


def test_next_session_limit_keeps_five_slot_account_and_idle_cash():
    engine,account,days = run()
    trade = account['trades'][0]
    assert len(account['trades'])==1 and trade['stock_id']=='1101'
    assert trade['date']==str(days[ENTRY].date())
    assert trade['signal_date']==str(days[ENTRY-1].date())
    assert trade['reference_price']==50. and trade['qty']==3000
    assert account['settings']['slots']==5 and account['settings']['live_qualified'] is False
    assert account['tick_plans'][0]['reserved_cash']==200_000.


def test_current_and_future_prices_cannot_change_entry_plan():
    _,before,days = run()
    def mutate(quotes,days):
        quotes.loc[quotes.stock_id.eq('1101') & quotes.date.ge(days[ENTRY]),'close']=50.8
    _,after,_ = run(mutate=mutate)
    assert before['tick_plans']==after['tick_plans']
    assert before['trades'][0]['reference_price']==after['trades'][0]['reference_price']
    assert before['daily'][-1]['nav']!=after['daily'][-1]['nav']


def test_same_price_does_not_fill_and_buy_is_not_retried_tomorrow():
    ticks=Ticks(50.)
    _,account,_=run(ticks)
    assert not account['trades'] and len(ticks.calls)==2  # one engine and independent audit
    assert len(account['tick_plans'])==1
    assert account['orders'][0]['failure']=='no_trade_through_capacity'


def test_insufficient_post_order_volume_only_partially_fills():
    _,account,_=run(Ticks(shares=150_000))
    assert account['trades'][0]['qty']==1000
    assert account['orders'][0]['failure']=='partial_trade_through_capacity'


def test_corrupt_plan_or_fill_fails_independent_audit():
    engine,account,_=run()
    changed=deepcopy(account)
    changed['tick_plans'][0]['limit_price']=49.
    with pytest.raises(ValueError,match='limit differs'):
        audit_tick_plans(changed,engine.ticks,engine.markets,fixture()[2][0],engine.days,engine.corporate)
    changed=deepcopy(account)
    changed['orders'][0]['filled_qty']+=1000
    with pytest.raises(ValueError,match='tick replay differs'):
        audit_tick_plans(changed,engine.ticks,engine.markets,fixture()[2][0],engine.days,engine.corporate)


def test_missing_ticks_and_conflicting_ticks_block_instead_of_daily_fallback():
    class Missing(Ticks):
        def get(self,*args):
            raise ReplayDataUnavailable('Missing tick source')
    with pytest.raises(ReplayDataUnavailable,match='Missing tick'):
        run(Missing())
    with pytest.raises(ReplayDataUnavailable,match='conflict'):
        run(Ticks(1.))


def test_residual_release_and_sells_keep_all_account_audits():
    days,args,kwargs=six_stocks()
    class BothSides(Ticks):
        def get(self,sid,day,market):
            self.price=49. if day in (str(days[ENTRY].date()),str(days[ENTRY+64].date())) else 51.
            return super().get(sid,day,market)
    engine=ResidualTickReplay(*args,**kwargs,residual_policy='release',ticks=BothSides())
    account=engine.run()
    audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,
        engine.board_decisions,engine.residual_days,args[0])
    audit_tick_plans(account,engine.ticks,engine.markets,args[0],days,engine.corporate)
    assert any(t['stock_id']=='1106' for t in account['trades'])
    assert any(h['qty']%1000 for h in account['holdings'])


def test_benchmark_same_tick_policy_never_reuses_volume_within_day():
    days,adjusted,args,kwargs=fixture()
    ticks=Ticks(99.)
    engine=ResidualTickBenchmark(*args,**kwargs,ticks=ticks)
    account=engine.run()
    audit_resources(account,engine.resource_plans,opening_cash_only=True,lock_slots=False,lock_unused=True)
    audit_verified_board_only(account,engine.board_decisions,engine.resource_plans)
    audit_tick_plans(account,ticks,engine.markets,args[0],days,engine.corporate)
    assert len(account['trades'])==1 and account['trades'][0]['stock_id']=='0050'


def test_sub_lot_attempt_locks_cash_before_sizing_next_candidate():
    days,args,kwargs=six_stocks()
    quotes=args[0]
    quotes.loc[quotes.stock_id.eq('1101'),'close']=500.
    engine=ResidualTickReplay(*args,**kwargs,residual_policy='release',ticks=Ticks())
    # A valid opening state: 650k of existing stock plus 350k cash.
    engine.previous_nav=1_000_000.
    engine.cash=350_000.
    engine.holdings['1106']=dict(qty=13000,event_id='prior-cohort',due_index=10**9)
    engine.marks['1106']=dict(price=50.,date=str(days[ENTRY-1].date()))
    engine.corporate_day(days[ENTRY])
    first,second=engine.tick_plans[:2]
    assert first['stock_id']=='1101' and first['planned_qty']==0
    assert first['reserved_cash']==200_000.
    assert second['reserved_cash']==150_000. and second['planned_qty']==2000
    assert sum(p['reserved_cash'] for p in engine.tick_plans)==350_000.


def test_audit_rejects_coherently_altered_reference_and_calendar():
    engine,account,_=run()
    changed=deepcopy(account)
    changed['tick_plans'][0].update(prior_reference=49.,limit_price=49.)
    with pytest.raises(ValueError,match='dated price source'):
        audit_tick_plans(changed,engine.ticks,engine.markets,fixture()[2][0],engine.days,engine.corporate)
    changed=deepcopy(account)
    changed['tick_plans'][0]['reference_date']=str(engine.days[ENTRY-2].date())
    with pytest.raises(ValueError,match='timing'):
        audit_tick_plans(changed,engine.ticks,engine.markets,fixture()[2][0],engine.days,engine.corporate)
