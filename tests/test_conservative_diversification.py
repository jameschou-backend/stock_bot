import pytest
from skills.conservative_diversification import ConservativeDiversification
from skills.slot_reuse_replay import SlotReuseReplay,audit_slots
from skills.scenario_exit_replay import ExitSignals
from test_reservation_replay import multi_stock
from test_cash_allocation_replay import Feeds,ENTRY


def setup():
    days,adjusted,args,kw=multi_stock()
    quotes,companies,calendar,events,_,corp=args
    return days,(quotes,companies,calendar,events,Feeds(quotes),corp),dict(kw,exit_signals=ExitSignals(adjusted,days))


@pytest.mark.parametrize('stress',['control','combined'])
def test_three_positions_matches_complete_parent_account(stress):
    _,args,kw=setup();kw['stress_mode']=stress
    old=SlotReuseReplay(*args,**kw,opening_cash_only=True,lock_unused=True,
        lock_opening_slots=True,lock_failed_slots=True).run()
    assert ConservativeDiversification(*args,**kw).run()==old


def test_five_slots_changes_budget_and_occupancy_together():
    _,args,kw=setup();e=ConservativeDiversification(*args,**kw,position_count=5)
    r=e.run()
    assert r['settings']['slots']==5
    assert len(r['cohorts'])==4
    assert all(p['budget']<=200_000.01 for p in e.resource_plans)
    audit_slots(r,e.resource_plans,e.slot_decisions,opening_cash_only=True,lock_unused=True,
        lock_opening_slots=True,lock_failed_slots=True)


def test_extra_slot_does_not_release_opening_or_unfilled_resources():
    days,args,kw=setup();q,c,cal,events,feeds,corp=args
    # All current signals fail at the limit; their nominal budget remains locked.
    feeds.get_limits=lambda sid:{str(d.date()):dict(upper=50.,lower=.001) for d in days}
    e=ConservativeDiversification(*args,**kw,position_count=5);r=e.run()
    assert not r['trades']
    assert e.resource_plans[-1]['locked_after']==800_000
    assert e.slot_decisions[-1]['attempts_before']==['1101','1102','1103']


@pytest.mark.parametrize('count',[1,7,True,5.0])
def test_unregistered_counts_are_rejected(count):
    _,args,kw=setup()
    with pytest.raises(ValueError):ConservativeDiversification(*args,**kw,position_count=count)
