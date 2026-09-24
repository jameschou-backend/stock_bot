import pytest
from skills.deferred_entry_replay import DeferredEntryReplay
from skills.slot_reuse_replay import SlotReuseReplay,audit_slots
from skills.scenario_exit_replay import ExitSignals
from test_reservation_replay import multi_stock
from test_cash_allocation_replay import ENTRY,Feeds


def setup():
    days,adjusted,args,kw=multi_stock()
    quotes,companies,calendar,events,_,corp=args
    events[-1].update(entry_date=str(days[ENTRY+2].date()),signal_date=str(days[ENTRY+1].date()))
    adjusted.loc[days[ENTRY+1]:,['1101','1102','1103']]=87.
    kw['exit_signals']=ExitSignals(adjusted,days)
    kw['end']=str(days[ENTRY+5].date())
    return days,(quotes,companies,calendar,events,Feeds(quotes),corp),kw


@pytest.mark.parametrize('stress',['control','combined'])
def test_one_session_reproduces_sealed_resource_policy(stress):
    _,args,kw=setup();kw['stress_mode']=stress
    old=SlotReuseReplay(*args,**kw,opening_cash_only=True,lock_unused=True,
        lock_opening_slots=True,lock_failed_slots=True).run()
    assert DeferredEntryReplay(*args,**kw).run()==old


def test_retry_can_use_next_day_slot_never_same_day_and_fills_only_once():
    days,args,kw=setup();engine=DeferredEntryReplay(*args,**kw,validity_sessions=2)
    account=engine.run()
    bought=[x for x in account['trades'] if x['side']=='buy' and x['stock_id']=='1104']
    assert bought and {x['date'] for x in bought}=={str(days[ENTRY+3].date())}
    assert all(x['signal_date']==str(days[ENTRY+1].date()) for x in bought)
    assert len([c for c in account['cohorts'] if c['stock_id']=='1104'])==1
    audit_slots(account,engine.resource_plans,engine.slot_decisions,lock_opening_slots=True,
        lock_failed_slots=True,opening_cash_only=True,lock_unused=True)


def test_action_between_signal_and_retry_cancels_only_retry():
    days,args,kw=setup();engine=DeferredEntryReplay(*args,**kw,validity_sessions=2,
        action_dates=[('1104',days[ENTRY+3])])
    account=engine.run()
    assert not any(x['stock_id']=='1104' for x in account['trades'])
    assert any(x['reason']=='corporate_action_since_signal' for x in engine.retry_decisions)


def test_no_parameter_grid_or_infinite_retry():
    _,args,kw=setup()
    for value in (0,3,True):
        with pytest.raises(ValueError):DeferredEntryReplay(*args,**kw,validity_sessions=value)
