from copy import deepcopy
import pytest

from skills.execution_resources import ResourceCapacityReplay, ResourceBenchmark, audit_resources
from skills.five_axis_replay import FiveAxisReplay
from skills.execution_stress import StressBenchmark
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture, ENTRY, Corporate
from test_reservation_replay import multi_stock


@pytest.mark.parametrize('stress', ['control','combined'])
@pytest.mark.parametrize('benchmark', [False,True])
def test_all_off_reproduces_legacy(stress,benchmark):
    days, adjusted, args, kw = fixture(entries=[ENTRY],end=ENTRY+5)
    kw['stress_mode']=stress
    if benchmark:
        expected=StressBenchmark(*args,**kw).run()
        actual=ResourceBenchmark(*args,**kw).run()
    else:
        kw['exit_signals']=ExitSignals(adjusted,days)
        expected=FiveAxisReplay(*args,arm='capacity',**kw).run()
        actual=ResourceCapacityReplay(*args,**kw).run()
    assert actual==expected


@pytest.mark.parametrize('slots', [False,True])
def test_failed_attempts_only_lock_slots_when_requested(slots):
    days, adjusted, args, kw=multi_stock()
    args[4].get_limits=lambda sid:{str(d.date()):dict(upper=50.,lower=.001) for d in days}
    engine=ResourceCapacityReplay(*args,exit_signals=ExitSignals(adjusted,days),lock_slots=slots,**kw)
    account=engine.run()
    if slots:
        assert engine.resource_plans[-1]['failure']=='resource_slots_locked'
        assert len(engine.resource_plans[-1]['occupied_before'])==3
    else:
        assert not engine.resource_plans
        assert not any(o['failure']=='resource_slots_locked' for o in account['orders'])
    assert not account['trades']


def test_unused_budget_stays_locked_after_unfilled_order():
    days, adjusted, args, kw=multi_stock()
    args[4].get_limits=lambda sid:{str(d.date()):dict(upper=50.,lower=.001) for d in days}
    engine=ResourceCapacityReplay(*args,exit_signals=ExitSignals(adjusted,days),lock_unused=True,**kw)
    account=engine.run()
    plans=engine.resource_plans
    assert plans[0]['locked_after']>300_000
    assert plans[1]['available_before']<plans[0]['available_before']
    assert plans[-1]['failure']=='resource_cash_locked'
    audit_resources(account,plans,opening_cash_only=False,lock_slots=False,lock_unused=True)


def test_board_and_odd_share_order_budget_and_slot_only_does_not_change_gap_sizing():
    def mutate(q,days):
        q.loc[q.stock_id.eq('1101') & q.date.ge(days[ENTRY]),['open','high','low','close']]=[60.,61.,59.,60.]
    days, adjusted, args, kw=fixture(entries=[ENTRY],end=ENTRY+2,mutate=mutate)
    kw['exit_signals']=ExitSignals(adjusted,days)
    legacy=FiveAxisReplay(*args,arm='capacity',**kw).run()
    slot=ResourceCapacityReplay(*args,lock_slots=True,**kw)
    assert slot.run()==legacy
    engine=ResourceCapacityReplay(*args,lock_unused=True,**kw)
    account=engine.run()
    audit_resources(account,engine.resource_plans,opening_cash_only=False,lock_slots=False,lock_unused=True)
    plan=engine.resource_plans[0]
    buys=[t for t in account['trades'] if t['side']=='buy']
    assert {t['channel'] for t in buys}=={'board','odd'}
    assert sum(-t['cash_change'] for t in buys)<=plan['budget']
    assert sum(t['qty'] for t in buys)<plan['planned_qty']
    corrupted=deepcopy(engine.resource_plans);corrupted[0]['spent']+=10
    with pytest.raises(ValueError):audit_resources(account,corrupted,opening_cash_only=False,lock_slots=False,lock_unused=True)


@pytest.mark.parametrize('lock_unused',[False,True])
def test_same_day_dividends_cannot_fund_benchmark_until_next_day(lock_unused):
    days, adjusted, args, kw=fixture(end=ENTRY+4)
    ex=str(days[ENTRY+1].date());pay=str(days[ENTRY+2].date())
    args=(*args[:-1],Corporate({('0050',ex):[dict(kind='cash_dividend',cash_per_share=10.,pay_date=pay,action_id='div',stock_id='0050')]}))
    engine=ResourceBenchmark(*args,opening_cash_only=True,lock_unused=lock_unused,**kw)
    account=engine.run()
    audit_resources(account,engine.resource_plans,opening_cash_only=True,lock_slots=False,lock_unused=lock_unused)
    prior=next(r['cash'] for r in account['daily'] if r['date']==ex)
    assert sum(-t['cash_change'] for t in account['trades'] if t['date']==pay and t['side']=='buy')<=prior
    assert any(t['side']=='buy' and t['date']==str(days[ENTRY+3].date()) for t in account['trades'])


@pytest.mark.parametrize('policy',['cash','slots'])
def test_same_day_sales_do_not_replenish_locked_resources(policy):
    from test_cash_allocation_replay import Feeds
    days,adjusted,args,kw=multi_stock()
    quotes,companies,calendar,events,feeds,corp=args
    args=(quotes,companies,calendar,events,Feeds(quotes),corp)
    events[-1].update(entry_date=str(days[ENTRY+2].date()),signal_date=str(days[ENTRY+1].date()))
    adjusted.loc[days[ENTRY+1]:,['1101','1102','1103']]=87.
    kw['exit_signals']=ExitSignals(adjusted,days)
    legacy=ResourceCapacityReplay(*args,**kw).run()
    opts=dict(opening_cash_only=policy=='cash',lock_slots=policy=='slots',lock_unused=False)
    engine=ResourceCapacityReplay(*args,**kw,**opts);account=engine.run()
    sale_day=str(days[ENTRY+2].date())
    assert any(t['side']=='sell' and t['date']==sale_day for t in account['trades'])
    assert any(t['side']=='buy' and t['stock_id']=='1104' for t in legacy['trades'])
    if policy=='slots':
        assert not any(t['stock_id']=='1104' for t in account['trades'])
        assert engine.resource_plans[-1]['failure']=='resource_slots_locked'
    else:
        cash=next(d['cash'] for d in account['daily'] if d['date']==str(days[ENTRY+1].date()))
        buys=sum(-t['cash_change'] for t in account['trades'] if t['side']=='buy' and t['date']==sale_day)
        assert buys<=cash+.01
        assert sum(-t['cash_change'] for t in legacy['trades'] if t['side']=='buy' and t['date']==sale_day)>cash
    audit_resources(account,engine.resource_plans,**opts)


def test_factorial_is_full_and_preparation_quota_survives_restarts(tmp_path):
    from scripts.research_execution_resources import configurations,Budget
    from skills.replay_market_feeds import ReplayDataUnavailable
    cases=list(configurations())
    assert len(cases)==26 and len({n for n,c in cases})==26
    for _ in range(30):Budget(tmp_path).take('finmind')
    with pytest.raises(ReplayDataUnavailable,match='ceiling'):Budget(tmp_path).take('finmind')
    Budget(tmp_path).take('official')
