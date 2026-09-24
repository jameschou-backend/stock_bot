from copy import deepcopy
import pytest

from skills.slot_reuse_replay import SlotReuseReplay, audit_slots
from skills.execution_resources import ResourceCapacityReplay
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture, ENTRY, Feeds
from test_reservation_replay import multi_stock


def run(args, kw, r=False, f=False, cash=False):
    opts=dict(lock_opening_slots=r, lock_failed_slots=f, opening_cash_only=cash, lock_unused=cash)
    engine=SlotReuseReplay(*args, **kw, **opts)
    account=engine.run()
    audit_slots(account,engine.resource_plans,engine.slot_decisions,**opts)
    return engine,account


@pytest.mark.parametrize('cash',[False,True])
@pytest.mark.parametrize('locked',[False,True])
@pytest.mark.parametrize('stress',['control','combined'])
def test_two_endpoints_match_full_parent_account(cash,locked,stress):
    days, adjusted, args, kw=fixture(entries=[ENTRY],end=ENTRY+5)
    kw.update(stress_mode=stress,exit_signals=ExitSignals(adjusted,days))
    old=ResourceCapacityReplay(*args,opening_cash_only=cash,lock_unused=cash,lock_slots=locked,**kw).run()
    _,account=run(args,kw,locked,locked,cash)
    assert account==old


@pytest.mark.parametrize('r',[False,True])
@pytest.mark.parametrize('f',[False,True])
def test_unfilled_attempt_release_is_independent_of_opening_slot_lock(r,f):
    days,adjusted,args,kw=multi_stock()
    quotes,companies,calendar,events,_,corp=args
    feeds=Feeds(quotes)
    feeds.get_limits=lambda sid:{str(d.date()):dict(upper=50. if sid!='1104' else 100.,lower=.001) for d in days}
    args=(quotes,companies,calendar,events,feeds,corp);kw['exit_signals']=ExitSignals(adjusted,days)
    engine,account=run(args,kw,r,f)
    assert bool(account['trades']) is (not f)
    if f:
        assert engine.slot_decisions[-1]['blocker_categories']==['unfilled_attempt_not_released']
        assert not engine.slot_decisions[-1]['attempted']
    else:
        assert {t['stock_id'] for t in account['trades']}=={'1104'}
    # Entire board/odd attempt occupies exactly one name, including zero fills.
    assert engine.slot_decisions[-1]['attempts_before']==['1101','1102','1103']


@pytest.mark.parametrize('r',[False,True])
@pytest.mark.parametrize('f',[False,True])
def test_sale_releases_slot_only_when_opening_lock_is_off(r,f):
    days,adjusted,args,kw=multi_stock()
    quotes,companies,calendar,events,_,corp=args
    events[-1].update(entry_date=str(days[ENTRY+2].date()),signal_date=str(days[ENTRY+1].date()))
    adjusted.loc[days[ENTRY+1]:,['1101','1102','1103']]=87.
    args=(quotes,companies,calendar,events,Feeds(quotes),corp);kw['exit_signals']=ExitSignals(adjusted,days)
    engine,account=run(args,kw,r,f)
    assert any(t['stock_id']=='1104' for t in account['trades']) is (not r)
    decision=engine.slot_decisions[-1]
    assert decision['opening_members']==['1101','1102','1103'] and decision['held_before']==[]
    if r:assert decision['blocker_categories']==['opening_slot_not_released']
    broken=deepcopy(engine.slot_decisions);broken[-1]['opening_members']=[]
    with pytest.raises(ValueError,match='reconstruction'):
        audit_slots(account,engine.resource_plans,broken,lock_opening_slots=r,lock_failed_slots=f,opening_cash_only=False,lock_unused=False)


def test_partial_buy_still_occupies_and_both_channels_are_one_slot():
    days,adjusted,args,kw=multi_stock()
    quotes,companies,calendar,events,_,corp=args
    # Prior liquidity qualifies. Current-day volume allows only two board lots.
    quotes.loc[quotes.date.eq(days[ENTRY]),'volume']=200_000
    args=(quotes,companies,calendar,events,Feeds(quotes,odd_volume=0),corp)
    kw['exit_signals']=ExitSignals(adjusted,days)
    engine,account=run(args,kw,False,False)
    assert {t['stock_id'] for t in account['trades']}=={'1101','1102','1103'}
    assert all(t['qty']==2000 for t in account['trades'])
    assert account['orders'][-1]['stock_id']=='1104' and account['orders'][-1]['failure']=='slots_full'
    assert len(engine.slot_decisions)==3
    assert engine.slot_decisions[-1]['held_before']==['1101','1102']


@pytest.mark.parametrize('rights',[False,True])
def test_partial_exit_or_undelivered_stock_rights_do_not_release_slot(rights):
    from test_cash_allocation_replay import Corporate
    days,adjusted,args,kw=multi_stock()
    quotes,companies,calendar,events,_,_=args
    events[-1].update(entry_date=str(days[ENTRY+2].date()),signal_date=str(days[ENTRY+1].date()))
    adjusted.loc[days[ENTRY+1]:,'1101']=87.
    corporate=Corporate()
    if rights:
        ex=str(days[ENTRY+1].date());pay=str(days[ENTRY+5].date())
        corporate=Corporate({('1101',ex):[dict(kind='stock_dividend',shares_per_share=.1,
            pay_date=pay,fractional_cash_per_share=0,action_id='rights',stock_id='1101')]})
    else:
        quotes.loc[quotes.date.eq(days[ENTRY+2]),'volume']=100_000
    args=(quotes,companies,calendar,events,Feeds(quotes),corporate)
    kw.update(exit_signals=ExitSignals(adjusted,days))
    engine,account=run(args,kw)
    assert any(t['side']=='sell' for t in account['trades'])
    assert not any(t['stock_id']=='1104' for t in account['trades'])
    assert account['orders'][-1]['failure']=='slots_full'
    if rights:
        assert not any(h['stock_id']=='1101' and h['date']==str(days[ENTRY+2].date()) for h in account['holdings'])
        assert account['daily'][-1]['receivable']>0
    else:
        assert any(h['date']==str(days[ENTRY+2].date()) and h['qty']>0 for h in account['holdings'])


def test_preregistered_matrix_has_twenty_unique_cases():
    from scripts.research_slot_reuse import configurations
    rows=list(configurations())
    assert len(rows)==len({name for name,_ in rows})==20
    assert sum(c['benchmark'] for _,c in rows)==4
    assert {(c['lock_opening_slots'],c['lock_failed_slots']) for _,c in rows if not c['benchmark']}=={(a,b) for a in (False,True) for b in (False,True)}
