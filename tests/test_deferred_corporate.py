from types import SimpleNamespace
from copy import deepcopy
import pytest
from skills.deferred_corporate import DeferredCorporatePreparation
from test_cash_allocation_replay import Corporate


class Deferred(DeferredCorporatePreparation, Corporate):
    pass


def test_only_known_slot_rejection_can_defer_and_next_eligible_order_loads():
    p=dict(date='2024-01-03',stock_id='1234',side='buy',rejection='opening_slots_locked',planned_qty=0,reserved_cash=0)
    c=Deferred();c.preparation_engine=SimpleNamespace(holdings={},receivables=[],day_plans={'e':p})
    c.prepare('1234');assert c.prepared==[]
    assert c.deferred_preparations=={('2024-01-03','1234')}
    p.update(rejection=None,planned_qty=1,reserved_cash=100)
    c.prepare('1234');assert c.prepared==['1234']


@pytest.mark.parametrize('protected', ['holdings','receivables','unknown','missing_plan'])
def test_required_or_unknown_corporate_data_never_skipped(protected):
    c=Deferred();p=dict(date='2024-01-03',stock_id='1234',side='buy',rejection='opening_slots_locked',planned_qty=0,reserved_cash=0)
    e=SimpleNamespace(holdings={},receivables=[],day_plans={'e':p});c.preparation_engine=e
    if protected=='holdings':e.holdings={'1234':{'qty':1}}
    elif protected=='receivables':e.receivables=[{'stock_id':'1234','qty':1}]
    elif protected=='unknown':p['rejection']='unknown'
    else:e.day_plans={}
    c.prepare('1234');assert c.prepared==['1234']


def test_full_account_identical_when_unfilled_orders_lock_later_candidate():
    from test_reservation_replay import multi_stock
    from test_allocation_2019 import Replay
    from test_historical_selector_replay import identities
    from test_midpoint_replay import NoTicks
    from test_mixed_odd_replay import Odds
    from skills.scenario_exit_replay import ExitSignals
    accounts=[];calls=[]
    for cls in (Corporate, Deferred):
        days, adjusted, args, kwargs=multi_stock()
        # First three cannot fill; the fourth has no pre-open reservation.
        q=args[0].copy();q.loc[q.stock_id.ne('0050'), ['open','high','low','close']]=50.
        corp=cls();args=(q,*args[1:5],corp)
        args[4].get_limits=lambda sid:{str(d.date()):dict(upper=50.,lower=45.) for d in days}
        identity=identities(('1101','1102','1103','1104'))
        e=Replay(*args,**kwargs,ordering='original',position_count=3,candidate_arm='original',
            factor_mask=0,residual_policy='release',identity_report=identity,liquidity_identity=identity,
            exit_signals=ExitSignals(adjusted,days),ticks=NoTicks(),odd_feeds=Odds(high=50,low=50))
        if cls is Deferred:corp.preparation_engine=e
        accounts.append(e.run());calls.append(corp.prepared)
    assert accounts[0]==accounts[1]
    assert len(calls[1])<len(calls[0])
