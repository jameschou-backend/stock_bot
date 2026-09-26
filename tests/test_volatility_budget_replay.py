from copy import deepcopy

import numpy as np
import pytest

from skills.volatility_budget_replay import VolatilityBudgetReplay, audit_volatility_budget, prior_volatility
from skills.residual_slot_replay import ResidualSlotReplay, audit_residual_slots
from skills.scenario_exit_replay import ExitSignals
from test_residual_slot_replay import six_stocks
from test_cash_allocation_replay import ENTRY


@pytest.mark.parametrize('mask', range(8))
def test_neutral_preserves_every_account_field(mask):
    _,args,kw=six_stocks();kw['factor_mask']=mask
    assert VolatilityBudgetReplay(*args,**kw,volatility_target=None).run()==ResidualSlotReplay(*args,**kw,residual_policy='release').run()


def setup(target=.4, gap=False, missing=False):
    days,args,kw=six_stocks(next_offset=64)
    adjusted=kw['exit_signals'].adjusted_close.copy()
    adjusted.loc[:days[ENTRY-1],'1101']=100*np.cumprod(1+np.resize([.09,-.08],ENTRY))
    if missing:adjusted.at[days[ENTRY-3],'1101']=np.nan
    kw['exit_signals']=ExitSignals(adjusted,days)
    if gap:
        args[0].loc[args[0].date.eq(days[ENTRY]) & args[0].stock_id.eq('1101'),['open','close','high','low']]*=1.5
    engine=VolatilityBudgetReplay(*args,**kw,volatility_target=target)
    account=engine.run()
    audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,engine.residual_days,args[0])
    audit_volatility_budget(account,engine.volatility_decisions,engine.residual_days,adjusted,days,target)
    return days,args,kw,engine,account


@pytest.mark.parametrize('gap', [False, True])
def test_high_volatility_caps_cost_inclusive_spend_even_after_gap(gap):
    _,_,_,engine,_=setup(gap=gap)
    row=next(r for r in engine.volatility_decisions if r['stock_id']=='1101')
    assert row['budget']<row['equal_budget']/2
    assert 0<=row['spent']<=row['budget']
    assert (row['spent']==0) if gap else (row['spent']>0)
    plan=next(r for r in engine.resource_plans if r['stock_id']=='1101')
    assert plan['locked_after']==pytest.approx(plan['budget']-row['spent'])


def test_missing_volatility_blocks_fill_but_does_not_recycle_same_day_budget():
    _,_,_,engine,account=setup(missing=True)
    assert not any(t['stock_id']=='1101' and t['side']=='buy' for t in account['trades'])
    row=next(r for r in engine.volatility_decisions if r['stock_id']=='1101')
    assert row['reason']=='missing_prior_volatility'
    plan=next(r for r in engine.resource_plans if r['stock_id']=='1101')
    assert plan['locked_after']==plan['budget'] and plan['budget']>0


def test_future_close_changes_cannot_change_earlier_sizing():
    days,_,kw,engine,_=setup()
    close=kw['exit_signals'].adjusted_close.copy()
    close.loc[days[ENTRY]:]*=7
    future=prior_volatility(close,days)
    assert future.loc[:days[ENTRY]].equals(engine.entry_volatility.loc[:days[ENTRY]])
    bad=deepcopy(engine.volatility_decisions);bad[0]['budget']+=100
    with pytest.raises(ValueError,match='did not reconstruct'):
        audit_volatility_budget({'trades':engine.trades},bad,engine.residual_days,
                                kw['exit_signals'].adjusted_close,days,.4)
