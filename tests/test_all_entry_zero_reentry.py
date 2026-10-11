"""A pre-open rejected buy stays rejected after the old holding sells."""
from copy import deepcopy
import pytest

from skills.all_entry_zero_reentry import ZeroBuyAfterSale

from test_poc_gap_execution import engine, execute, tracked, DAY, SID, EID  # noqa: F401


def zero_after_sale(obj):
    plan=obj.day_plans[(EID,'buy')]
    plan.update(planned_qty=0,board_qty=0,odd_qty=0,reserved_cash=0.,sizing_budget=0.,
                rejection='opening_slots_locked')
    obj.tick_plans=[deepcopy(plan)]
    obj.tick_attempts={SID}
    obj.orders=[dict(date=str(DAY.date()),stock_id=SID,side='sell',event_id='old-holding',
                     channel='odd',filled_qty=115)]
    return obj


def test_precommitted_zero_buy_after_sale_is_a_rejection_not_a_second_tape_attempt(engine):
    obj=zero_after_sale(engine)
    obj.__class__=type('PatchedFixture',(ZeroBuyAfterSale,type(obj)),{})
    before=tracked(obj)
    assert execute(obj)==0
    assert obj.orders[-1]['failure']=='opening_slots_locked'
    assert obj.orders[-1]['requested_qty']==obj.orders[-1]['filled_qty']==0
    after=tracked(obj)
    assert {k:v for k,v in after.items() if k!='orders'}=={k:v for k,v in before.items() if k!='orders'}


@pytest.mark.parametrize('change',['positive','signal','duplicate_plan','repeat','no_sale','cash'])
def test_positive_or_fabricated_reentry_is_still_rejected(engine,change):
    obj=zero_after_sale(engine)
    obj.__class__=type('PatchedFixture',(ZeroBuyAfterSale,type(obj)),{})
    plan=obj.day_plans[(EID,'buy')]
    if change=='positive':plan.update(planned_qty=1,odd_qty=1)
    elif change=='signal':plan['signal_date']=str(DAY.date())
    elif change=='duplicate_plan':obj.tick_plans.append(deepcopy(plan))
    elif change=='repeat':execute(obj)
    elif change=='no_sale':obj.orders=[]
    else:plan['reserved_cash']=100.
    with pytest.raises(ValueError,match='unique committed plan'):execute(obj)
