import copy
import pytest
from skills.contingent_execution import Plan, ConfirmationGate, OPEN, CUTOFF


def plan(oid, sid, side, qty=1000, channel='board', budget=None):
    return Plan(oid, sid, side, channel, qty, 1000,
                (qty*1000+10000 if side == 'buy' else 0) if budget is None else budget, '2026-09-11')


def gate(plans=None, cash=0):
    return ConfirmationGate('2026-09-14', plans or [plan('s','2330','sell'),plan('b','2317','buy')],
                            {'2330':1000}, cash, slots=1)


def test_sell_confirmation_alone_cannot_fund_buy_and_credit_is_not_reused():
    g=gate(cash=20000)
    assert g.submit_ready(OPEN)==['s']
    g.confirm_fill('f','s',OPEN+1,1000,1000,990000)
    assert g.submit_ready(OPEN+2)==[]
    g.confirm_available('c','f',OPEN+3,990000)
    assert g.submit_ready(OPEN+3)==['b']
    assert g.available==0
    with pytest.raises(ValueError): g.confirm_available('c2','f',OPEN+4,1)
    with pytest.raises(ValueError): g.confirm_fill('too-early','b',OPEN+3,1000,1000,1005000)
    g.confirm_fill('bf','b',OPEN+4,1000,1000,1005000)
    assert g.holdings=={'2317':1000} and g.available==5000


def test_one_odd_share_blocks_slot_until_confirmed_sold():
    plans=[plan('s','2330','sell'),plan('o','2330','sell',1,'odd'),plan('b','2317','buy')]
    g=ConfirmationGate('2026-09-14',plans,{'2330':1001},2000000,1)
    assert g.submit_ready(OPEN)==['s','o']
    g.confirm_fill('f','s',OPEN+1,1000,1000,990000)
    assert g.submit_ready(OPEN+2)==[]
    g.confirm_fill('of','o',OPEN+3,1,1000,900)
    assert g.submit_ready(OPEN+3)==['b']


def test_head_candidate_waits_no_hindsight_substitution():
    g=ConfirmationGate('2026-09-14',[plan('b1','2330','buy'),plan('b2','2317','buy',1,'odd')],{},50000,3)
    assert g.submit_ready(OPEN)==[]


def test_board_and_odd_orders_share_cash_and_stock_slot():
    plans=[plan('b','2330','buy'),plan('o','2330','buy',20,'odd')]
    g=ConfirmationGate('2026-09-14',plans,{},1040000,1)
    assert g.submit_ready(OPEN)==['b','o'] and g.available==0
    g.confirm_fill('f','b',OPEN+1,1000,1000,1005000)
    g.confirm_fill('fo','o',OPEN+2,20,1000,22000)
    assert g.holdings=={'2330':1020} and g.available==13000


def test_partial_fill_does_not_release_unspent_reservation():
    g=ConfirmationGate('2026-09-14',[plan('b','2330','buy',2000)],{},2010000)
    g.submit_ready(OPEN)
    g.confirm_fill('f','b',OPEN+1,1000,1000,1005000)
    assert g.available==0 and g.orders['b']['reserved']==1005000
    g.close()
    assert g.available==1005000
    with pytest.raises(ValueError): g.submit_ready(CUTOFF)


@pytest.mark.parametrize('qty,price,cash,at',[(1001,1000,1000000,OPEN+1),
    (1000,999,990000,OPEN+1),(1000,1000,1000001,OPEN+1),(1000,1000,990000,OPEN),
    (1000,1000,990000,CUTOFF)])
def test_invalid_confirmations_are_atomic(qty,price,cash,at):
    g=gate();g.submit_ready(OPEN);before=copy.deepcopy(g.__dict__)
    with pytest.raises(ValueError):g.confirm_fill('f','s',at,qty,price,cash)
    assert g.__dict__==before


def test_credit_requires_known_sale_and_monotonic_clock():
    g=gate();g.submit_ready(OPEN)
    with pytest.raises(ValueError):g.confirm_available('c','missing',OPEN+1,100)
    g.confirm_fill('f','s',OPEN+2,1000,1000,990000)
    with pytest.raises(ValueError):g.confirm_available('c','f',OPEN+1,100)


def test_plan_does_not_change_with_later_prices_and_rejects_same_day_signal():
    p=plan('b','2330','buy')
    for price in (990,1000):
        g=ConfirmationGate('2026-09-14',[p],{},1010000)
        g.submit_ready(OPEN);g.confirm_fill('f','b',OPEN+1,1000,price,price*1000+5000)
        assert g.orders['b']['plan']==p
    with pytest.raises(ValueError):
        ConfirmationGate('2026-09-11',[p],{},1010000)
