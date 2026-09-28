import pandas as pd
import pytest
from skills.midpoint_risk_research import market_states, protective_exit
from scripts.publish_midpoint_risk import event_profits


def test_market_state_prefix_is_unchanged_by_future_crash():
    dates=pd.bdate_range('2024-01-01',periods=100)
    close=pd.Series([100+i*.2 for i in range(100)],index=dates)
    expected=market_states(close.iloc[:85])
    close.iloc[85:]=30
    pd.testing.assert_frame_equal(expected,market_states(close).iloc[:85])


def test_market_recovery_passes_one_slot_before_three():
    values=[100+i*.2 for i in range(80)]+[85]*8+[100,101,102,103,104,105,115,116,117,118,119,120]
    close=pd.Series(values,index=pd.bdate_range('2024-01-01',periods=len(values)))
    result=market_states(close)
    assert result.iloc[80]['slots']==0
    tail=result.iloc[88:]
    assert 1 in set(tail.slots)
    assert 3 in set(tail.slots)
    assert tail.index[tail.slots.eq(1)][0] < tail.index[tail.slots.eq(3)][0]


def test_missing_market_price_cannot_authorize_new_positions():
    close=pd.Series(range(100,200),index=pd.bdate_range('2024-01-01',periods=100),dtype=float)
    close.iloc[-1]=float('nan')
    row=market_states(close).iloc[-1]
    assert row['slots']==0 and row['reason']=='missing'


def test_protection_requires_profit_arm_or_confirmed_relative_weakness():
    context=dict(peak_return=.19,peak_drawdown=-.2,below_ma20_two=False,relative20=.1)
    assert protective_exit(context) is None
    assert protective_exit(dict(context,peak_return=.21))=='profit_trail12'
    assert protective_exit(dict(context,below_ma20_two=True,relative20=-.01))=='weak_ma20'
    assert protective_exit(dict(context,below_ma20_two=True,relative20=.01)) is None
    both=dict(context,peak_return=.3,below_ma20_two=True,relative20=-.01)
    assert protective_exit(both,'trail')=='profit_trail12'
    assert protective_exit(both,'weak')=='weak_ma20'
    assert protective_exit(context,'weak') is None


def test_zero_fractional_payment_is_not_an_unattributed_profit():
    row=dict(kind='fractional_share_payment',cash_change=0,stock_id='2374')
    result=dict(account=dict(corporate_actions=[],cash_ledger=[row]),
                summary=dict(final_holdings=[],final_receivables=[],profit=0))
    assert event_profits(result)=={}
    row['cash_change']=1
    with pytest.raises(ValueError,match='attribution'):
        event_profits(result)
