from copy import deepcopy
import pytest

from skills.poc_split_aggregation import ARMS, SUM_FIELDS, combine, audit_combination, capacity_check


def sleeve(navs):
    daily=[]
    for i, nav in enumerate(navs):
        r={k:0 for k in SUM_FIELDS}
        r.update(date=f'2024-01-0{i+2}', cash=100, market_value=nav-100, nav=nav)
        daily.append(r)
    return dict(settings=dict(initial_cash=500000,commission=.001425,slippage=.0045), daily=daily,
        cash_ledger=[dict(date='2024-01-02',kind='initial_deposit',cash_change=500000)],
        trades=[],orders=[],holdings=[],cohorts=[],corporate_actions=[],receivables=[])


def test_combined_drawdown_is_not_the_average_of_sleeve_drawdowns():
    accounts=dict(zip(ARMS,[sleeve([600000,400000,600000]),sleeve([400000,600000,400000])]))
    result=combine(accounts,'baseline')
    assert [r['nav'] for r in result['daily']]==[1000000]*3
    assert [r['drawdown'] for r in result['daily']]==[0]*3
    assert result['daily'][-1]['sleeve_nav'][ARMS[0]]==600000
    assert result['daily'][-1]['sleeve_nav'][ARMS[1]]==400000
    # The sleeves drift to 60/40; no transfer restores 50/50.
    result['daily'][1]['drawdown']=-.2
    with pytest.raises(ValueError,match='drawdown'): audit_combination(result,accounts)


def test_missing_sleeve_dates_are_not_filled_or_truncated():
    accounts=dict(zip(ARMS,[sleeve([500000]*3),sleeve([500000]*2)]))
    with pytest.raises(ValueError,match='calendars'): combine(accounts,'baseline')


def test_same_security_is_separate_by_sleeve_and_capacity_is_not_doubled():
    accounts={a:sleeve([500000]*3) for a in ARMS}
    for a in ARMS:
        accounts[a]['orders']=[dict(date='2024-01-02',stock_id='2330',channel='odd',side='buy',filled_qty=8,daily_capacity_qty=10)]
        accounts[a]['cohorts']=[dict(event_id='shared',entry_date='2024-01-02')]
    before=deepcopy(accounts)
    result=combine(accounts,'baseline')
    assert accounts==before
    assert len({r['event_id'] for r in result['cohorts']})==2
    assert result['shared_capacity']['capacity_collision_count']==1
    assert result['shared_capacity']['collisions'][0]['excess_qty']==6
    assert result['shared_capacity']['joint_capacity_pass'] is False
    assert result['aggregation_audit']['passed'] is True
    accounts[ARMS[0]]['orders'][0]['daily_capacity_qty']=20
    with pytest.raises(ValueError,match='disagree'): capacity_check(accounts)
