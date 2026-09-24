import numpy as np
import pandas as pd
import pytest
from skills.broker_persistence import select_branches,validate_interval,persistence


def source():
    # Duplicate price rows must be combined before choosing branches.
    return pd.DataFrame([dict(stock_id='2330',date='2026-09-01',securities_trader_id=b,buy=v,sell=0)
                         for b,v in [('1',10),('1',10),('2',18),('3',16),('4',14),('5',12)]]+
                        [dict(stock_id='2330',date='2026-09-01',securities_trader_id='6',buy=0,sell=80)])


def test_select_unique_branches_and_demand_market_balance():
    x=source(); top,error=select_branches(x,'2330','2026-09-01')
    assert error is None
    assert [r['securities_trader_id'] for r in top]==['1','2','3','4','5']
    x.loc[len(x)-1,'sell']=1
    assert select_branches(x,'2330','2026-09-01')[1]=='raw_market_imbalance'


def fixture():
    days=pd.bdate_range('2026-01-01',periods=22)
    branches=[dict(securities_trader_id=str(b),buy=20.,sell=10.) for b in range(5)]
    frames={str(b):pd.DataFrame(dict(date=days,buy=20.,sell=10.)) for b in range(5)}
    return days,branches,frames


def test_both_windows_and_future_rows_cannot_change_score():
    days,branches,frames=fixture(); signal=days[19]
    for n in (5,20):
        before=persistence(branches,frames,days,signal,n)
        assert before['known'] and before['passed'] and before['score']==pytest.approx(1/3)
        for f in frames.values(): f.loc[f.date>signal,'sell']=999999
        assert persistence(branches,frames,days,signal,n)==before


def test_no_activity_cannot_be_assumed_from_missing_row():
    days,branches,frames=fixture();frames['0']=frames['0'].drop(index=18)
    for n in (5,20):
        assert persistence(branches,frames,days,days[19],n)['reason']=='missing_branch_market_day'


def test_signal_day_revision_is_unknown():
    days,branches,frames=fixture();frames['0'].loc[19,'buy']=21.
    assert persistence(branches,frames,days,days[19],5)['reason']=='signal_day_revision_conflict'


def test_positive_last_day_alone_is_not_persistence():
    days,branches,frames=fixture()
    for f in frames.values(): f.loc[:18,['buy','sell']]=[1.,20.]
    for n in (5,20):
        score=persistence(branches,frames,days,days[19],n)
        assert score['known'] and not score['passed'] and score['score']<0


@pytest.mark.parametrize('mutation',['wrong_stock','duplicate','negative','future'])
def test_invalid_interval_cannot_enter_research(mutation):
    x=pd.DataFrame([dict(date='2026-01-01',stock_id='2330',securities_trader_id='1020',buy_volume=10,sell_volume=1)])
    if mutation=='wrong_stock': x['stock_id']='2317'
    if mutation=='duplicate': x=pd.concat([x,x])
    if mutation=='negative': x['buy_volume']=-1
    if mutation=='future': x['date']='2026-01-03'
    with pytest.raises(ValueError): validate_interval(x,'2330','1020','2026-01-01','2026-01-02')
