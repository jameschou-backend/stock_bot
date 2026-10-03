import numpy as np
import pandas as pd
import pytest
from scripts.research_flow_response_20261003 import flow_response_features, opportunity


def sample():
    days=pd.bdate_range('2025-01-01',periods=30)
    c=pd.DataFrame({'0050':100.,'2330':np.arange(100.,130.)},index=days)
    q=c.copy();e=c.notna();v=c*0+1000
    f=pd.DataFrame(dict(date=days,stock_id='2330',foreign=10.,trust=1.))
    s=pd.DataFrame([dict(signal_id='a',stock_id='2330',signal_date=str(days[20].date()))])
    return s,c,q,e,v,f


def feature(args,lag=1):return flow_response_features(*args,lag)[0]


def test_lag_price_window_and_normalized_shares():
    args=sample();r=feature(args)
    assert r['flow_quadrant']=='buy_strong' and r['flow_buy_strong'] is True
    assert r['flow_ratio5']==pytest.approx(.011)
    assert r['flow_end']==str(args[1].index[19].date())
    assert r['price_start']==str(args[1].index[14].date())
    assert feature(args,3)['flow_end']==str(args[1].index[17].date())


def test_future_inputs_do_not_change_feature():
    args=list(sample());before=feature(args)
    for f in (args[1],args[2],args[4]):f.iloc[20:]=9999
    args[3].iloc[20:]=False
    args[5].loc[20:,'foreign']=-999999
    assert feature(args)==before


def test_missing_not_zero_and_sell_strong_separate():
    args=list(sample());args[5].loc[16,'trust']=np.nan
    assert feature(args)['flow_buy_strong'] is None
    args=list(sample());args[5]['foreign']=-20
    assert feature(args)['flow_quadrant']=='sell_strong'
    args[5]['foreign']=-1
    assert feature(args)['flow_quadrant']=='neutral'


def test_invalid_price_volume_identity_and_future_columns_rejected():
    for idx,col,value in ((4,'2330',0),(3,'2330',False),(2,'2330',999)):
        args=list(sample());args[idx].iloc[17,args[idx].columns.get_loc(col)]=value
        assert feature(args)['flow_buy_strong'] is None
    args=list(sample());args[0]['net_return']=100
    with pytest.raises(ValueError,match='T-only'):feature(args)


def test_original_known_opportunity_denominator_keeps_cash():
    rows=[dict(status='closed',keep=True,net_return=.2),dict(status='closed',keep=False,net_return=.1),
          dict(status='closed',keep=None,net_return=-.9)]
    r=opportunity(rows,'keep')
    assert r['known_closed']==2 and r['selected_plus_cash_mean']==.1
    assert r['baseline_mean']==pytest.approx(.15)
