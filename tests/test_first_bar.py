import numpy as np
import pandas as pd
import pytest
from skills.first_bar import features, episodes, wait_for_breakout, entries, outcomes, with_chips


def fixture():
    days=pd.bdate_range('2021-01-01',periods=190)
    close=pd.DataFrame({'0050':100.,'2313':100.},index=days)
    volume=close*0+1_000_000
    ohlc=dict(open=close-1,high=close+1,low=close-2,close=close.copy(),volume=volume.copy())
    for pos in (140,144,175):
        close.iloc[pos,1]=103;volume.iloc[pos,1]=2_000_000
        for k,v in dict(open=100.,high=104.,low=99.,close=103.,volume=2_000_000).items():ohlc[k].iloc[pos,1]=v
    companies=pd.DataFrame([dict(stock_id='2313',name='test',industry='test',market='TWSE',listed_date='2000-01-01')])
    return close,close.copy(),volume,ohlc,companies


def test_first_bar_uses_prior_volume_and_does_not_label_second_burst_first():
    inputs=fixture();f=features(*inputs);days=inputs[0].index
    assert f['first'].at[days[140],'2313']
    assert not f['first'].at[days[144],'2313']
    assert f['first'].at[days[175],'2313']
    assert f['volume_ratio'].at[days[140],'2313']==2
    assert len(episodes(f,start='2021-01-01'))==2


@pytest.mark.parametrize('broken',['inverted','raw_mismatch','missing_volume','upper_shadow'])
def test_inconsistent_or_weak_candle_does_not_trigger(broken):
    inputs=fixture();c,raw,v,q,companies=inputs;d=c.index[140]
    if broken=='inverted':q['high'].at[d,'2313']=98
    elif broken=='raw_mismatch':q['close'].at[d,'2313']=104
    elif broken=='missing_volume':v.at[d,'2313']=np.nan
    else:q['high'].at[d,'2313']=110
    assert not features(*inputs)['first'].at[d,'2313']


def test_wait_can_be_same_day_later_never_or_unknown_without_using_outcomes():
    inputs=fixture();f=features(*inputs);days=inputs[0].index;ev=episodes(f,start='2021-01-01').iloc[:1]
    assert wait_for_breakout(ev,f).iloc[0].wait_sessions==0
    f['breakout'].loc[days[140]:days[160],'2313']=False
    assert wait_for_breakout(ev,f).iloc[0].wait_state=='not_triggered'
    f['breakout'].at[days[145],'2313']=True
    assert wait_for_breakout(ev,f).iloc[0].wait_sessions==5
    f['breakout_known'].at[days[143],'2313']=False
    assert wait_for_breakout(ev,f).iloc[0].wait_state=='missing_price_before_breakout'


def test_unknown_chip_is_not_negative_and_only_lagged_week_is_used():
    ev=pd.DataFrame([dict(stock_id='2313',signal_date=d) for d in ('2021-06-10','2021-06-11')])
    weekly=pd.DataFrame([dict(stock_id='2313',date=pd.Timestamp('2021-06-03'),change_known=True,
        change_reason='ok',large_pct_delta4=.01,small_pct_delta4=-.01,large_units_delta4=100)])
    r=with_chips(ev,weekly,8)
    assert pd.isna(r.iloc[0].concentration) and bool(r.iloc[1].concentration)
    assert with_chips(ev,weekly,15).concentration.isna().all()


def test_orders_start_next_day_and_concentration_is_not_future_backfilled():
    inputs=fixture();f=features(*inputs);days=inputs[0].index;ev=episodes(f,start='2021-01-01')
    ev['concentration']=pd.Series([True,pd.NA],dtype='boolean')
    ev['available_date']=ev.signal_date;ev['observed_date']='2021-06-01'
    result=entries(ev,wait_for_breakout(ev,f),f,arm='first')
    assert len(result)==1 and result[0]['entry_date']==str(days[141].date())


def test_paired_outcomes_keep_same_endpoint_and_never_breakout_as_cash():
    days=pd.bdate_range('2022-01-03',periods=70)
    close=pd.DataFrame({'2313':np.arange(70)+100.,'0050':100.},index=days)
    ev=pd.DataFrame([dict(event_id='a',stock_id='2313',signal_date=str(days[1].date()),named_case=False)])
    same=pd.DataFrame([dict(event_id='a',wait_state='triggered',wait_signal_date=str(days[1].date()),wait_sessions=0)])
    r=outcomes(ev,same,close,close)
    assert r.paired_difference.eq(0).all()
    later=same.copy();later['wait_signal_date']=str(days[6].date());later['wait_sessions']=5
    delayed=outcomes(ev,later,close,close)
    assert delayed.iloc[0].wait_return==pytest.approx(close.iloc[22,0]/close.iloc[7,0]-1)
    never=same.copy();never['wait_state']='not_triggered';never['wait_signal_date']=None;never['wait_sessions']=None
    assert outcomes(ev,never,close,close).wait_return.eq(0).all()


def test_unknown_wait_is_not_zero_and_bad_price_is_not_a_loss():
    days=pd.bdate_range('2022-01-03',periods=70)
    c=pd.DataFrame({'2313':100.,'0050':100.},index=days)
    ev=pd.DataFrame([dict(event_id='a',stock_id='2313',signal_date=str(days[1].date()),named_case=False)])
    wait=pd.DataFrame([dict(event_id='a',wait_state='missing_price_before_breakout',wait_signal_date=None,wait_sessions=None)])
    r=outcomes(ev,wait,c,c);assert r.wait_return.isna().all() and r.first_return.eq(0).all()
    c.iloc[5,0]=np.nan
    assert outcomes(ev,wait,c,c).first_return.isna().all()


def test_false_start_means_below_close_before_launch_not_below_launch_close():
    days=pd.bdate_range('2022-01-03',periods=70)
    c=pd.DataFrame({'2313':100.,'0050':100.},index=days);c.iloc[1,0]=104;c.iloc[2:7,0]=102
    ev=pd.DataFrame([dict(event_id='a',stock_id='2313',signal_date=str(days[1].date()),named_case=False)])
    wait=pd.DataFrame([dict(event_id='a',wait_state='not_triggered',wait_signal_date=None,wait_sessions=None)])
    assert not outcomes(ev,wait,c,c).iloc[0].false_start5
    c.iloc[3,0]=99
    assert outcomes(ev,wait,c,c).iloc[0].false_start5


def test_future_changes_cannot_change_past_first_bar_events():
    c,raw,v,q,companies=fixture();cut=c.index[150]
    before=episodes(features(c,raw,v,q,companies),start='2021-01-01')
    for frame in (c,raw,v,*q.values()):frame.loc[frame.index>cut]*=1.5
    after=episodes(features(c,raw,v,q,companies),start='2021-01-01')
    pd.testing.assert_frame_equal(before[before.signal_date<=str(cut.date())],after[after.signal_date<=str(cut.date())])
