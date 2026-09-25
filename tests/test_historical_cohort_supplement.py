import pandas as pd
import pytest

from scripts.prepare_historical_cohort_supplement import summarize,FIELDS


def prices():
    return pd.DataFrame([['1507',pd.Timestamp('2022-04-13'),10.,11.,9.,10.5,1000]],columns=FIELDS)


def test_zero_volume_is_preserved_without_inventing_missing_dates():
    frame=prices();frame.loc[0,'volume']=0
    result=summarize(frame,['1507'])
    assert result['stocks'][0]['zero_volume_rows']==1
    assert result['stocks'][0]['positive_volume_rows']==0
    assert result['missing_market_sessions_not_inferred'] is True
    assert result['research_period_rows']==1 and result['warmup_rows']==0


def test_raw_inventory_can_preserve_bad_prices_only_as_explicit_quarantine():
    frame=prices();frame.loc[0,'high']=8.
    result=summarize(frame,['1507'],allow_quarantine=True)
    assert result['quarantined_rows']==1
    assert result['quarantine'][0]['accepted_for_features'] is False
    assert result['quarantine'][0]['high']==8.
    assert frame.loc[0,'high']==8.


@pytest.mark.parametrize('ohlc',[(0,11,9,10.5),(0,8,9,0)])
def test_positive_volume_with_zero_price_is_never_accepted(ohlc):
    frame=prices();frame.loc[0,['open','high','low','close']]=ohlc
    result=summarize(frame,['1507'],allow_quarantine=True)
    assert result['quarantined_rows']==1
    assert 'positive_volume_without_complete_ohlc' in result['quarantine'][0]['reasons']
    with pytest.raises(ValueError):summarize(frame,['1507'])


@pytest.mark.parametrize('problem',['missing_stock','duplicate','future','ohlc','fractional_volume'])
def test_invalid_inventory_is_rejected_before_freezing(problem):
    frame=prices();ids=['1507']
    if problem=='missing_stock':ids.append('2358')
    if problem=='duplicate':frame=pd.concat([frame,frame],ignore_index=True)
    if problem=='future':frame.loc[0,'date']=pd.Timestamp('2026-09-10')
    if problem=='ohlc':frame.loc[0,'high']=8.
    if problem=='fractional_volume':frame['volume']=1.5
    with pytest.raises(ValueError):summarize(frame,ids)
