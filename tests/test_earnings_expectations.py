import numpy as np
import pandas as pd
import pytest
from skills.earnings_expectations import expected_revenue,revenue_daily,quality_daily


def test_forecast_cannot_see_target_or_future_month():
    index=pd.date_range('2020-01-01',periods=40,freq='MS')
    monthly=pd.DataFrame({'1101':np.arange(40)+100.},index=index)
    baseline=expected_revenue(monthly)
    changed=monthly.copy();changed.iloc[25:]*=100
    pd.testing.assert_frame_equal(baseline.iloc[:26],expected_revenue(changed).iloc[:26])


def test_revenue_lag_does_not_activate_at_period_end_or_forward_fill_missing_month():
    index=pd.date_range('2020-01-01',periods=30,freq='MS')
    rows=pd.DataFrame({'stock_id':'1101','trading_date':index,'revenue_current_month':100.})
    rows.loc[29,'revenue_current_month']=np.nan
    days=pd.date_range('2022-05-01','2022-06-20')
    out=revenue_daily(rows,days,['1101'],15)
    assert out.at[pd.Timestamp('2022-06-14'),'1101']==0
    assert pd.isna(out.at[pd.Timestamp('2022-06-15'),'1101'])


def test_financial_uses_year_ago_quarter_and_delayed_availability():
    rows=[]
    for i,day in enumerate(pd.period_range('2020Q1','2021Q1',freq='Q').end_time.normalize()):
        for kind,value in [('Revenue',100),('GrossProfit',20+i),('IncomeAfterTaxes',10)]:
            rows.append(dict(stock_id='1101',date=day,type=kind,value=value))
    days=pd.date_range('2021-07-01','2021-08-01')
    result=quality_daily(pd.DataFrame(rows),days,['1101'],120)
    available=pd.Timestamp('2021-03-31')+pd.Timedelta(days=120)
    assert pd.isna(result.at[available-pd.Timedelta(days=1),'1101'])
    assert result.at[available,'1101']==1


def test_duplicate_financial_fact_is_not_silently_averaged():
    row=dict(stock_id='1101',date='2021-03-31',type='Revenue',value=100)
    with pytest.raises(ValueError,match='duplicates'):
        quality_daily(pd.DataFrame([row,row]),pd.date_range('2021','2022'),['1101'],120)
