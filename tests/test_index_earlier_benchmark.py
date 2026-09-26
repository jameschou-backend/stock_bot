from copy import deepcopy
import pandas as pd
import pytest
from skills.index_earlier_benchmark import run_benchmark,audit_benchmark
from skills.index_earlier_inputs import EarlierBenchmarkCorporate

def inputs():
    calendar=pd.bdate_range('2015-11-02','2016-02-15').strftime('%Y-%m-%d').tolist()
    days=[d for d in calendar if d>='2016-01-04']
    quotes=pd.DataFrame([dict(date=d,stock_id='0050',open=price,close=price,high=price+1,low=price-1,volume=10000000.)
        for d in calendar for price in [100. if d<'2016-01-11' else 90.]])
    dividend=dict(action_id='0050-cash-20160111',stock_id='0050',date='2016-01-11',kind='cash_dividend',
        cash_per_share=10.,pay_date='2016-01-15',announcement_date=None,source='fixture')
    return dict(calendar=calendar,days=days,benchmark_quotes=quotes,
        benchmark_limits={d:dict(upper=120.,lower=80.) for d in calendar},dividends=[dividend])

@pytest.mark.parametrize('stress',[False,True])
def test_actual_payment_enters_cash_before_reinvestment_only_next_day(stress):
    data=inputs();v=run_benchmark(data,stress);account=v['account']
    days={r['date']:r for r in account['daily']}
    assert days['2016-01-11']['receivable']>0 and days['2016-01-14']['receivable']>0
    assert days['2016-01-15']['receivable']==0
    assert not any(t['date']=='2016-01-15' for t in account['trades'])
    assert any(t['date']=='2016-01-18' for t in account['trades'])
    assert all(t['qty']%1000==0 for t in account['trades'])

def test_missing_announcement_never_changes_prior_planning_price():
    provider=EarlierBenchmarkCorporate(inputs()['dividends'])
    assert provider.reference_price('0050','2016-01-11',100)==100
    with pytest.raises(ValueError):provider.prepare('2330')

@pytest.mark.parametrize('change',['payment','amount','price'])
def test_independent_benchmark_audit_checks_data_binding(change):
    data=inputs();v=run_benchmark(data,False);bad=deepcopy(data)
    if change=='payment':bad['dividends'][0]['pay_date']='2016-01-14'
    elif change=='amount':bad['dividends'][0]['cash_per_share']=11
    else:bad['benchmark_quotes'].loc[bad['benchmark_quotes'].date=='2016-01-04','close']=99
    with pytest.raises(ValueError):audit_benchmark(v,bad)
