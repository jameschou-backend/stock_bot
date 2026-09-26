from copy import deepcopy
import pandas as pd
import pytest
from skills.etf_limit_audit import bounds,reconcile
from skills.index_continuous_inputs import combine,ContinuousCorporate
from skills.index_continuous_benchmark import run_benchmark,audit_benchmark
from tests.test_index_earlier_benchmark import inputs

@pytest.mark.parametrize('reference,lower,upper',[(176,158.4,193.6),(146.2,131.6,160.8),(93.5,84.15,102.85),(49.9,44.91,54.85)])
def test_etf_ticks_use_five_cents_above_fifty(reference,lower,upper):
    assert bounds(reference)==dict(lower=lower,upper=upper)

@pytest.mark.parametrize('reference',[0,-1,float('nan'),float('inf')])
def test_invalid_references_do_not_generate_limits(reference):
    with pytest.raises(ValueError):bounds(reference)

def test_conflicting_provider_bounds_remain_visible():
    q={'2025-04-07':dict(open=158.4,close=158.4,low=158.4,high=158.4,volume=1000)}
    r={'2025-04-07':dict(reference_price=176,limit_up=193.5,limit_down=158.5)}
    before=deepcopy(r);limits,audit=reconcile(q,r,list(q))
    assert r==before and limits['2025-04-07']['lower']==158.4
    assert len(audit['provider_ohlc_conflicts'])==1 and not audit['strict_data_ready']
    q['2025-04-07']['low']=158.35
    with pytest.raises(ValueError):reconcile(q,r,list(q))

def test_overlap_cannot_silently_replace_raw_price():
    assert combine({'a':{'close':1}},{'a':{'close':1},'b':{'close':2}},['close'])['b']['close']==2
    with pytest.raises(ValueError):combine({'a':{'close':1}},{'a':{'close':2}},['close'])

def test_only_explicit_split_changes_reference_units():
    c=ContinuousCorporate([])
    assert c.reference_price('0050','2025-06-18',200)==50
    assert c.reference_price('0050','2016-01-11',100)==100

@pytest.mark.parametrize('stress',[False,True])
def test_continuous_cash_rights_and_missing_source_are_audited(stress):
    data=inputs();data['actions']=data.pop('dividends');v=run_benchmark(data,stress)
    daily={r['date']:r for r in v['account']['daily']}
    assert daily['2016-01-14']['receivable']>0 and daily['2016-01-15']['receivable']==0
    assert not any(t['date']=='2016-01-15' for t in v['account']['trades'])
    bad=deepcopy(data);bad['actions'][0]['pay_date']='2016-01-14'
    with pytest.raises(ValueError):audit_benchmark(v,bad)

def test_split_preserves_account_value_and_converts_held_units():
    from skills.replay_corporate_actions import SPLIT_0050
    calendar=pd.bdate_range('2025-04-01','2025-06-20').strftime('%Y-%m-%d').tolist()
    days=[d for d in calendar if d>='2025-06-02']
    quotes=pd.DataFrame([dict(date=d,stock_id='0050',open=p,close=p,high=p+1,low=p-1,volume=10000000.)
        for d in calendar for p in [200. if d<'2025-06-18' else 50.]])
    data=dict(calendar=calendar,days=days,benchmark_quotes=quotes,actions=[deepcopy(SPLIT_0050)],
        benchmark_limits={d:dict(upper=220.,lower=40.) for d in calendar})
    v=run_benchmark(data,False);daily={r['date']:r for r in v['account']['daily']}
    action=v['account']['corporate_actions'][0]
    assert action['qty_after']==action['entitled_qty']*4
    # The lower post-split lot price allows investing previously idle cash.
    # NAV changes only by those additional execution costs, not by the split.
    day_cost=sum(t['total_cost'] for t in v['account']['trades'] if t['date']=='2025-06-18')
    assert daily['2025-06-18']['nav']==daily['2025-06-17']['nav']-day_cost
    bad=deepcopy(data);bad['actions'][0]['multiplier']=5
    with pytest.raises(ValueError):audit_benchmark(v,bad)

def test_continuous_prefix_rejects_year_boundary_reset():
    from scripts.research_index_continuous import prefix_audit
    account={k:[] for k in ('daily','trades','orders','cash_ledger','holdings','corporate_actions')}
    account['daily']=[dict(date='2021-12-30',nav=2000000),dict(date='2022-01-03',nav=2010000)]
    expected=deepcopy(account);expected['daily']=expected['daily'][:1]
    assert prefix_audit(account,expected,'2021-12-30')['daily_and_all_journal_prefixes_identical']
    account['daily'][0]['nav']=1000000
    with pytest.raises(ValueError):prefix_audit(account,expected,'2021-12-30')
