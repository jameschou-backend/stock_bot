import numpy as np
import pandas as pd
import pytest
from skills.absorption import candidates, orders, outcomes, statistics, matched_pairs, comparisons
from skills.launch_flows import normalize
from scripts.prepare_absorption import plan, validate


def fixture():
    days=pd.bdate_range('2021-01-04',periods=450); ids=['0050','2300','2301','2302']
    c=pd.DataFrame(100.,index=days,columns=ids); raw=c.copy(); v=c*0+1_000_000
    companies=pd.DataFrame([dict(stock_id=sid,name=sid,industry='01',market='TWSE',listed_date='2000-01-01') for sid in ids[1:]])
    inst=pd.DataFrame([dict(date=day,stock_id=sid,foreign=-40_000.,trust=0.,dealer=0.) for day in days for sid in ids[1:]])
    return c,raw,v,companies,inst


def first(table,sid='2300',lag=1):
    return table[(table.stock_id==sid)&(table.flow_lag==lag)].iloc[0]


def test_broad_cohort_does_not_require_volume_burst_or_breakout():
    a=fixture(); table=candidates(*a); r=first(table)
    assert r.flow_ratio==pytest.approx(-.04)
    assert r.absorption==1 and r.buy_pressure==0 and r.sell_weak==0
    assert r.price_base_date < r.window_start <= r.feature_date < r.signal_date
    e=orders(table,a[0].index)['absorption'][0]
    assert e['entry_date'] > e['signal_date'] and e['institutional_feature_date'] < e['signal_date']


def test_today_flow_is_not_available_and_lags_use_aligned_price_windows():
    a=fixture(); baseline=candidates(*a); r=first(baseline); day=pd.Timestamp(r.signal_date)
    a[-1].loc[a[-1].date==day,['foreign','trust']]=999_999
    pd.testing.assert_series_equal(r,first(candidates(*a)))
    assert first(baseline,lag=3).feature_date==str(a[0].index[a[0].index.get_loc(day)-3].date())


def test_missing_day_is_unknown_but_zero_net_is_known():
    a=fixture(); r=first(candidates(*a)); day=pd.Timestamp(r.feature_date)
    a[-1].loc[(a[-1].date==day)&(a[-1].stock_id=='2300'),'trust']=np.nan
    row=first(candidates(*a)); assert not row.feature_known and np.isnan(row.absorption)
    a[-1].loc[a[-1].stock_id=='2300',['foreign','trust']]=0
    row=first(candidates(*a)); assert row.feature_known and row.resilient==1 and row.absorption==0


def test_low_volume_denominator_and_wrong_unit_cannot_be_hidden():
    a=fixture(); before=first(candidates(*a)); p=a[0].index.get_loc(pd.Timestamp(before.feature_date))
    a[2].iloc[p-4:p+1,1]=2_000_000
    after=first(candidates(*a)); assert after.flow_ratio==pytest.approx(-.02) and after.absorption==0


def test_intra_window_breach_fails_even_when_last_price_recovers():
    a=fixture(); r=first(candidates(*a)); p=a[0].index.get_loc(pd.Timestamp(r.feature_date))
    a[0].iloc[p-2,1]=97.
    row=first(candidates(*a)); assert row.return5==0 and row.min_close_return5==pytest.approx(-.03)
    assert row.sell_weak==1 and row.absorption==0


def test_positive_return_that_lags_market_is_not_resilient():
    a=fixture(); r=first(candidates(*a)); p=a[0].index.get_loc(pd.Timestamp(r.feature_date))
    a[0].iloc[p,0]=101.; a[0].iloc[p,1]=100.5
    row=first(candidates(*a)); assert row.return5>0 and row.excess5<0 and row.resilient==0


def test_missing_category_never_becomes_zero_and_legacy_schema_is_unknown():
    raw=pd.DataFrame([dict(date='2022-01-03',stock_id='2300',name=name,buy=100,sell=200)
        for name in ('Foreign_Investor','Investment_Trust','Dealer_self','Dealer_Hedging')])
    n=normalize(raw); assert np.isnan(n.iloc[0].foreign) and n.iloc[0].trust==-100
    raw.loc[len(raw)]=dict(date='2022-01-03',stock_id='2300',name='Dealer',buy=1,sell=2)
    n=normalize(raw); assert not n.iloc[0].schema_supported and np.isnan(n.iloc[0].trust)


def test_future_outcome_changes_do_not_change_candidate_or_order():
    a=fixture(); table=candidates(*a); r=first(table); cutoff=pd.Timestamp(r.signal_date)
    for f in a[:3]: f.loc[f.index>cutoff]*=2
    a[-1].loc[a[-1].date>cutoff,['foreign','trust']]*=-7
    changed=candidates(*a)
    pd.testing.assert_frame_equal(table[table.signal_date<=r.signal_date],changed[changed.signal_date<=r.signal_date])


def test_future_missing_price_and_unmatured_stay_unknown():
    a=fixture(); t=candidates(*a); day=pd.Timestamp(first(t).signal_date); p=a[0].index.get_loc(day)
    quality=a[0].copy(); quality.iloc[p+2,1]=np.nan
    result=outcomes(t,a[0],quality)
    r=result[(result.stock_id=='2300')&(result.signal_date==str(day.date()))]
    assert r.forward_return.isna().all() and r.label_reason.eq('price_missing').all()
    assert result[result.signal_date==t.signal_date.max()].label_reason.eq('unmatured').all()
    s=statistics(result); assert (s.outcome_unknown>0).any()


def test_matching_is_date_industry_pressure_bound_and_outcome_independent():
    a=fixture(); t=candidates(*a)
    t=t[t.signal_date.eq(t.signal_date.iloc[0])].copy()
    t.loc[t.stock_id.eq('2301'),['resilient','absorption']]=0
    t.loc[t.stock_id.eq('2301'),'sell_weak']=1
    t.loc[t.stock_id.eq('2302'),'industry']='02'
    pairs=matched_pairs(t)
    assert pairs[pairs.event_id.str.endswith('2300')].control_id.str.endswith('2301').all()
    assert pairs[pairs.event_id.str.endswith('2302')].control_id.isna().all()
    t['future_fake']=999999
    pd.testing.assert_frame_equal(pairs,matched_pairs(t))
    future=outcomes(t,a[0],a[0]); attached,report=comparisons(future,pairs)
    assert report[report.comparison.eq('matched_sell_weak')].unknown.gt(0).all()


def test_market_pages_require_requested_date_and_preserve_noncohort_exclusions():
    ids={str(1000+i) for i in range(501)}
    raw=pd.DataFrame([dict(date='2022-01-03',stock_id=sid,name=name,buy=0,sell=0)
        for sid in ids|{'00631L'} for name in ('Foreign_Investor','Foreign_Dealer_Self','Investment_Trust','Dealer_self','Dealer_Hedging')])
    result,rejected=validate(raw,'2022-01-03',ids)
    assert len(result)==501 and rejected==['00631L'] and result.foreign.eq(0).all()
    with pytest.raises(ValueError,match='single date'): validate(raw,'2022-01-04',ids)
    with pytest.raises(ValueError,match='Duplicate'): validate(pd.concat([raw,raw.iloc[:1]]),'2022-01-03',ids)


def test_collection_dates_are_fixed_and_not_based_on_prices():
    days=fixture()[0].index; anchors,dates=plan(days)
    assert len(dates)==len(anchors)*7 and all(day < days[-1] for day in dates)
    assert len(set(dates))==len(dates)
