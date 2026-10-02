from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from skills.independent_three_black import ThreeBlackPath,observe
from scripts.research_entry_distance_20261002 import signal_distance,entry_proxy,execution_row,opportunity_comparison,retention


def path():
    a=np.full(150,100.)
    return ThreeBlackPath(pd.bdate_range('2024-01-01',periods=150),a.copy(),a.copy(),np.ones(150,dtype=bool),a.copy(),a+2,a-2,a*1000,a.copy())


def original(p,entry=62):
    return dict(observe(p,entry),signal_id='e',stock_id='1234',name='N',signal_date=str(p.days[entry-1].date()),entry_date=str(p.days[entry].date()))


def test_signal_screen_and_limit_use_only_signal_time_prices_and_tick_floor():
    p=path();p.close[61]=p.other[61]=p.raw_close[61]=105
    f=signal_distance(p,61)
    assert f['signal_distance5'] and f['support']==100 and f['t0_limit_raw']==105
    p.close[62:]=900;p.raw_close[62:]=3;p.high[62:]=1000
    assert signal_distance(p,61)==f
    p.close[61]=p.raw_close[61]=105.001
    assert not signal_distance(p,61)['signal_distance5']


def test_truncating_future_preserves_t0_instruction_and_t1_no_fill():
    p=path();f=signal_distance(p,61)
    truncated=replace(p,days=p.days[:62],**{k:getattr(p,k)[:62] for k in ('close','other','eligible','raw_close','high','low','volume','opened')})
    assert signal_distance(truncated,61)==f
    p.opened[62]=110;p.high[62]=112;p.low[62]=109;p.raw_close[62]=p.close[62]=p.other[62]=110
    result=entry_proxy(p,62,f,'preplaced_limit5')
    truncated=replace(p,days=p.days[:63],**{k:getattr(p,k)[:63] for k in ('close','other','eligible','raw_close','high','low','volume','opened')})
    assert result==entry_proxy(truncated,62,f,'preplaced_limit5')
    for key in ('close','other','raw_close','high','low','volume','opened'):getattr(p,key)[63:]=np.nan
    p.eligible[63:]=False
    assert result==entry_proxy(p,62,f,'preplaced_limit5')


@pytest.mark.parametrize('support,expected',[(9.99,10.45),(49.99,52.4),(99.99,104.5),(499.99,524.),(999.99,1045.)])
def test_limit_rounding_never_exceeds_fixed_instruction(support,expected):
    p=path();p.close[:61]=support;p.raw_close[:61]=support
    f=signal_distance(p,61)
    assert f['t0_limit_raw']==expected and f['t0_limit_raw']<=support*1.05


def test_open_and_strict_intraday_limit_fill_branches():
    p=path();f=signal_distance(p,61)
    assert entry_proxy(p,62,f,'preplaced_limit5')['entry_proxy_price']==100
    p.opened[62]=107;p.high[62]=110;p.low[62]=104;p.raw_close[62]=p.close[62]=p.other[62]=106
    assert entry_proxy(p,62,f,'open_control')['entry_proxy_price']==107
    assert entry_proxy(p,62,f,'preplaced_limit5')['entry_proxy_price']==105
    p.low[62]=105
    assert entry_proxy(p,62,f,'preplaced_limit5')['entry_proxy_status']=='not_filled'
    p.opened[62]=105
    assert entry_proxy(p,62,f,'preplaced_limit5')['entry_proxy_price']==105


@pytest.mark.parametrize('mutation,issue',[('split','t0_t1_adjustment_factor_changed'),('single','single_price_session_without_queue_evidence'),('volume','missing_or_nonpositive_entry_ohlcv'),('ohlc','impossible_entry_ohlc')])
def test_open_and_limit_share_explicit_unknown_execution_guards(mutation,issue):
    p=path();f=signal_distance(p,61)
    if mutation=='split':
        for a in (p.raw_close,p.opened,p.low,p.high):a[62]/=2
    elif mutation=='single':p.high[62]=p.low[62]=100
    elif mutation=='volume':p.volume[62]=0
    else:p.opened[62]=110
    for arm in ('open_control','preplaced_limit5'):
        r=entry_proxy(p,62,f,arm)
        assert r['entry_proxy_status']=='unresolved' and r['entry_proxy_issue']==issue


def test_signal_day_quality_guards_are_shared_by_both_execution_arms():
    p=path();f=signal_distance(p,61);p.volume[61]=0
    for arm in ('open_control','preplaced_limit5'):
        assert entry_proxy(p,62,f,arm)['entry_proxy_issue']=='missing_or_nonpositive_signal_ohlcv'
    p.volume[61]=100;p.opened[61]=999
    for arm in ('open_control','preplaced_limit5'):
        assert entry_proxy(p,62,f,arm)['entry_proxy_issue']=='impossible_signal_ohlc'


def test_known_no_fill_is_not_erased_by_unneeded_future_data_gap():
    p=path();p.opened[62]=110;p.high[62]=112;p.low[62]=109;p.raw_close[62]=p.close[62]=p.other[62]=110
    p.close[64]=np.nan
    row=original(p);assert row['status']=='unresolved'
    f=signal_distance(p,61)
    unfilled=execution_row(row,p,62,f,'preplaced_limit5')
    assert unfilled['status']=='not_filled' and unfilled['entry_date'] is None
    assert unfilled['planned_entry_date']==row['entry_date']
    assert execution_row(row,p,62,f,'open_control')['status']=='unresolved'


def test_repricing_preserves_exit_anchor_timing_and_original_unrealized_status():
    p=path();p.opened[62]=101
    row=original(p);f=signal_distance(p,61);new=execution_row(row,p,62,f,'open_control')
    for k in ('exit_reason','exit_trigger_date','exit_date','holding_days','stop_anchor_adjusted_close'):
        assert new[k]==row[k]
    assert new['net_return']<row['net_return'] and new['entry_price']==101


def test_opportunity_comparison_preserves_cash_zero_and_excludes_unknown_separately():
    b=[dict(signal_id=str(i),status='closed',net_return=.4) for i in range(3)]
    c=deepcopy(b);v=deepcopy(b);v[1]['status']='not_filled';v[2]['status']='unresolved'
    r=opportunity_comparison(b,c,v)
    assert r['common_known_events']==2 and r['known_not_filled']==1 and r['excluded_unknown']==1
    assert r['variant_mean_with_known_unfilled_cash_zero']==.2 and r['control_mean']==.4
    for row in v:row['entry_proxy_status']='filled_proxy' if row['status']=='closed' else row['status']
    counts=retention(b,v)['net30']
    assert counts['fill_retention']==pytest.approx(1/3) and counts['known_not_filled']==counts['unknown']==1


@pytest.mark.parametrize('mutation,issue',[('identity','signal_or_entry_identity_unresolved'),('other','signal_entry_adjustment_reference_conflict')])
def test_shared_entry_identity_and_two_source_transition_unknowns(mutation,issue):
    p=path();f=signal_distance(p,61)
    if mutation=='identity':p.eligible[62]=False
    else:p.other[62]=105
    for arm in ('open_control','preplaced_limit5'):
        assert entry_proxy(p,62,f,arm)['entry_proxy_issue']==issue
