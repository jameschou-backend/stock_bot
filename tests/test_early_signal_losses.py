from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts.research_early_signal_losses import (
    known_signal_features, posthoc_features, statistics, evaluate_filter, distributions,
)
from skills.independent_three_black import ThreeBlackPath


def fixture():
    days=pd.bdate_range('2024-01-01',periods=90)
    c=np.arange(100.,190.)
    return ThreeBlackPath(days,c.copy(),c.copy(),np.ones(90,dtype=bool),c.copy(),
                          c+2,c-2,np.ones(90)*1000,c-1)


def row(path,index,key='a'):
    return dict(signal_id=key,stock_id='2330',signal_date=str(path.days[index].date()),
                entry_date=str(path.days[index+1].date()) if index+1<len(path.days) else None)


def test_signal_features_do_not_read_future_observations_or_outcomes():
    p=fixture();r=row(p,65)
    before=known_signal_features(p.days,[r],p)
    changed={}
    for name in ('close','other','raw_close','high','low','opened','volume'):
        a=getattr(p,name).copy();a[66:]=np.nan;changed[name]=a
    altered=replace(p,**changed)
    after=known_signal_features(p.days,[{**r,'net_return':-1,'exit_date':'2099-12-31'}],altered)
    assert before==after
    assert before['a']['t0_breakout_threshold']==164
    assert before['a']['t0_available_at'].startswith(r['signal_date'])


def test_repeat_screen_counts_original_signals_even_when_prior_signal_rejected():
    p=fixture();rows=[row(p,i,str(i)) for i in (60,65,75,86)]
    values=known_signal_features(p.days,rows,p)
    assert values['60']['no_previous_signal_10'] is True
    assert values['65']['no_previous_signal_10'] is False
    # 65 was rejected but still blocks 75 at an inclusive 10-session distance.
    assert values['75']['no_previous_signal_10'] is False
    assert values['86']['no_previous_signal_10'] is True


def test_signal_source_label_does_not_depend_on_future_exit_or_path_scope():
    p=fixture();r=row(p,65)
    before=known_signal_features(p.days,[{**r,'observed_end_date':'2024-05-01',
        'data_source_scope':'historical_only'}],p)
    after=known_signal_features(p.days,[{**r,'observed_end_date':'2026-10-02',
        'data_source_scope':'includes_provider_extension'}],p)
    assert before==after
    assert before['a']['t0_source_scope'].startswith('修復後封存歷史訊號資料')


def test_single_price_close_location_and_bad_ohlc_are_unknown_not_pass():
    p=fixture();high=p.high.copy();low=p.low.copy();opened=p.opened.copy()
    high[65]=low[65]=opened[65]=p.raw_close[65]
    p=replace(p,high=high,low=low,opened=opened)
    value=known_signal_features(p.days,[row(p,65)],p)['a']
    assert value['close_location_75'] is None
    assert value['t0_close_location'] is None
    high[65]=p.raw_close[65]-1
    value=known_signal_features(p.days,[row(p,65)],replace(p,high=high))['a']
    assert value['close_above_open'] is None
    assert value['t0_raw_bar_valid'] is False


def test_posthoc_uses_three_entry_sessions_and_has_explicit_later_availability():
    p=fixture();r=row(p,65)
    result=posthoc_features(r,p,p,164.)
    assert result['posthoc_first3_known_at'].startswith(str(p.days[68].date()))
    assert result['posthoc_entry_known_at'].startswith(str(p.days[66].date()))
    assert result['posthoc_entry_day_close_vs_assumed_entry']==pytest.approx(0)
    assert result['posthoc_first3_mean_volume_vs_signal']==1
    assert result['posthoc_first3_benchmark_return']==pytest.approx(168/165-1)
    changed=p.close.copy();changed[68]=10
    result=posthoc_features(r,replace(p,close=changed),p,164.)
    assert result['posthoc_first3_issue']=='daily_adjustment_conflict'
    assert result['posthoc_first3_below_breakout'] is None


def test_posthoc_close_loss_is_separate_from_signal_filter():
    p=fixture();r=row(p,65);high=p.high.copy();high[66]=180
    changed=replace(p,high=high)
    post=posthoc_features(r,changed,p,164.)
    assert post['posthoc_entry_day_close_vs_assumed_entry']==pytest.approx(166/172-1)
    assert known_signal_features(p.days,[r],p)==known_signal_features(p.days,[r],changed)


def test_missing_followup_and_unknown_filters_keep_denominators_visible():
    p=fixture();r=row(p,88)
    assert posthoc_features(r,p,p,170)['posthoc_first3_issue']=='insufficient_followup'
    rows=[dict(status='closed',net_return=-.05,holding_days=5,exit_reason='three_black',f=True),
          dict(status='closed',net_return=.4,holding_days=20,exit_reason='time63',f=None),
          dict(status='open',net_return=None,holding_days=2,exit_reason=None,f=False)]
    stats=statistics(rows)
    assert stats['closed']==2 and stats['early_loss_rate']==.5 and stats['worst5_mean']==-.05
    results=evaluate_filter(rows,'f')
    assert results['unknown']['total']==1 and results['unknown_return30']==1
    assert results['excluded']['closed']==0 and results['return30_retention']==0
    assert results['kept']['win_rate']==0


def test_boolean_diagnostic_distributions_preserve_unknowns():
    rows=[dict(status='closed',net_return=-.02,gross_return=-.005,holding_days=4,
               exit_reason='three_black',diagnostic_group='early_loss',close_above_open=v)
          for v in (True,False,None)]
    result=distributions(rows)['early_loss']['features']['close_above_open']
    assert result['known']==2 and result['unknown']==1
    assert result['mean']==.5 and result['median']==.5
