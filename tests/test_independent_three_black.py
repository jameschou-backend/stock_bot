from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from skills.independent_three_black import ThreeBlackPath, observe, summarize
from skills.three_black_exit import ThreeBlackSignals


def path(n=75):
    p = np.full(n, 100.)
    return ThreeBlackPath(pd.bdate_range('2024-01-01', periods=n), p.copy(), p.copy(),
        np.ones(n, dtype=bool), p.copy(), p+2, p-2, p*10000, p.copy())


def prices(p, values):
    for k, v in values.items():
        for a in (p.close,p.other,p.raw_close): a[k] = v
        p.high[k],p.low[k],p.opened[k] = v+2,v-2,v+1


def test_first_three_held_black_candles_sell_only_the_following_session():
    p = path(8);prices(p, {1:99,2:98,3:97,4:96})
    result = observe(p, 1)
    assert result['exit_reason']=='three_black'
    assert result['exit_trigger_date']==str(p.days[3].date())
    assert result['exit_date']==str(p.days[4].date())
    assert result['holding_days']==3 and result['holding_days_inclusive']==4
    quotes=pd.DataFrame(dict(date=p.days,stock_id='1234',open=p.opened,close=p.raw_close,
                             high=p.high,low=p.low,volume=p.volume))
    oracle=ThreeBlackSignals(pd.DataFrame({'1234':p.close},index=p.days),quotes,p.days)
    assert oracle.exits(4,1,'1234') and not oracle.exits(3,1,'1234')


def test_stop_has_priority_and_preserves_overnight_gap():
    p = path(8);prices(p,{1:99,2:95,3:87,4:80})
    r=observe(p,1)
    assert r['exit_reason']=='loss12' and r['exit_date']==str(p.days[4].date())
    assert r['gross_return']==pytest.approx(80/99-1)


def test_time_has_priority_over_three_black_and_counts_elapsed_sessions():
    p=path();prices(p,{61:99,62:98,63:97,64:96})
    r=observe(p,1)
    assert r['exit_reason']=='time63'
    assert r['holding_days']==63 and r['holding_days_inclusive']==64


def test_stop_anchor_is_entry_close_not_hl2_or_intraday_low():
    p=path(8);p.high[1]=130;p.low[2]=80
    assert observe(p,1)['status']=='open'


def test_final_close_trigger_is_pending_and_excluded_from_win_rate():
    p=path(4);prices(p,{1:99,2:98,3:97})
    r=observe(p,1);r['stock_id']='1234'
    assert r['status']=='pending_exit' and r['exit_date'] is None
    assert r['net_return'] is None and r['unrealized_net_return']<0
    assert summarize([r])['win_rate'] is None


def test_future_price_cannot_change_exit_or_peak_and_exit_day_high_is_only_upper_bound():
    p=path(8);prices(p,{1:99,2:98,3:97,4:96})
    r=observe(p,1)
    p.high[4]=150
    upper=observe(p,1)
    assert upper['mfe']>r['mfe']
    assert upper['confirmed_high_return']==r['confirmed_high_return']
    assert upper['peak_close_return']==r['peak_close_return']
    for a in (p.close,p.other,p.raw_close,p.high,p.low,p.opened):a[5:]=np.nan
    assert observe(p,1)==upper


def test_missing_earlier_path_does_not_keep_a_later_apparent_exit():
    p=path();p.opened[3]=np.nan
    r=observe(p,1)
    assert r['status']=='unresolved' and r['net_return'] is None
    assert r['exit_reason'] is None and r['exit_trigger_date'] is None


def test_split_does_not_become_a_stop_or_profit():
    p=path()
    for a in (p.raw_close,p.opened,p.high,p.low):a[20:]/=2
    r=observe(p,1)
    assert r['exit_reason']=='time63' and r['gross_return']==pytest.approx(0)
    assert r['exit_price']==50 and r['outcome']=='loss'


def test_summaries_keep_overlapping_signals_and_only_count_resolved_closes():
    p=path();a=observe(p,1);b=observe(p,2)
    rows=[dict(a,stock_id='1234'),dict(b,stock_id='1234'),
          dict(stock_id='1234',status='not_entered'),dict(stock_id='2345',status='unresolved')]
    s=summarize(rows)
    assert s['total']==4 and s['stocks']==2 and s['closed']==2 and s['loss']==2
    assert s['win_rate']==0 and s['not_entered']==1 and s['unresolved']==1


def test_zero_volume_or_conflicting_adjustments_are_not_synthetic_fills():
    p=path();p.volume[64]=0
    assert observe(p,1)['data_issue']=='no_volume_on_assumed_exit'
    p=path();p.other[5]=95
    assert observe(p,1)['data_issue']=='daily_adjustment_conflict'
    with pytest.raises(ValueError):observe(replace(p,opened=np.ones(3)),1)
