import numpy as np
import pandas as pd
import pytest

from skills.independent_three_black import ThreeBlackPath, observe
from scripts.research_early_exit_support import simulate, compare, followup, followup_summary, metrics


def path(n=90):
    a=np.full(n,100.)
    return ThreeBlackPath(pd.bdate_range('2024-01-01',periods=n),a.copy(),a.copy(),np.ones(n,dtype=bool),a.copy(),a+2,a-2,a*10000,a.copy())


def change(p,changes):
    for i,v in changes.items():
        for a in (p.close,p.other,p.raw_close):a[i]=v
        p.opened[i],p.high[i],p.low[i]=v+1,v+2,v-2


def test_scalar_control_reproduces_existing_three_black_and_next_session_fill():
    p=path();change(p,{1:105,2:104,3:103,4:102,5:101})
    a,b=simulate(p,1),observe(p,1)
    for k in ('status','exit_reason','exit_trigger_date','exit_date','holding_days','gross_return','net_return'):
        assert a[k]==b[k]
    assert a['exit_date']==str(p.days[5].date())


def test_support_is_strict_and_does_not_trigger_just_because_later_price_broke_it():
    p=path();change(p,{1:105,2:104,3:103,4:102,5:101})
    a=simulate(p,1,support=102)
    assert a['exit_trigger_date']==str(p.days[5].date())  # equal102 cannot exit
    assert a['exit_date']==str(p.days[6].date())
    p.opened[5]=100  # support broke but no third black
    assert simulate(p,1,support=102)['exit_reason']=='time63'


@pytest.mark.parametrize('support',[1.,1000.])
def test_stop_and_deadline_do_not_wait_for_support(support):
    p=path();change(p,{1:99,2:95,3:87,4:80})
    r=simulate(p,1,support)
    assert r['exit_reason']=='loss12' and r['exit_date']==str(p.days[4].date())
    p=path();change(p,{61:99,62:98,63:97,64:96})
    r=simulate(p,1,support)
    assert r['exit_reason']=='time63' and r['holding_days']==63


def test_delay_can_turn_a_closed_baseline_into_pending_or_open_and_stays_in_counts():
    p=path(6);change(p,{1:99,2:98,3:97,4:96,5:95})
    a=dict(simulate(p,1),signal_id='e',stock_id='1234',name='N')
    b=dict(simulate(p,1,95),signal_id='e',stock_id='1234',name='N')
    r=compare([a],[b])
    assert a['status']=='closed' and b['status']=='open'
    assert r['paired_closed']==0 and r['variant_all']['status_counts']['open']==1
    assert len(r['baseline_closed_to_other'])==1
    assert simulate(p,1,95.5)['status']=='pending_exit'


def test_invalid_path_remains_unresolved_and_prices_after_exit_do_not_change_control():
    p=path();change(p,{1:99,2:98,3:97,4:96});a=simulate(p,1)
    p.close[5:]=np.nan
    assert simulate(p,1)==a
    assert simulate(p,1,1)['status']=='unresolved'
    assert simulate(p,1,1)['exit_reason'] is None


def test_post_exit_recovery_is_after_sale_and_requires_full_window():
    p=path(12);change(p,{1:99,2:98,3:97,4:96,5:101,6:98,7:97,8:96,9:95})
    row=dict(simulate(p,1),signal_id='e',stock_id='1234',name='N',signal_date=str(p.days[0].date()),entry_date=str(p.days[1].date()))
    r=followup(p,row,5)
    assert r['status']=='complete' and r['recovered_entry_within_window']
    assert not r['recovered_entry_at_end']
    assert r['exit_to_close_return']==pytest.approx(95/96-1)
    assert r['entry_to_close_return']==pytest.approx(95/99-1)
    assert followup(p,row,20)['status']=='incomplete_window'
    p.other[6]=90
    bad=followup(p,row,5)
    assert bad['status']=='unresolved' and bad['exit_to_close_return'] is None
    assert followup_summary([r,bad,followup(p,row,20)])['complete']==1


def test_worst_tail_uses_ceiling_and_excludes_open_or_unknown():
    rows=[dict(status='closed',exit_reason='x',net_return=-i/100,holding_days=i) for i in range(1,22)]
    rows.append(dict(status='unresolved',data_issue_code='gap'))
    m=metrics(rows)
    assert m['closed']==21 and m['worst5_count']==2
    assert m['worst5_mean']==pytest.approx(-.205)
