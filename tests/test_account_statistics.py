from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from skills.account_statistics import monthly_pairs,describe_account


def accounts():
    dates=[str(d.date()) for d in pd.bdate_range('2022-01-03','2023-04-12')]
    nav=1_000_000.;rows=[]
    for i,day in enumerate(dates):
        r=.001+np.sin(i)*.01;end=nav*(1+r)
        rows.append(dict(date=day,opening_nav=nav,nav=end,daily_return=r));nav=end
    a=dict(settings=dict(initial_cash=1_000_000.),daily=rows)
    return a,deepcopy(a),dates


def test_calendar_aligned_monthly_compounding_and_edge_exclusion():
    a,b,dates=accounts();result=monthly_pairs(a,b,dates)
    assert result['months'][0]=='2022-02' and result['months'][-1]=='2023-03'
    rows=[r for r in a['daily'] if r['date'].startswith('2022-02')]
    assert result['strategy'][0]==pytest.approx(np.prod([1+r['daily_return'] for r in rows])-1)
    assert result['excluded_boundary_months']==['2022-01','2023-04']


def test_identical_accounts_have_zero_excess_interval_without_qualification():
    a,b,dates=accounts();result=describe_account(a,b,dates)
    assert result['bootstrap']['excess_ci_low']==result['bootstrap']['excess_ci_high']==0.
    assert not result['live_qualified'] and not result['dsr']['available']


@pytest.mark.parametrize('corruption',['missing_benchmark','duplicate','funding','nan','return'])
def test_incomplete_or_tampered_nav_is_rejected(corruption):
    a,b,dates=accounts()
    if corruption=='missing_benchmark':b['daily'].pop(3)
    elif corruption=='duplicate':b['daily'][3]['date']=b['daily'][2]['date']
    elif corruption=='funding':a['daily'][3]['opening_nav']+=1000
    elif corruption=='nan':a['daily'][3]['nav']=float('nan')
    else:a['daily'][3]['daily_return']+=.01
    with pytest.raises(ValueError):monthly_pairs(a,b,dates)
