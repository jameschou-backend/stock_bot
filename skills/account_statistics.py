"""Descriptive uncertainty from aligned, audited account NAV, never filled gaps."""
from collections import OrderedDict
from dataclasses import asdict
from datetime import date
import math
from numbers import Real

import numpy as np

from skills.statistics import paired_block_bootstrap_sharpe_ci


def monthly_pairs(account, benchmark, expected_dates):
    """Use only complete interior months; edge months are explicitly excluded.

    The caller supplies the independently validated market calendar. No month
    or missing benchmark return is forward-filled or replaced by zero.
    This research account has no deposits/withdrawals after initial funding.
    """
    if not expected_dates or expected_dates != sorted(set(expected_dates)):
        raise ValueError('Expected market calendar must be unique and ordered')
    if any(date.fromisoformat(d).isoformat()!=d for d in expected_dates):
        raise ValueError('Expected dates must be ISO dates')
    series=[]
    for value in (account,benchmark):
        rows=value['daily']
        if [r['date'] for r in rows]!=expected_dates:
            raise ValueError('Account and benchmark must cover the exact market calendar')
        previous=value['settings']['initial_cash'];months=OrderedDict()
        for row in rows:
            before,after=row['opening_nav'],row['nav']
            if any(isinstance(x,bool) or not isinstance(x,Real) or not math.isfinite(x) or x<=0
                   for x in (previous,before,after)):
                raise ValueError('NAV must be positive and finite')
            if abs(before-previous)>.02:
                raise ValueError('NAV chain is broken or contains external funding')
            if abs(row['daily_return']-(after/before-1))>1e-10 or not math.isfinite(row['daily_return']):
                raise ValueError('Reported daily return differs from NAV')
            month=row['date'][:7]
            if month not in months:months[month]=[before,after]
            else:months[month][1]=after
            previous=after
        series.append(months)
    keys=list(series[0]);interior=keys[1:-1]
    if len(interior)<12:
        raise ValueError('At least 12 complete interior months are required')
    values=[np.array([s[m][1]/s[m][0]-1 for m in interior]) for s in series]
    return dict(months=interior,strategy=values[0],benchmark=values[1],
                excluded_boundary_months=[keys[0],keys[-1]])


def describe_account(account, benchmark, expected_dates):
    pairs=monthly_pairs(account,benchmark,expected_dates)
    result=paired_block_bootstrap_sharpe_ci(pairs['strategy'],pairs['benchmark'],
        block_size=6,n_boot=2000,seed=20260927,periods_per_year=12,risk_free_rate=0.)
    return dict(method='paired_circular_monthly_block_bootstrap',
        scope='descriptive_current_account_not_selection_adjusted',
        months=pairs['months'],excluded_boundary_months=pairs['excluded_boundary_months'],
        monthly_strategy=pairs['strategy'].tolist(),monthly_benchmark=pairs['benchmark'].tolist(),
        bootstrap=asdict(result),
        excess_interval_above_zero=result.excess_ci_low>0,
        selection_adjusted_statistics_verified=False,live_qualified=False,
        dsr=dict(available=False,reason='Complete historical trials and their Sharpe dispersion are unverified'))
