from copy import deepcopy

import pandas as pd
import pytest

from skills.support_risk_source_audit import audit_candidate_binding


def fixture(mask=0):
    days=pd.bdate_range('2026-01-05',periods=5)
    dates=[str(d.date()) for d in days]
    original=dict(event_id='x',members=['1101'],entry_date=dates[1],signal_date=dates[0])
    row=dict(event_id='x',stock_id='1101',date=dates[1+bool(mask&2)],signal_date=dates[0],side='buy')
    cohort=dict(row,entry_date=row['date'])
    case=dict(completed=True,config=dict(factor_mask=mask),technical_entries=[deepcopy(row)],
        account=dict(cohorts=[cohort],trades=[deepcopy(row)],orders=[deepcopy(row)]))
    return case,[original],days


@pytest.mark.parametrize('mask',range(8))
def test_every_stress_keeps_original_signal(mask):
    result=audit_candidate_binding(*fixture(mask))
    assert result['checked_records']==4 and result['entries_reanchored_after_delay'] is False


@pytest.mark.parametrize('field',['technical_entries','cohorts','trades','orders'])
@pytest.mark.parametrize('mutation',['signal_date','stock_id','event_id','date'])
def test_consistent_internal_logs_still_cannot_override_frozen_candidates(field,mutation):
    case,source,days=fixture(2)
    row=(case[field] if field=='technical_entries' else case['account'][field])[0]
    key='entry_date' if field=='cohorts' and mutation=='date' else mutation
    row[key]={'signal_date':'2026-01-06','stock_id':'1102','event_id':'other','date':'2026-01-06'}[mutation]
    with pytest.raises(ValueError):audit_candidate_binding(case,source,days)
