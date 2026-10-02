import numpy as np
import pandas as pd
import pytest

from scripts.research_breakout_phase_20261002 import (price_breakout_history,
    classify_phase,annotate_signals,opportunity_summary,PHASES)


def prices(previous=None):
    days=pd.bdate_range('2024-01-01',periods=110)
    c=pd.Series(100.,index=days)
    if previous is not None:c.iloc[previous]=101.
    c.iloc[90]=102.
    return c,c.copy(),pd.Series(True,index=days)


@pytest.mark.parametrize('previous,expected',[(None,'first_after_20'),(69,'first_after_20'),
    (70,'rebreak_after_5'),(84,'rebreak_after_5'),(85,'recent_repeat'),(89,'recent_repeat')])
def test_fixed_boundary_definitions_are_disjoint(previous,expected):
    c,other,e=prices(previous)
    value=classify_phase(price_breakout_history(c,other,e),c.index[90])
    assert value['phase']==expected
    assert value['phase_current_price_breakout'] is True
    assert value['phase_available_at'].startswith(str(c.index[90].date()))


def test_price_breakouts_not_strategy_signal_membership_control_phase():
    c,other,e=prices(80);day=str(c.index[90].date())
    # The prior price breakout at80 is deliberately absent from original signals.
    rows=[dict(signal_id='only_current',signal_date=day,stock_id='2330',status='closed',net_return=.8)]
    result=annotate_signals(rows,c.to_frame('2330'),other.to_frame('2330'),e.to_frame('2330'))
    assert result[0]['phase']=='rebreak_after_5'
    assert result[0]['phase_last_breakout_distance_in20']==10
    assert sum(result[0]['phase_keep_'+k] for k in PHASES)==1


def test_cutoff_prefix_and_modified_future_prices_give_identical_phase():
    c,other,e=prices(80);day=c.index[90]
    full=classify_phase(price_breakout_history(c,other,e),day)
    prefix=classify_phase(price_breakout_history(c.loc[:day],other.loc[:day],e.loc[:day]),day)
    changed=c.copy();changed.iloc[91:]=np.linspace(0,999999,len(changed)-91)
    changed_other=other.copy();changed_other.iloc[91:]=np.nan
    future_elig=e.copy();future_elig.iloc[91:]=False
    after=classify_phase(price_breakout_history(changed,changed_other,future_elig),day)
    assert full==prefix==after


def test_unknown_prior_flags_are_not_interpreted_as_no_breakout():
    c,other,e=prices();c.iloc[29]=np.nan
    # Today's60-prior observations are valid, but a prior20 flag window overlaps29.
    value=classify_phase(price_breakout_history(c,other,e),c.index[90])
    assert value['phase']=='unknown'
    assert value['phase_issue']=='unknown_price_breakout_in_prior20'
    assert value['phase_previous20_unknown_count']>0


@pytest.mark.parametrize('kind',['missing_price','bad_adjustment','ineligible'])
def test_invalid_reference_history_has_unknown_phase(kind):
    c,other,e=prices(80)
    if kind=='missing_price':c.iloc[60]=np.nan
    elif kind=='bad_adjustment':other.iloc[60]=50.
    else:e.iloc[60]=False
    value=classify_phase(price_breakout_history(c,other,e),c.index[90])
    assert value['phase']=='unknown'
    assert value['phase_current_price_breakout'] is None


def test_unknown_opportunities_are_separate_from_rejected_cash_units():
    rows=[dict(status='closed',net_return=.4,phase='first_after_20'),
          dict(status='closed',net_return=-.2,phase='recent_repeat'),
          dict(status='closed',net_return=.8,phase='unknown'),
          dict(status='open',net_return=None,phase='first_after_20')]
    value=opportunity_summary(rows,'first_after_20')
    assert value['original_closed_opportunities']==3
    assert value['known_closed_opportunities']==2 and value['unknown_closed_opportunities']==1
    assert value['selected_closed_opportunities']==1 and value['rejected_known_closed_opportunities']==1
    assert value['equal_units_mean_per_known_original_opportunity']==.2
    assert value['unfiltered_mean_on_same_known_opportunities']==.1
    assert value['equal_units_mean_per_all_original_opportunity'] is None


def test_original_signal_outcomes_do_not_change_phase_or_source_metadata():
    c,other,e=prices(80);r=dict(signal_id='a',stock_id='2330',signal_date=str(c.index[90].date()))
    def features(row):
        result=annotate_signals([row],c.to_frame('2330'),other.to_frame('2330'),e.to_frame('2330'))[0]
        return {k:v for k,v in result.items() if k.startswith('phase')}
    assert features({**r,'net_return':-.9,'exit_date':'2024-01-01'})==features({**r,'net_return':50.,'exit_date':'2030-01-01'})
