import numpy as np
import pandas as pd
import pytest
from skills.stock_launch import (features,outcomes,snapshot,episode_positions,choose_controls,
                                 paired_summary,causal_checks)


def inputs():
    days=pd.bdate_range('2021-01-04',periods=230)
    close=pd.DataFrame(100.,index=days,columns=['0050','1101','1102'])
    volume=close*10000
    companies=pd.DataFrame([dict(stock_id=sid,name=sid,market='TWSE',industry='01',
        listed_date=days[0]) for sid in ('1101','1102')])
    return close,close.copy(),volume,companies


@pytest.mark.parametrize('horizon,gain',[(20,.4),(60,.6)])
def test_next_session_entry_exact_endpoint_and_maturity(horizon,gain):
    close,_,_,_=inputs();p=130
    close.iloc[p+1:p+horizon+2,1]=np.linspace(105,105*(1+gain),horizon+1)
    close.iloc[p+horizon+2:,1]=105*(1+gain)
    label=outcomes(close,close,horizon)
    assert label['event'].iloc[p,1]
    assert label['forward_return'].iloc[p,1]==pytest.approx(gain)
    assert label['event'].iloc[-horizon-1:].isna().all().all()
    close.iloc[p+horizon+2:,1]=1  # Outside the labelled window.
    assert outcomes(close,close,horizon)['event'].iloc[p,1]


@pytest.mark.parametrize('defect',['entry_jump','missing','infinite','disagree','benchmark','signal_missing'])
def test_unresolved_prices_never_become_false_events(defect):
    close,_,_,_=inputs();other=close.copy();p=130
    if defect=='entry_jump':other.iloc[p+1:,1]=120
    if defect=='missing':other.iloc[p+10,1]=np.nan
    if defect=='infinite':other.iloc[p+10,1]=np.inf
    if defect=='disagree':other.iloc[p+1:p+22,1]=np.linspace(100,110,21)
    if defect=='benchmark':other.iloc[p+10,0]=np.nan
    if defect=='signal_missing':other.iloc[p,1]=np.nan
    assert pd.isna(outcomes(close,other,20)['event'].iloc[p,1])


def test_amount_heat_is_causal_and_not_a_share_split():
    close,raw,volume,companies=inputs();p=150
    raw.iloc[p:,1]/=4;volume.iloc[p:,1]*=4
    result=features(close,raw,volume,companies)
    assert result['numeric']['turnover_ratio'].iloc[p+5,0]==1
    assert not result['rules']['turnover_heat'].iloc[p+5,0]
    assert all(x['passed'] for x in causal_checks(close,raw,volume,companies,[close.index[p]]))
    raw.iloc[p,1]=np.nan
    assert pd.isna(features(close,raw,volume,companies)['rules']['turnover_heat'].iloc[p,0])


def test_snapshot_keeps_ineligible_named_cases_and_unknown_outcomes():
    close,raw,volume,companies=inputs();volume['1101']=10
    computed=features(close,raw,volume,companies);label=outcomes(close,close,20)
    rows=snapshot(computed,label,close.index,220,20,only=['1101'],eligible_only=False)
    assert len(rows)==1 and not rows.iloc[0].eligible and not rows.iloc[0].label_mature
    assert pd.isna(rows.iloc[0].event)
    assert snapshot(computed,label,close.index,130,20).stock_id.tolist()==['1102']


def test_retrospective_events_use_first_positive_then_fixed_cooldown():
    days=pd.bdate_range('2022-01-03',periods=50)
    panel=pd.DataFrame(dict(stock_id=['1101']*50,signal_date=[str(d.date()) for d in days],
                            eligible=True,event=True))
    panel.loc[0,'event']=pd.NA;panel.loc[1,'eligible']=False
    assert episode_positions(panel,days,20)==[('1101',2),('1101',23),('1101',44)]


def test_controls_exclude_named_winners_unknowns_and_wrong_strata():
    pool=pd.DataFrame([dict(stock_id=str(1101+i),event=False,market='TWSE',industry='01',
                           liquidity_bin=3,adv20=100e6+i) for i in range(7)])
    case=pool.iloc[0].to_dict()
    pool.loc[1,'event']=pd.NA;pool.loc[2,'event']=True;pool.loc[3,'industry']='02'
    pool.loc[4,'liquidity_bin']=2
    assert choose_controls(pool,case,{'1101'}).stock_id.tolist()==['1106','1107']


def test_matched_summary_equal_weights_events_not_number_of_controls():
    from skills.stock_launch import NUMERIC,OBSERVED
    rows=[]
    for case_id,role,values in [('a','case',[1]),('b','case',[1]),('a','control',[0,0,0]),('b','control',[1])]:
        for value in values:
            rows.append(dict(case_id=case_id,role=role,horizon=20,offset=0,
                             **{k:value for k in [*NUMERIC,*OBSERVED]}))
    stats=paired_summary(pd.DataFrame(rows))
    assert all(s['mean_difference']==.5 and s['matched_events']==2 for s in stats)
