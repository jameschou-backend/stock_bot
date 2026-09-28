import numpy as np
import pandas as pd
import pytest
from skills.rally_recall import signals, labels, sample, legacy_barrier


def inputs():
    days=pd.bdate_range('2021-01-04',periods=300)
    c=pd.DataFrame({'0050':np.linspace(100,120,300),'1101':np.linspace(100,150,300)},index=days)
    c.loc[days[160],'1101']*=1.05
    v=pd.DataFrame(1_000_000.,index=days,columns=c.columns);v.loc[days[160],'1101']*=2
    eligible=pd.DataFrame(True,index=days,columns=c.columns)
    companies=pd.DataFrame([dict(stock_id='1101',listed_date=days[0])])
    return c,c.copy(),c.copy(),v,eligible,companies


def test_new_rules_are_causal_and_identity_blocks_same_day():
    args=inputs();full=signals(*args);day=args[0].index[160]
    assert full['breakout_unrestricted'].at[day,'1101']
    assert full['first_expansion'].at[day,'1101']
    partial=signals(*(f.loc[:day] for f in args[:5]),args[5])
    mutated=[f.copy() for f in args[:5]]
    for f in mutated[:4]:f.loc[f.index>day]*=2
    changed=signals(*mutated,args[5])
    for k in ('eligible','breakout_unrestricted','first_expansion'):
        pd.testing.assert_frame_equal(full[k].loc[:day],partial[k])
        pd.testing.assert_frame_equal(full[k].loc[:day],changed[k].loc[:day])
    args[4].at[day,'1101']=False
    assert not signals(*args)['breakout_unrestricted'].at[day,'1101']


def test_uses_next_open_not_signal_close_or_same_day_open():
    c,q,raw,*_=inputs();op=raw.copy();i=150
    op.iloc[i+1,1]=raw.iloc[i+1,1]*1.05
    result,known=labels(c,q,raw,op,raw)
    assert known.iloc[i,1]
    assert result.iloc[i,1]==pytest.approx(c.iloc[i+64,1]/op.iloc[i+1,1]-1)
    assert result.iloc[-1].isna().all()


@pytest.mark.parametrize('kind',['open','conflict','future_gap','future_anomaly','disagreement'])
def test_missing_or_disputed_labels_are_unknown(kind):
    c,q,raw,*_=inputs();op=raw.copy();qc=raw.copy();i=150
    if kind=='open':op.iloc[i+1,1]=np.nan
    if kind=='conflict':qc.iloc[i+1,1]+=1
    if kind=='future_gap':q.iloc[i+30,1]=np.nan
    if kind=='future_anomaly':q.iloc[i+30,1]*=2
    if kind=='disagreement':q.iloc[i+2:i+65,1]*=np.linspace(1,1.2,63)
    result,known=labels(c,q,raw,op,qc)
    assert not known.iloc[i,1] and pd.isna(result.iloc[i,1])


def test_nonoverlap_cooldown_does_not_use_outcomes():
    c,*_=inputs();mask=c.gt(0);mask['0050']=False
    rows=sample(mask,start='2021-01-01',end='2023-01-01')
    assert [i for i,sid in rows]==[0,64,128,192,256]


def test_exact_legacy_barriers_distinguish_group_reuse_and_order_rejection():
    group=dict(month='2025-01',selected_ids=['1101'],clusters=[dict(group_id='g',members=['1101'])],
               exclusions={},leader_rejections=[])
    event=dict(group_id='g',leader_id='1102',leader_date='2025-01-02')
    assert legacy_barrier('1101','2025-01-03',[group],[event],[],[],[])=='group_already_used:1102'
    entry=dict(members=['1101'],signal_date='2025-01-03',event_id='x')
    plan=dict(event_id='x',side='buy',rejection='opening_slots_locked')
    assert legacy_barrier('1101','2025-01-03',[group],[],[entry],[],[plan])=='opening_slots_locked'
    assert legacy_barrier('1103','2025-01-03',[group],[],[],[],[])=='outside_monthly_top300'
