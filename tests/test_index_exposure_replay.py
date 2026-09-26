from copy import deepcopy
import pandas as pd
import pytest
from skills.index_exposure_replay import IndexExposureReplay,costs,affordable
from skills.index_exposure_audit import audit_index
from skills.index_exposure_inputs import derived_limits

def data(start='2025-02-03',sessions=35):
    days=pd.bdate_range(start,periods=sessions).strftime('%Y-%m-%d').tolist()
    warmup=pd.bdate_range(end=pd.Timestamp(days[0])-pd.Timedelta(days=1),periods=260).strftime('%Y-%m-%d').tolist()
    calendar=warmup+days
    return dict(days=days,calendar=calendar,signals={d:100+i*.01 for i,d in enumerate(calendar)},
        quotes={d:dict(open=100.,close=100.,high=102.,low=98.,volume=10_000_000.,upper=120.,lower=80.) for d in calendar})

def replay(x,window=0,mask=0):
    engine=IndexExposureReplay(x,window,mask);account=engine.run()
    value=dict(account=account,config=dict(window=window,factor_mask=mask),decisions=engine.decisions,
        plans=engine.plans,pending=engine.pending)
    audit_index(value,x)
    return value

@pytest.mark.parametrize('window',[0,180,200,220])
@pytest.mark.parametrize('mask',range(8))
def test_every_fixed_arm_is_causal_and_cash_reconciles(window,mask):
    x=data();value=replay(x,window,mask)
    first=value['account']['trades'][0]
    assert first['date']==x['days'][int(bool(mask&2))]
    assert first['qty']==7000 and first['signal_date']<first['date']
    assert all(t['qty']%1000==0 for t in value['account']['trades'])

def test_etf_sell_tax_and_integer_costs():
    c=costs(101.23,1000,'sell',.0045)
    assert c==dict(gross=101230.,commission=144.,tax=101.,slippage=456.,total_cost=701.,cash_change=100529.)
    assert affordable(1000,100,100500,.0045)==0
    assert costs(.05,1000,'buy',.0045)['commission']==20

def test_derived_limits_round_inside_interval_and_cross_tick_band():
    assert derived_limits(20.14)['upper']==24.16
    assert derived_limits(20.14)['lower']==16.12
    assert derived_limits(49.99)['upper']==59.95
    assert derived_limits(49.99)['lower']==40.
    with pytest.raises(ValueError):derived_limits(float('nan'))

@pytest.mark.parametrize('condition,reason',[
    ('volume','missing_or_zero_quote_volume'),('single','single_price_session'),
    ('limit','at_upper_limit'),('missing_limits','missing_price_limits')])
def test_rejection_is_recorded_and_not_filled(condition,reason):
    x=data(sessions=3);day=x['days'][0];q=x['quotes'][day]
    if condition=='volume':q['volume']=0
    if condition=='single':q['high']=q['low']=q['close']
    if condition=='limit':q['upper']=q['close']
    if condition=='missing_limits':q.pop('lower')
    v=replay(x);assert v['account']['orders'][0]['failure']==reason
    assert not any(t['date']==day for t in v['account']['trades'])
    assert v['account']['trades'][0]['date']==x['days'][1]

def test_partial_fill_replans_with_fresh_prior_signal():
    x=data(sessions=3);x['quotes'][x['days'][0]]['volume']=200_000
    v=replay(x);trades=v['account']['trades']
    assert trades[0]['qty']==2000
    assert trades[1]['signal_date']==x['days'][0]
    assert trades[0]['order_id']!=trades[1]['order_id']

def test_delay_keeps_original_quantity_and_cancels_on_reversal():
    x=data(sessions=5);x['signals'][x['days'][0]]=50
    v=replay(x,200,2)
    assert v['decisions'][1]['cancelled_order']['signal_date']<x['days'][0]
    assert not any(t['date']==x['days'][1] for t in v['account']['trades'])
    x=data(sessions=3);x['quotes'][x['days'][0]]['close']=200
    v=replay(x,0,2)
    assert v['account']['trades'][0]['requested_qty']==7000

def test_latched_exit_survives_rebound_and_never_rebuys_same_day():
    x=data(sessions=6);x['signals'][x['days'][0]]=50
    v=replay(x,200,4);trades=v['account']['trades']
    sells=[t for t in trades if t['side']=='sell'];assert sells[0]['date']==x['days'][2]
    assert sells[0]['reason']=='trend_off'
    assert not any(t['side']=='buy' and t['date']==sells[0]['date'] for t in trades)
    assert any(t['side']=='buy' and t['date']==x['days'][3] for t in trades)

def test_future_quote_and_signal_changes_cannot_change_past_decisions():
    x=data();base=replay(x,200,7);changed=deepcopy(x);cut=x['days'][20]
    for day in x['calendar']:
        if day>cut:
            changed['signals'][day]*=.1
            changed['quotes'][day].update(close=500,high=510,low=490,upper=600,lower=400,volume=200000)
    other=replay(changed,200,7)
    assert [r for r in base['decisions'] if r['date']<=cut]==[r for r in other['decisions'] if r['date']<=cut]
    assert [r for r in base['account']['daily'] if r['date']<=cut]==[r for r in other['account']['daily'] if r['date']<=cut]

def test_split_preserves_equity_converts_delayed_units_and_halt_has_no_fills():
    x=data('2026-03-24',8)
    for q in x['quotes'].values():q.update(open=110.,close=110.,high=112.,low=108.,upper=132.,lower=88.)
    for d in ('2026-03-25','2026-03-26','2026-03-27','2026-03-30'):x['quotes'].pop(d)
    for d in x['days']:
        if d>='2026-03-31':x['quotes'][d].update(close=5.,open=5.,high=5.1,low=4.9,upper=6.,lower=4.)
    v=replay(x,0,2);assert all(t['date'] not in ('2026-03-25','2026-03-26','2026-03-27','2026-03-30') for t in v['account']['trades'])
    split=v['account']['corporate_actions'][0]
    assert split['pending_after']['qty']==22*split['pending_before']['qty']
    assert v['account']['trades'][0]['requested_qty']%22000==0
    assert v['account']['trades'][0]['qty']==100000  # Post-split day-volume capacity still applies.
    assert max(abs(r['daily_return']) for r in v['account']['daily'])<.01

@pytest.mark.parametrize('mutation',['tax','future','shares','cash','missing_decision','inflated_pending','source'])
def test_audit_rejects_corrupted_evidence(mutation):
    x=data(sessions=3);v=replay(x,0,2)
    if mutation=='tax':v['account']['trades'][0]['tax']=300
    if mutation=='future':v['plans'][0]['signal_date']=x['days'][1]
    if mutation=='shares':v['account']['holdings'][0]['qty']+=1000
    if mutation=='cash':v['account']['daily'][-1]['cash']+=1
    if mutation=='missing_decision':v['decisions'].pop(0)
    if mutation=='inflated_pending':v['decisions'][1]['executing_instruction']['qty']+=1000
    if mutation=='source':x['signals'][x['calendar'][0]]=10000;v['config']['window']=220
    with pytest.raises(ValueError):audit_index(v,x)
