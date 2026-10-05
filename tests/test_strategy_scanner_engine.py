import copy

import numpy as np
import pandas as pd
import pytest

from skills.strategy_scanner.engine import ACTIVE_IDS, scan_market


def inputs(n=430):
    days=pd.bdate_range('2024-01-01',periods=n)
    records=[]
    for sid in ('2330','2492','0050'):
        for i,d in enumerate(days):
            c=100+i*.2
            if sid=='2330' and i==n-1: c+=10
            records.append(dict(date=d,stock_id=sid,open=c-.3,high=c+.5,low=c-.6,close=c,
                volume=1_000_000 if i<n-1 else 3_000_000,amount=c*1_000_000,
                adjusted_close=c,quality=True,eligible=True))
    return pd.DataFrame(records),days


def run(bars,days,**kw):
    return scan_market(bars,days,start=str(days[-1].date()),end=str(days[-1].date()),
        provenance={'source_end':str(days[-1].date()),'original_candidates_complete':True},**kw)


def stock(result,sid='2330'):
    return next(s for s in result['days'][-1]['stocks'] if s['stock_id']==sid)


def event(day,sid='2330'):
    return dict(signal_date=str(day.date()),members=[sid],priority=.2)


def test_complete_matrix_and_multiple_strategies_without_slots():
    b,d=inputs();r=run(b,d,original_signals=[event(d[-1])])
    assert len(r['days'][0]['stocks'])==2  # Benchmark remains context only.
    assert sum(r['days'][0]['counts'].values())==2*len(ACTIVE_IDS)
    s=stock(r)
    assert s['results']['original_red']['status']=='matched'
    assert s['results']['donchian20']['status']=='matched'
    assert s['results']['momentum']['status']=='matched'
    assert r['portfolio_model'] is None and r['execution_model'] is None
    assert r['returns_inherited'] is False and r['live_qualified'] is False


def test_future_prices_and_future_new_stock_cannot_change_past():
    b,d=inputs();end=d[-1];events=[event(end)];r=run(b,d,original_signals=events)
    future=end+pd.offsets.BDay();future_rows=b[b.date==end].copy()
    future_rows['date']=future;future_rows['close']=999999
    future_rows.loc[future_rows.stock_id=='2330','stock_id']='9999'
    extended=pd.concat([b,future_rows],ignore_index=True)
    other=scan_market(extended,d.append(pd.DatetimeIndex([future])),start=str(end.date()),end=str(end.date()),
        original_signals=events+[event(future,'9999')],provenance=r['provenance'])
    assert other==r


def test_missing_session_is_unknown_not_compressed_or_no_signal():
    b,d=inputs();b=b[~((b.stock_id=='2330')&(b.date==d[-10]))]
    s=stock(run(b,d))
    assert s['results']['donchian20']['status']=='unknown'
    assert s['results']['momentum']['status']=='unknown'


def test_original_ledger_absence_is_unknown_and_poc_is_not_inferred():
    b,d=inputs()
    r=scan_market(b,d,start=str(d[-1].date()),end=str(d[-1].date()))
    assert stock(r)['results']['original_breakout']['status']=='unknown'
    r=run(b,d,original_signals=[event(d[-1])]);s=stock(r)
    assert s['results']['poc_red_priority']['status']=='matched'
    assert s['results']['poc_red_priority']['metrics']['poc_status']=='not_covered'
    assert s['results']['poc_up_red']['status']=='unknown'
    assert stock(r,'2492')['results']['poc_up_red']['status']=='not_matched'


def test_poc_hard_filter_is_separate_and_rejects_signal_day_data():
    b,d=inputs();p=dict(signal_date=str(d[-1].date()),stock_id='2330',status='up',
        source_date_end=str(d[-2].date()),poc_before=100.,poc_after=110.,available=True,
        prior_dates=[str(x.date()) for x in d[-21:-1]])
    r=run(b,d,original_signals=[event(d[-1])],poc=[p])
    assert stock(r)['results']['poc_up_red']['status']=='matched'
    p['source_date_end']=str(d[-1].date())
    with pytest.raises(ValueError,match='before its signal'): run(b,d,poc=[p])


def test_first_signal_uses_prior_day_even_for_single_day_run():
    b,d=inputs()
    assert stock(run(b,d,original_signals=[event(d[-1])]))['results']['original_red']['first_signal'] is True
    r=run(b,d,original_signals=[event(d[-2]),event(d[-1])])
    assert stock(r)['results']['original_red']['first_signal'] is False
    b.loc[(b.stock_id=='2330')&(b.date==d[-2]),'quality']=False
    r=run(b,d,original_signals=[event(d[-1])])
    assert stock(r)['results']['original_red']['first_signal'] is None


def test_frozen_and_invalid_inputs_fail_explicitly():
    b,d=inputs()
    with pytest.raises(ValueError,match='Duplicate stock'):run(pd.concat([b,b.iloc[:1]]),d)
    with pytest.raises(ValueError,match='Duplicate original'):run(b,d,original_signals=[event(d[-1])]*2)
    with pytest.raises(ValueError,match='registered'):run(b,d,strategies=['not_registered'])
    b['eligible']=b.eligible.astype(object)
    b.loc[0,'eligible']='True'
    with pytest.raises(ValueError,match='booleans'):run(b,d)


def test_identity_exclusion_and_quality_conflict_are_distinct():
    b,d=inputs();b.loc[(b.stock_id=='2330')&(b.date==d[-1]),'eligible']=False
    b.loc[(b.stock_id=='2492')&(b.date==d[-1]),'quality']=False
    r=run(b,d)
    assert {x['status'] for x in stock(r)['results'].values()}=={'ineligible'}
    assert {x['status'] for x in stock(r,'2492')['results'].values()}=={'unknown'}


def test_market_calendar_gap_and_partial_candidate_coverage():
    b,d=inputs();r=scan_market(b,d,start=str(d[-1].date()),end=str(d[-1].date()),
        provenance={'original_candidates_complete':True,'original_signal_start':str((d[-1]+pd.offsets.BDay()).date())})
    assert stock(r)['results']['original_breakout']['status']=='unknown'
    with pytest.raises(ValueError,match='calendar'):
        run(b,d.delete(-10))


def test_previous_ma60_missing_keeps_pullback_unknown():
    b,d=inputs(60)
    assert stock(run(b,d))['results']['ma_pullback']['status']=='unknown'


def test_unknown_nullable_identity_does_not_crash_or_become_eligible():
    b,d=inputs();b['eligible']=b.eligible.astype('boolean')
    b.loc[(b.stock_id=='2330')&(b.date==d[-1]),'eligible']=pd.NA
    b['source_disagreement']=pd.array([pd.NA]*len(b),dtype='boolean')
    r=run(b,d)
    assert {x['status'] for x in stock(r)['results'].values()}=={'unknown'}
