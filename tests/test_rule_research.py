import numpy as np
import pandas as pd
import pytest
from skills.rule_research import scores_for,simulate,executable


def panel(values):
    close=pd.DataFrame(values,index=pd.bdate_range('2020-01-01',periods=len(values)),columns=['2330','2317'])
    return close,pd.DataFrame(True,index=close.index,columns=close.columns)


def test_signal_executes_next_session_not_same_close():
    close,flags=panel([[10,10],[10,10],[20,10],[20,10],[20,10]])
    scores=pd.DataFrame(np.nan,index=close.index,columns=close.columns)
    scores.iloc[1,0]=1
    run=simulate(close,flags,scores,start=close.index[2],topn=1,slippage=0)
    assert run.trades[0]['date']==str(close.index[2].date())
    assert run.summary['total_return']<0  # The rise into the execution close was missed.
    assert run.decisions[0]['signal_date']==str(close.index[1].date())


def test_future_prices_do_not_change_past_scores_or_fills():
    days=pd.bdate_range('2018-01-01',periods=400)
    close=pd.DataFrame({'2330':np.linspace(10,20,400),'2317':np.linspace(12,18,400)},index=days)
    turnover=pd.DataFrame(100_000_000.,index=days,columns=close.columns)
    changed=close.copy();changed.iloc[350:]*=10
    a,b=scores_for(close,turnover),scores_for(changed,turnover)
    for key in a: pd.testing.assert_frame_equal(a[key].iloc[:350],b[key].iloc[:350])
    flags=pd.DataFrame(True,index=days,columns=close.columns)
    ra=simulate(close,flags,a['momentum'],start=days[280])
    rb=simulate(changed,flags,b['momentum'],start=days[280])
    pd.testing.assert_frame_equal(ra.curve[ra.curve.date<days[350]],rb.curve[rb.curve.date<days[350]])


def test_unchanged_position_does_not_pay_full_round_trip_each_month():
    close=pd.DataFrame(10.,index=pd.bdate_range('2020-01-01',periods=90),columns=['2330'])
    flags=pd.DataFrame(True,index=close.index,columns=close.columns)
    run=simulate(close,flags,close,start=close.index[1],topn=1,slippage=0)
    assert [t['side'] for t in run.trades]==['buy','sell']
    assert run.summary['total_return']==pytest.approx((1-.004425)/(1+.001425)-1)


def test_blocked_buy_does_not_substitute_future_known_runner_up():
    close,flags=panel([[10,10]]*5)
    scores=close.copy();scores['2330']=20
    flags.iloc[1,0]=False
    run=simulate(close,flags,scores,start=close.index[1],topn=1)
    assert not run.trades
    assert run.summary['blocked_orders']==1
    assert run.summary['total_return']==0


def test_missing_held_quote_is_flagged_not_silently_liquidated():
    close,flags=panel([[10,10],[10,10],[np.nan,10],[np.nan,10]])
    scores=close.copy();scores['2330']=20
    run=simulate(close,flags,scores,start=close.index[1],topn=1)
    assert run.summary['missing_hold_days']==2
    assert run.summary['unliquidated_positions']==1


def test_locked_limit_and_zero_volume_cannot_fill():
    close=pd.DataFrame([[10.,10.,10.]])
    assert executable(close,pd.DataFrame([[100,0,100]]),pd.DataFrame([[10,11,11]]),
                      pd.DataFrame([[10,9,9]])).iloc[0].tolist()==[False,False,True]


def test_missing_benchmark_must_not_be_reported_as_zero_return_cash():
    close,flags=panel([[10,10]]*5)
    flags.iloc[1,0]=False
    with pytest.raises(ValueError,match='Benchmark cannot execute'):
        simulate(close,flags,None,start=close.index[1],benchmark='2330')


def test_blocked_exit_keeps_position_limit_and_retries_without_future_selection():
    close,flags=panel([[10,10]]*40)
    scores=close.copy();scores['2330']=20
    # February wants the other stock, but the existing stock is locked at the first session.
    feb=int(close.index.searchsorted(pd.Timestamp('2020-02-01')))
    scores.iloc[feb-1:,0]=0
    flags.iloc[feb,0]=False
    run=simulate(close,flags,scores,start=close.index[1],topn=1)
    assert run.curve.positions.max()==1
    assert not any(t['side']=='buy' and t['date']==str(close.index[feb].date()) for t in run.trades)
    assert any(t['side']=='sell' and t['date']==str(close.index[feb+1].date()) for t in run.trades)


def test_preparation_uses_shared_gateway_once_per_missing_file(monkeypatch,tmp_path):
    from scripts import research_rules as cli
    from app import config,finmind
    from types import SimpleNamespace
    monkeypatch.setattr(cli,'INPUT_DIR',tmp_path)
    monkeypatch.setattr(config,'load_config',lambda:SimpleNamespace(finmind_token='test-only',finmind_requests_per_hour=5400))
    calls=[]
    def fetch(dataset,start,end,**kwargs):
        calls.append((dataset,start,end,kwargs['data_id']))
        ids=[kwargs['data_id']] if kwargs['data_id'] else [str(i) for i in range(2000,3600)]
        return pd.DataFrame({'stock_id':ids,'date':str(start),'open':10,'max':11,'min':9,
                             'close':10,'Trading_Volume':10000})
    monkeypatch.setattr(finmind,'fetch_dataset',fetch)
    cli.prepare_inputs()
    cli.prepare_inputs()
    assert len(calls)==3
    assert calls[0][3]=='0050'
    assert all(start==end and sid is None for _,start,end,sid in calls[1:])
    assert len(list(tmp_path.glob('*.parquet')))==3
