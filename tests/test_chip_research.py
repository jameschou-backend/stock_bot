from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skills.chip_research import (holder_rows, broker_concentration, ChipSignals,
                                  ChipReplay, tri_and)
from skills.technical_replay import TechnicalReplay
from skills.technical_signals import TechnicalSignals
from scripts.replay_million import audit
from test_technical_replay import fixture, Signals, ENTRY, SIZE


LEVELS = ['1-999','1,000-5,000','5,001-10,000','10,001-15,000',
    '15,001-20,000','20,001-30,000','30,001-40,000','40,001-50,000',
    '50,001-100,000','100,001-200,000','200,001-400,000',
    '400,001-600,000','600,001-800,000','800,001-1,000,000','more than 1,000,001']


def holding_fixture():
    return pd.DataFrame([dict(date='2026-01-02',stock_id='2330',HoldingSharesLevel=level,
        percent=50 if level in (LEVELS[0],LEVELS[-1]) else 0) for level in LEVELS]
        + [dict(date='2026-01-02',stock_id='2330',HoldingSharesLevel='total',percent=100),
           dict(date='2026-01-02',stock_id='2330',HoldingSharesLevel='差異數調整（說明4）',percent=0)])


def test_holding_total_not_double_counted_and_upper_bucket_recognized():
    row = holder_rows(holding_fixture()).iloc[0]
    assert row.valid and row.large_pct == 50


@pytest.mark.parametrize('corruption', ['duplicate','missing','nan','unknown','bad_sum'])
def test_holding_bad_observation_is_unknown(corruption):
    raw=holding_fixture()
    if corruption=='duplicate': raw=pd.concat([raw,raw.iloc[[0]]])
    elif corruption=='missing': raw=raw.iloc[1:]
    elif corruption=='nan': raw.loc[0,'percent']=np.nan
    elif corruption=='unknown': raw.loc[0,'HoldingSharesLevel']='unmapped'
    else: raw.loc[0,'percent']=60
    result=holder_rows(raw).iloc[0]
    assert not result.valid and np.isnan(result.large_pct)


def test_broker_aggregates_distinct_branches_before_top5():
    raw=pd.DataFrame([dict(securities_trader_id='a',buy=40,sell=0)]*2
        +[dict(securities_trader_id=s,buy=10,sell=0) for s in 'bcdef']
        +[dict(securities_trader_id='seller',buy=0,sell=130)])
    assert broker_concentration(raw)==pytest.approx(120/130)
    raw.loc[len(raw)-1,'sell']=150
    assert broker_concentration(raw) is None


def test_unknown_combination_stays_unknown():
    assert tri_and(False,None) is None
    assert tri_and(True,False) is False


class ChipStub:
    def __init__(self,days,trigger=None,value=True):
        self.days,self.trigger,self.value=days,trigger,value
    def context(self,index,sid,event_id=None):
        return dict(signal_date=str(self.days[index-1].date()),
                    selling=self.trigger is not None and index-1>=self.trigger,
                    trust=self.value)


def run_chip(mode='control',trigger=None,value=True,**options):
    days,adjusted,args,kwargs=fixture(entries=[ENTRY],**options)
    replay=ChipReplay(*args,technical_signals=Signals(adjusted,days),
        chip_signals=ChipStub(days,trigger,value),chip_mode=mode,**kwargs)
    result=replay.run()
    audit(result)
    return replay,result,days


def test_chip_control_exact_original_account():
    days,adjusted,args,kwargs=fixture(entries=[ENTRY])
    expected=TechnicalReplay(*args,technical_signals=Signals(adjusted,days),mode='control',**kwargs).run()
    _,actual,_=run_chip()
    assert actual==expected


@pytest.mark.parametrize('value',[False,None])
def test_filter_blocks_before_etf_funding(value):
    engine,account,_=run_chip('trust',value=value)
    assert not account['cohorts']
    assert not any(t['reason']=='fund_stock' for t in account['trades'])


def test_half_exit_once_and_next_session_with_original_fees():
    engine,account,days=run_chip('sell_half',trigger=ENTRY+1)
    bought=sum(t['qty'] for t in account['trades'] if t['reason']=='leader_entry')
    sales=[t for t in account['trades'] if t['reason']=='chip_sell_half']
    assert sum(t['qty'] for t in sales)==bought//2
    assert {t['date'] for t in sales}=={str(days[ENTRY+2].date())}
    assert all(t['total_cost']>0 and t['signal_date']==str(days[ENTRY+1].date()) for t in sales)
    assert len(engine.half_states)==1


def test_stop_has_priority_over_half_reduction():
    stock=np.full(SIZE,100.);stock[ENTRY+1:]=80.
    _,account,_=run_chip('sell_half',trigger=ENTRY+1,stock=stock)
    assert not any(t['reason']=='chip_sell_half' for t in account['trades'])
    assert any(t['side']=='sell' and t['stock_id']=='1101' for t in account['trades'])


def test_full_exit_next_day():
    _,account,days=run_chip('sell_full',trigger=ENTRY+1)
    sales=[t for t in account['trades'] if t['reason']=='chip_sell_full']
    assert sales and {t['date'] for t in sales}=={str(days[ENTRY+2].date())}


def test_half_unfilled_portion_retries_without_trigger_reset():
    days,_,_,_=fixture()
    engine,account,_=run_chip('sell_half',trigger=ENTRY+1,blocked=[('1101',days[ENTRY+2])])
    sales=[t for t in account['trades'] if t['reason']=='chip_sell_half']
    assert sales and min(t['date'] for t in sales)>str(days[ENTRY+2].date())
    assert all(t['signal_date']==str(days[ENTRY+1].date()) for t in sales)
    assert all(s['remaining']==0 for s in engine.half_states.values())


def test_context_never_reads_execution_day_or_future_holder_week():
    days=pd.bdate_range('2026-01-01',periods=40)
    signals=object.__new__(ChipSignals)
    signals.days=days
    columns=['2330']
    signals.matrices={k:pd.DataFrame(1.,index=days,columns=columns)
        for k in ['trust','foreign','price','margin','selling','sbl']}
    signals.brokers={}
    signals.holders={'2330':pd.DataFrame(dict(date=[days[5],days[20]],delta4=[1.,-1.]))}
    signals.events={}
    prior=signals.context(20,'2330')
    assert prior['holder'] is True
    for matrix in signals.matrices.values():matrix.iloc[20:]=0.
    assert signals.context(20,'2330')==prior
    assert signals.context(21,'2330')['trust'] is False


def test_acceleration_uses_nonoverlapping_complete_windows(tmp_path):
    days,adjusted,args,_=fixture(entries=[ENTRY])
    technical=TechnicalSignals(adjusted,args[0],days)
    frame=pd.DataFrame(dict(date=days,stock_id='1101',trust_net=1000.,foreign_net=0.))
    frame.loc[ENTRY-5:ENTRY-1,'trust_net']=20000.
    frame.loc[ENTRY:,'trust_net']=-2_000_000.
    frame.to_parquet(tmp_path/'institutional_verified.parquet',index=False)
    pd.DataFrame(dict(date=days,stock_id='1101',margin_purchase_balance=100.)).to_parquet(tmp_path/'margin_verified.parquet',index=False)
    data=SimpleNamespace(days=days,features=technical,entries=[])
    signals=ChipSignals(data,tmp_path)
    context=signals.context(ENTRY,'1101')
    assert context['trust'] is True
    assert context['trust_ratio5']==pytest.approx(.01)
    assert context['trust_ratio20_prior']==pytest.approx(.0005)
    assert signals.context(ENTRY+1,'1101')['trust'] is False
    frame.loc[ENTRY-7,'trust_net']=np.nan
    frame.to_parquet(tmp_path/'institutional_verified.parquet',index=False)
    assert ChipSignals(data,tmp_path).context(ENTRY,'1101')['trust'] is None
