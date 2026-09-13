from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from skills.intraday_limit_replay import (normalize_ticks,match_ticks,limit_price,
    IntradayReplay,IntradayBenchmark,audit_intraday)
from skills.scenario_exit_replay import ExitSignals
from skills.replay_market_feeds import ReplayDataUnavailable
from scripts.research_intraday_limit import TickCache
from test_cash_allocation_replay import fixture,ENTRY,SIZE


def tape(prices=(49.,49.), volumes=(100,900), times=('09:02:00','10:00:00'),sid='1101',day='2022-01-03'):
    return pd.DataFrame(dict(date=[day]*len(prices),stock_id=[sid]*len(prices),
        deal_price=prices,volume=volumes,Time=times,TickType=['0']*len(prices)))


def test_limits_use_stock_and_etf_price_steps():
    assert limit_price(50.08,'1101','buy')==50.0
    assert limit_price(50.08,'1101','sell')==50.1
    assert limit_price(50.08,'0050','buy')==50.05


def test_only_post_submission_strict_trade_through_volume_fills():
    raw=tape(prices=(49,49,50,49,49),volumes=(1000,1000,1000,100,900),
             times=('09:00:00','09:01:00','09:02:00','09:03:00','13:25:00'))
    ticks=normalize_ticks(raw,'1101','2022-01-03','TWSE')
    fill=match_ticks(ticks,'buy',50.,5000,2e6,.01)
    assert fill['filled_qty']==1000 and fill['eligible_shares']==100000
    assert fill['last_fill_time']=='09:03:00'
    assert match_ticks(ticks,'sell',50.,5000,2e6,.01)['filled_qty']==0


def test_partial_and_adv_cap_and_first_threshold_crossing():
    ticks=normalize_ticks(tape(),'1101','2022-01-03','TWSE')
    fill=match_ticks(ticks,'buy',50.,15000,300000.,.01)
    assert fill['filled_qty']==3000 and fill['last_fill_time']=='10:00:00'
    assert match_ticks(ticks,'buy',50.,1000,300000.,.01)['last_fill_time']=='09:02:00'


@pytest.mark.parametrize('change',[{'date':'2022-01-04'},{'stock_id':'1102'},
    {'volume':-1},{'volume':.5},{'deal_price':float('nan')},{'Time':'25:00:00'}])
def test_bad_tick_evidence_blocks(change):
    raw=tape()
    raw['volume']=raw.volume.astype(float)
    for key,value in change.items():raw.loc[0,key]=value
    with pytest.raises(ReplayDataUnavailable):normalize_ticks(raw,'1101','2022-01-03','TWSE')


def test_duplicates_preserved_and_emerging_not_assumed_lots():
    raw=tape(prices=(49,49),volumes=(100,100),times=('10:00:00','10:00:00'))
    assert len(normalize_ticks(raw,'1101','2022-01-03','TWSE'))==2
    with pytest.raises(ReplayDataUnavailable):normalize_ticks(raw,'1101','2022-01-03','EMERGING')


class Ticks:
    def __init__(self,price=49.):self.price=price
    def get(self,sid,day,market):
        raw=tape(prices=(self.price,self.price),sid=sid,day=day)
        return normalize_ticks(raw,sid,day,market),'synthetic'


def execute(ticks=None,**options):
    days,adjusted,args,kwargs=fixture(entries=[ENTRY],end=ENTRY+4,**options)
    ticks=ticks or Ticks()
    engine=IntradayReplay(*args,ticks=ticks,exit_signals=ExitSignals(adjusted,days),**kwargs)
    account=engine.run()
    audit_intraday(account,ticks,engine.markets)
    return engine,account,days


def test_full_account_reconciles_and_idle_cash_stays_cash():
    engine,account,_=execute()
    assert len(account['trades'])==1
    buy=account['trades'][0]
    assert buy['qty']==6000 and buy['reference_price']==50
    assert account['daily'][-1]['nav']==1e6-buy['total_cost']
    assert all(t['channel']=='board' and t['stock_id']!='0050' for t in account['trades'])


def test_future_execution_changes_fill_but_not_frozen_plan():
    _,filled,_=execute()
    _,unfilled,_=execute(Ticks(price=50))
    assert filled['plans']==unfilled['plans']
    assert filled['trades'] and not unfilled['trades']


def test_audit_rejects_tampered_fill_time_and_cash_plan():
    engine,account,_=execute()
    bad=deepcopy(account);bad['orders'][0]['last_fill_time']='09:00:00'
    with pytest.raises(ValueError,match='independent audit'):audit_intraday(bad,Ticks(),engine.markets)
    bad=deepcopy(account);bad['plans'][0]['reserved_cash']=2e6
    with pytest.raises(ValueError,match='later cash'):audit_intraday(bad,Ticks(),engine.markets)


def test_missing_previous_session_price_does_not_use_stale_price():
    def mutate(quotes,days):
        quotes.loc[quotes.stock_id.eq('1101') & quotes.date.eq(days[ENTRY-1]),'close']=np.nan
    _,account,_=execute(mutate=mutate)
    assert not account['trades']


def test_offline_missing_tick_is_error_not_zero_fill(tmp_path):
    cache=TickCache(tmp_path)
    with pytest.raises(ReplayDataUnavailable,match='Offline source missing'):
        cache.get('1101','2022-01-03','TWSE')
    assert cache.calls==0


def test_benchmark_never_double_uses_same_day_tape():
    days,adjusted,args,kwargs=fixture(end=ENTRY+4)
    ticks=Ticks(price=99.)
    engine=IntradayBenchmark(*args,ticks=ticks,**kwargs)
    account=engine.run();audit_intraday(account,ticks,engine.markets)
    assert len(account['trades'])==1
    assert account['trades'][0]['stock_id']=='0050'


def test_benchmark_missing_adv_is_serializable_and_not_filled():
    import json
    days,adjusted,args,kwargs=fixture(end=ENTRY+4)
    engine=IntradayBenchmark(*args,ticks=Ticks(price=99.),**kwargs)
    engine.volume20.loc[:,:]=np.nan
    account=engine.run()
    assert not account['trades']
    assert all(o['failure']=='missing_previous_price_or_adv' for o in account['orders'])
    json.dumps(account,allow_nan=False)


def test_unfilled_early_orders_do_not_release_cash_or_slots_for_later_candidates():
    days,adjusted,args,kwargs=fixture(entries=[ENTRY],end=ENTRY+1)
    quotes,companies,calendar,events,feeds,corporate=args
    for index,sid in enumerate(('1102','1103','1104')):
        extra=quotes[quotes.stock_id.eq('1101')].copy();extra.stock_id=sid
        quotes=pd.concat([quotes,extra],ignore_index=True)
        companies=pd.concat([companies,pd.DataFrame([dict(stock_id=sid,name=sid,market='TWSE')])])
        adjusted[sid]=adjusted['1101']
        events.append(dict(events[0],members=[sid],event_id=sid,priority=-index))
    ticks=Ticks(price=50.)
    engine=IntradayReplay(quotes,companies,calendar,events,feeds,corporate,
        ticks=ticks,exit_signals=ExitSignals(adjusted,days),**kwargs)
    account=engine.run();audit_intraday(account,ticks,engine.markets)
    assert not account['trades']
    assert len([p for p in account['plans'] if p['planned_qty']])==3
    assert account['plans'][-1]['stock_id']=='1104' and account['plans'][-1]['planned_qty']==0
    assert not any(o['stock_id']=='1104' for o in account['orders'])


def test_limit_exit_is_latched_but_cannot_fill_below_sell_price():
    stock=np.full(SIZE,100.);stock[ENTRY+1:]=87.
    engine,account,days=execute(stock=stock)
    sales=[r for r in account['orders'] if r['side']=='sell']
    assert sales and all(r['filled_qty']==0 for r in sales)
    assert sales[0]['date']==str(days[ENTRY+2].date())
    assert all(r['signal_date']==str(days[ENTRY+1].date()) for r in sales)
    assert account['daily'][-1]['market_value']==300000


def test_cache_budget_is_durable_and_cannot_retry_around_limit(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import scripts.research_intraday_limit as driver
    monkeypatch.setattr(driver,'load_config',lambda:SimpleNamespace(finmind_token='test-only'))
    calls=[]
    def fetch(dataset,day,**kwargs):
        calls.append((dataset,day,kwargs))
        return tape(sid=kwargs['data_id'],day=day.isoformat())
    monkeypatch.setattr(driver,'fetch_dataset',fetch)
    cache=TickCache(tmp_path,online=True,maximum=1)
    cache.get('1101','2022-01-03','TWSE')
    cache=TickCache(tmp_path,online=True,maximum=1)
    cache.get('1101','2022-01-03','TWSE')
    with pytest.raises(ReplayDataUnavailable,match='ceiling'):
        cache.get('1101','2022-01-04','TWSE')
    assert len(calls)==1 and calls[0][2]['max_retries']==0
