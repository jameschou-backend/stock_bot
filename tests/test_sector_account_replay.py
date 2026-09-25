"""Synthetic fixed-anchor signals and complete accounts; no external requests."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from skills import sector_account_replay as research
from skills.scenario_exit_replay import ExitSignals
from skills.backtest_contract import validate_comparison
from skills.replay_market_feeds import ReplayDataUnavailable


def frames(n=215):
    days = pd.bdate_range('2021-01-04', periods=n)
    ids = ['1101','1102','1103','1104','1201','1202','1203','1204']
    close = pd.DataFrame({sid: 50.*1.008**np.arange(n) for sid in ids}, index=days)
    close['0050'] = 100.
    raw = close*0+50.; raw['0050']=100.
    volume = close*0+2_000_000.
    volume.loc[days[147]:days[151],ids[:4]] *= 3
    companies = pd.DataFrame([dict(stock_id=sid,name=sid,industry='group',market='TWSE',listed_date='2020-01-01') for sid in ids])
    members = pd.DataFrame([dict(stock_id=sid,industry='A' if sid<'1200' else 'B') for sid in ids])
    return days,close,raw,volume,companies,members


def signals(frames_tuple=None,end=210):
    days,close,raw,volume,companies,members = frames_tuple or frames()
    result = research.build_signals(close,raw,volume,companies,members,
        membership_snapshot_date='2026-09-09',start=str(days[130].date()),end=str(days[end].date()))
    return result, (days,close,raw,volume,companies,members)


class Feeds:
    offline = True
    def __init__(self,quotes, *, missing_day=None, lower_day=None):
        self.quotes=quotes.set_index(['date','stock_id'])
        self.days=sorted(quotes.date.unique())
        self.missing_day,self.lower_day=missing_day,lower_day
    def get_limits(self,sid):
        return {str(pd.Timestamp(day).date()):dict(upper=1e9,lower=
            float(self.quotes.loc[(day,sid),'close']) if (sid,pd.Timestamp(day))==self.lower_day else .001)
            for day in self.days if (sid,pd.Timestamp(day))!=self.missing_day}
    def get_odd(self,day,sid,market):
        price=float(self.quotes.loc[(pd.Timestamp(day),sid),'close'])
        return dict(odd_shares=100_000,odd_last=price,odd_bid=price-.01,odd_ask=price+.01,bid_qty=100_000,ask_qty=100_000)


class Corporate:
    def __init__(self,actions=None,fail=None):
        self.actions=actions or {}; self.fail=fail
    def prepare(self,sid):
        if sid==self.fail:
            raise ValueError('Frozen dividend source missing: '+sid)
    def on_date(self,sid,day):
        self.prepare(sid)
        return deepcopy(self.actions.get((sid,day),[]))


def account_data(end=210, stock_path=None):
    generated, values = signals(end=end)
    days,close,raw,volume,companies,_ = values
    if stock_path is not None:
        close['1101'] = stock_path
    # Isolate account logic with one frozen, fully documented event per arm.
    event = next(e for e in generated['entries_by_arm']['relative_strength'] if e['stock_id']=='1101')
    entries = {arm:[deepcopy(event)] for arm in research.ARMS}
    quotes = pd.DataFrame([dict(date=day,stock_id=sid,open=price,high=price+1,low=price-1,
        close=price,volume=2_000_000) for day in days for sid,price in [('0050',100.),('1101',50.)]])
    empty = pd.DataFrame(columns=['stock_id','event_date'])
    data=research.SectorAccountInputs(quotes,companies,days,entries,empty,
        ExitSignals(close[['0050','1101']],days),'2026-09-09',str(days[130].date()),str(days[end].date()))
    return data


def configuration(*,arm='relative_strength',board=False,stress='control'):
    return next(c for _,c in research.configurations() if c['arm']==arm and c['board_only']==board and c['stress']==stress)


def case(data, *, config=None, **feed_kwargs):
    return research.run_case(data,config or configuration(),'.',{},
        feeds=Feeds(data.quotes,**feed_kwargs),corporate=Corporate())


def test_signals_include_immature_last_anchor_without_reading_future_labels():
    result,values=signals()
    days=values[0]
    assert result['anchors']==[str(days[i].date()) for i in (130,151,172,193)]
    assert result['future_labels_used'] is False
    entries=result['entries_by_arm']
    assert any(e['signal_date']==str(days[193].date()) for e in entries['relative_strength'])
    assert all('event' not in e and 'forward_return' not in e and 'label_known' not in e for e in entries['relative_strength'])
    assert {e['event_id'] for e in entries['strength_with_turnover']} <= {e['event_id'] for e in entries['relative_strength']}
    assert any(e['signal_date']==str(days[151].date()) for e in entries['strength_with_turnover'])
    assert research.validate_entries(entries['relative_strength'],days)['next_market_day']


def test_unknown_peers_are_excluded_from_both_arms_not_converted_to_false():
    values=list(frames());values[1]['1104']=np.nan
    result,_=signals(tuple(values))
    assert any(row['turnover_unknown'] for row in result['coverage'])
    for arm in research.ARMS:
        assert not any(e['stock_id'].startswith('11') for e in result['entries_by_arm'][arm])


def test_cross_group_or_keeps_true_before_unknown_and_unknown_before_false():
    assert research._or(pd.Series([True,pd.NA],dtype='boolean')) is True
    assert research._or(pd.Series([False,pd.NA],dtype='boolean')) is None
    assert research._or(pd.Series([False,False],dtype='boolean')) is False


def test_future_mutation_and_truncation_do_not_change_existing_candidates():
    original,values=signals()
    cutoff=values[0][180]
    for truncate in (False,True):
        changed=list(values)
        for i in (1,2,3):
            changed[i]=values[i].loc[:cutoff].copy() if truncate else values[i].copy()
            if not truncate:
                changed[i].loc[changed[i].index>cutoff]*=7
        changed[0]=changed[1].index
        actual=research.build_signals(*changed[1:],membership_snapshot_date='2026-09-09',
            start=str(values[0][130].date()),end=str(cutoff.date()) if truncate else str(values[0][210].date()))
        for arm in research.ARMS:
            known=lambda e:e['entry_date']<=str(cutoff.date())
            assert list(filter(known,actual['entries_by_arm'][arm]))==list(filter(known,original['entries_by_arm'][arm]))


def test_membership_duplicates_do_not_change_candidates_and_metadata_is_never_pit():
    original,values=signals()
    changed=list(values);changed[5]=pd.concat([values[5],values[5]])
    actual,_=signals(tuple(changed))
    assert actual==original
    for e in actual['entries_by_arm']['relative_strength']:
        assert e['membership_point_in_time'] is False
        assert e['membership_snapshot_date']=='2026-09-09'
        assert e['group_cutoff_date_meaning']=='price_and_turnover_feature_cutoff_only'


@pytest.mark.parametrize('config',[c for _,c in research.configurations()])
def test_all_twelve_configurations_reconcile_cash_shares_costs_and_dates(config):
    data=account_data()
    result=case(data,config=config)
    assert result['completed']
    assert result['membership_point_in_time'] is result['live_qualified'] is result['unseen_validation'] is False
    assert result['audit']['full_calendar'] and result['audit']['fees_recomputed']
    assert result['account']['settings']['initial_cash']==1_000_000
    if config['board_only']:
        assert all(t['qty']%1000==0 for t in result['account']['trades'])
        assert result['audit']['final_fill_decisions_exact']
    if not config['benchmark']:
        assert not any(t['stock_id']=='0050' for t in result['account']['trades'])
        assert result['account']['settings']['slots']==5
        buys=[t for t in result['account']['trades'] if t['side']=='buy']
        expected=data.days[132 if config['stress']=='combined' else 131]
        assert {t['date'] for t in buys}=={str(expected.date())}
        assert result['audit']['slot_membership_independently_reconstructed']


@pytest.mark.parametrize('board',[False,True])
def test_matched_benchmark_has_same_capital_calendar_cost_policy(board):
    data=account_data()
    strategy=case(data,config=configuration(board=board,stress='combined'))
    benchmark=case(data,config=configuration(arm='benchmark',board=board,stress='combined'))
    assert all(validate_comparison(strategy,benchmark).values())


@pytest.mark.parametrize('board',[False,True])
def test_missing_date_limits_blocks_instead_of_publishing_a_skipped_trade_return(board):
    data=account_data()
    result=case(data,config=configuration(board=board),missing_day=('1101',data.days[131]))
    assert result['completed'] is False
    assert 'price-limit date: 1101' in result['reason']
    assert 'summary' not in result and 'partial_account' in result


def test_missing_dividend_and_raw_quote_explicitly_block():
    data=account_data()
    missing_dividend=research.run_case(data,configuration(),'.',{},feeds=Feeds(data.quotes),corporate=Corporate(fail='1101'))
    assert not missing_dividend['completed'] and 'Frozen dividend source' in missing_dividend['reason']
    quotes=data.quotes.copy()
    quotes.loc[quotes.stock_id.eq('1101') & quotes.date.eq(data.days[131]),'high']=np.nan
    missing_quote=case(replace(data,quotes=quotes))
    assert not missing_quote['completed'] and 'Raw execution quote missing' in missing_quote['reason']


def test_valid_lower_limit_is_a_real_execution_rejection_not_missing_data():
    data=account_data(end=195)
    # Fixed63 triggers 131+63=194. A real lower-limit observation defers sale.
    result=case(data,lower_day=('1101',data.days[194]))
    assert result['completed']
    sales=[t for t in result['account']['trades'] if t['side']=='sell']
    assert {t['date'] for t in sales}=={str(data.days[195].date())}


def test_signal_contract_rejects_same_day_entry_and_pit_promotion():
    data=account_data();event=deepcopy(data.entries_by_arm['relative_strength'][0])
    event['entry_date']=event['signal_date']
    with pytest.raises(ValueError,match='timing'):
        research.validate_entries([event],data.days)
    event=deepcopy(data.entries_by_arm['relative_strength'][0]);event['membership_point_in_time']=True
    with pytest.raises(ValueError,match='PIT'):
        research.validate_entries([event],data.days)


def test_combined_priority_is_signal_adv_not_execution_day_adv():
    data=account_data()
    event=deepcopy(data.entries_by_arm['relative_strength'][0]);event['stock_id']='1201';event['members']=['1201'];event['event_id']='other';event['priority']*=.5
    quotes=pd.concat([data.quotes,data.quotes[data.quotes.stock_id.eq('1101')].assign(stock_id='1201',volume=100_000_000)])
    close=data.features.adjusted_close.copy();close['1201']=close['1101']
    entries={arm:[*data.entries_by_arm[arm],deepcopy(event)] for arm in research.ARMS}
    changed=replace(data,quotes=quotes,entries_by_arm=entries,features=ExitSignals(close,data.days))
    result=case(changed,config=configuration(stress='combined'))
    assert result['completed']
    assert [c['stock_id'] for c in result['account']['cohorts']]==['1101','1201']


def test_combined_terminal_candidate_remains_in_signals_but_is_explicitly_unscheduled():
    data=account_data()
    last=len(data.days)-1
    event=deepcopy(data.entries_by_arm['relative_strength'][0])
    event.update(event_id='terminal',signal_date=str(data.days[last-1].date()),
                 entry_date=str(data.days[last].date()),feature_cutoff_date=str(data.days[last-1].date()),
                 group_cutoff_date=str(data.days[last-1].date()))
    for key in ('liquidity_at_signal','liquidity_before_entry'):
        event[key]['as_of']=event['signal_date']
    changed=replace(data,end=str(data.days[last].date()),entries_by_arm={arm:[deepcopy(event)] for arm in research.ARMS})
    result=case(changed,config=configuration(stress='combined'))
    assert result['completed'] and result['candidate_count']==result['signal_contract']['signal_count']==1
    assert result['schedule_decisions'][0]['reason']=='scheduled_outside_account_window'
    assert result['schedule_decisions'][0]['execution_date'] is None
    assert not result['account']['trades']
    assert changed.entries_by_arm['relative_strength']==[event]


@pytest.mark.parametrize('board',[False,True])
def test_delayed_corporate_preparation_preserves_complete_account_exactly(board):
    data=account_data()
    config=configuration(board=board)
    actual=case(data,config=config)
    from skills.conservative_diversification import ConservativeDiversification
    from skills.board_only_verified_replay import BoardOnlyVerifiedReplay
    cls=BoardOnlyVerifiedReplay if board else ConservativeDiversification
    kwargs={} if board else {'position_count':5}
    engine=cls(data.quotes,data.companies,data.days,data.entries_by_arm['relative_strength'],
        Feeds(data.quotes),Corporate(),start=data.start,end=data.end,exit_signals=data.features,
        action_dates=[],stress_mode='control',**kwargs)
    assert actual['account']==engine.run()


def test_deferred_preparation_never_fetches_unfunded_or_below_board_lot_candidates():
    data=account_data()
    quotes=data.quotes.copy()
    for field,price in [('open',400.),('high',401.),('low',399.),('close',400.)]:
        quotes.loc[quotes.stock_id.eq('1101'),field]=price
    data=replace(data,quotes=quotes)
    # Five-slot budget is 200k, insufficient for one NT$400 board lot. The
    # provider deliberately lacks this candidate; no executable order needs it.
    result=research.run_case(data,configuration(board=True),'.',{},feeds=Feeds(quotes),corporate=Corporate(fail='1101'))
    assert result['completed'] and not result['account']['trades']
    assert any(r['failure']=='board_only_below_one_lot' for r in result['board_decisions'])


@pytest.fixture
def driver_sandbox(tmp_path,monkeypatch):
    from scripts import research_sector_accounts as driver
    from skills import verified_backtest_tool
    cache=tmp_path/'.cache/inputs'
    (cache/'execution-feeds').mkdir(parents=True);(cache/'dividends').mkdir()
    driver.write(cache/'execution-feeds/index.json',dict(entries={}))
    source=tmp_path/'source.txt';source.write_text('frozen')
    identity=dict(source_sha256={'source.txt':driver.sha(source)})
    data=account_data()
    signal=dict(entries_by_arm=data.entries_by_arm,membership_metadata={'retrieved_at':1})
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    monkeypatch.setattr(verified_backtest_tool,'ROOT',tmp_path)
    monkeypatch.setattr(driver,'sources',lambda *a:(deepcopy(identity),{}))
    monkeypatch.setattr(driver,'load_inputs',lambda:(data,deepcopy(signal)))
    monkeypatch.setattr(driver,'run_case',lambda data,config,*a:research.run_case(
        data,config,cache,{},feeds=Feeds(data.quotes),corporate=Corporate()))
    return driver,cache,tmp_path/'.cache/results',source


def test_driver_seals_all_twelve_cases_and_replays_them_exactly(driver_sandbox):
    driver,cache,output,_=driver_sandbox
    report=driver.run(output,cache)
    assert report['completed'] and len(report['case_rows'])==12 and len(report['comparisons'])==8
    assert report['membership_point_in_time'] is False
    assert report['metrics']['finmind_requests']==0
    second=driver.run(output,cache,offline_replay=True)
    assert second['case_rows']==report['case_rows']
    assert driver.read(output/'offline.json')['all_cases_identical']
    with pytest.raises(ValueError,match='Output exists'):
        driver.run(output,cache)


def test_strict_pit_blocks_without_executing_an_account(driver_sandbox,monkeypatch):
    driver,cache,output,_=driver_sandbox
    def forbidden(*a):
        raise AssertionError('Strict PIT must not execute the retrospective strategy')
    monkeypatch.setattr(driver,'run_case',forbidden)
    report=driver.run(output,cache,strict_pit=True)
    assert not report['completed'] and report['status']=='blocked'
    assert all(row['total_return'] is None and 'strict PIT blocked' in row['reason'] for row in report['case_rows'])


def test_driver_source_mutation_prevents_publishing_a_complete_report(driver_sandbox,monkeypatch):
    driver,cache,output,source=driver_sandbox
    original=driver.run_case
    def mutate(*args):
        value=original(*args);source.write_text('changed');return value
    monkeypatch.setattr(driver,'run_case',mutate)
    with pytest.raises(ValueError,match='Sources changed'):
        driver.run(output,cache)
    assert not (output/'manifest.json').exists()
