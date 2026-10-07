"""Independent candidate, availability and funded exit clocks for eight account arms."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib

import pandas as pd
import pytest

from scripts import research_strategy_account_comparison as study
from scripts import research_poc_range_first_account as frozen
from skills.poc_range_execution import RangeGapOrders
from skills.scenario_exit_replay import ScenarioExitReplay, ExitSignals
from skills.million_replay import Replay
from test_volume_profile_selection import Planner, events


def test_eight_arms_keep_shared_price_and_sizing_rules_and_existing_seals():
    assert set(study.ARMS) == {'red_original','poc_priority','poc_filter','red_known',
        'poc_priority_known','rsi_shared_exit','rsi_time20','benchmark'}
    assert all(row[1:] == ('none',.7,.3,False) for row in study.ARM_RULES.values())
    assert study.ARM_RULES['poc_priority'] == frozen.ARM_RULES['poc_range70_30_all']
    assert study.SELECTION_ARMS['poc_priority'] == 'poc_priority_available'
    assert not any(study.SELECTION_ARMS[a].endswith('_available') for a in study.COMMON_KNOWN_ARMS)
    for name,digest in study.FROZEN_BINDINGS.items():
        assert hashlib.sha256((study.ROOT/name).read_bytes()).hexdigest() == digest
    for name,digest in study.RANGE_BINDINGS.items():
        assert hashlib.sha256((study.ROOT/name).read_bytes()).hexdigest() == digest


def test_red_family_keeps_every_red_event_and_existing_anchor_gate_exact():
    source=events(4)
    source[3].update(signal_date='2024-01-03',entry_date='2024-01-04')
    signals=SimpleNamespace(filter_entries=lambda rows:([r for r in rows if r['event_id']!='e1'],
        [{'event_id':r['event_id'],'passed':r['event_id']!='e1'} for r in rows]))
    old=frozen.select_arm_entries('poc_range70_30_all',source,signals,study.START,study.END)
    for arm in ('red_original','poc_priority','poc_filter','red_known','poc_priority_known'):
        assert study.select_arm_entries(arm,source,signals,study.START,study.END)==old
    assert [e['event_id'] for e in old[0]]==['e0','e2','e3']


def test_rsi_family_is_not_red_filtered_or_re_first_filtered_and_union_keeps_new_stocks():
    source=events(1);rsi=[dict(events(1)[0],event_id='rsi-9999',members=['9999'])]
    def fail(rows):raise AssertionError('RSI must not use the red gate')
    for arm in study.RSI_ARMS:
        result,decisions,first=study.select_arm_entries(arm,source,SimpleNamespace(filter_entries=fail),
            study.START,study.END,rsi_entries=rsi)
        assert result==rsi and decisions==first==[]
        result[0]['priority']=-1
        assert rsi[0]['priority']==1.
    merged=study.merge_candidate_families(source,rsi)
    assert {e['members'][0] for e in merged}=={'1000','9999'}
    with pytest.raises(ValueError,match='event ID'):
        study.merge_candidate_families(source,source)


def test_calendar_rejects_same_day_delayed_entry_etf_or_unknown_session():
    days=pd.bdate_range('2024-01-02',periods=5)
    source=events(1);study.validate_candidate_calendar(source,days)
    for update in ({'entry_date':'2024-01-02'},{'entry_date':'2024-01-04'},
                   {'signal_date':'2024-01-01'},{'members':['0050']},
                   {'priority':float('nan')}):
        with pytest.raises(ValueError):study.validate_candidate_calendar([dict(source[0],**update)],days)


def profile_source(flags,calls):
    def provider(arm,event):
        assert arm=='poc_red_executable'
        eid=event['event_id'];calls.append(eid);flag=flags[eid]
        return (dict(available=False,poc_up=None,reason='ordinary_tape_conflict',recoverable=False)
                if flag is None else dict(available=True,poc_up=flag))
    return provider


def selection(arm,source,shared,*,slots=3):
    planner=Planner(slots=slots);before=planner.snapshot()
    result=study.select_known_profiles(source,arm=study.SELECTION_ARMS[arm],
        provider=lambda e:study.profile_for_arm(arm,shared,e),planner=planner,
        planner_factory=lambda _:Planner(slots=slots))
    assert planner.snapshot()==before
    return result


def test_shared_known_definition_keeps_false_in_original_but_never_hard_filter():
    calls=[];shared=study.MemoizedProfiles(profile_source(dict(e0=False,e1=None,e2=True,e3=False,e4=True),calls))
    source=events(5)
    original=selection('red_known',source,shared)
    priority=selection('poc_priority_known',source,shared)
    hard=selection('poc_filter',source,shared)
    assert [e['event_id'] for e in original['events']]==['e0','e2','e3']
    assert [e['event_id'] for e in priority['events']]==['e2','e4','e0']
    assert [e['event_id'] for e in hard['events']]==['e2','e4']
    assert calls==['e0','e1','e2','e3','e4']
    assert shared.values['e0']['poc_up'] is False  # red_known mapping never corrupts shared evidence.
    for result in (original,priority,hard):
        assert [r['event_id'] for r in result['certificate']['profile_data_exclusions']]==['e1']
        assert result['certificate']['profile_data_exclusions'][0]['poc_up'] is None
        assert result['certificate']['source_planner_unchanged']


def test_lazy_full_slots_do_not_fetch_or_disqualify_the_unqueried_suffix():
    calls=[];shared=study.MemoizedProfiles(profile_source(dict(e0=True),calls))
    result=selection('poc_filter',events(3),shared,slots=1)
    assert calls==['e0']
    assert result['certificate']['profile_data_exclusions']==[]
    assert [r['profile_status'] for r in result['certificate']['decisions'][1:]]==[
        'not_needed_resource_exhausted','not_needed_resource_exhausted']


@pytest.mark.parametrize('reason',['not_requested','raw_tape_unavailable','quota_paused','request_budget_paused','provider_error'])
def test_unacquired_profile_is_incomplete_not_a_false_or_skipped_candidate(reason):
    provider=lambda *args:dict(available=False,poc_up=None,reason=reason,recoverable=True)
    shared=study.MemoizedProfiles(provider)
    with pytest.raises(RuntimeError,match='incomplete'):
        selection('poc_filter',events(1),shared)
    assert shared.values=={}
    with pytest.raises(RuntimeError,match='incomplete'):
        study.select_known_profiles(events(1),arm='poc_filter',provider=lambda _:provider(),
            planner=Planner(),planner_factory=lambda _:Planner())


def test_provider_program_error_or_event_mutation_is_fatal():
    def fail(*args):raise RuntimeError('budget exhausted')
    shared=study.MemoizedProfiles(fail)
    with pytest.raises(RuntimeError,match='budget'):selection('poc_filter',events(1),shared)
    calls=[];shared=study.MemoizedProfiles(profile_source({'e0':True},calls))
    shared(events(1)[0])
    with pytest.raises(ValueError,match='identity changed'):
        shared(dict(events(1)[0],priority=999))


def test_hard_filter_audit_rejects_false_unknown_and_quality_fallback():
    account={'cohorts':[{'event_id':'e0'}],'selection_decisions':[]}
    assert study.audit_profile_admissions(account,{'e0':dict(available=True,poc_up=True)},hard_filter=True)
    for value in (dict(available=True,poc_up=False),dict(available=False,poc_up=None)):
        with pytest.raises(ValueError):study.audit_profile_admissions(account,{'e0':value},hard_filter=True)
    account['selection_decisions']=[{'fallback_to_original':True}]
    with pytest.raises(ValueError,match='restored'):
        study.audit_profile_admissions(account,{'e0':dict(available=True,poc_up=True)},hard_filter=True)


def test_native_mro_changes_only_scenario_decision_and_retains_execution_and_financial_chain():
    normal,benchmark=study.engine_types(RangeGapOrders,study.HistoricalOddEra)
    native,_=study.engine_types(RangeGapOrders,study.HistoricalOddEra,native_time20=True)
    assert [c.__qualname__ for c in native.mro() if c.__name__ not in ('NativeAccount','NativeTime20Scenario')] == [c.__qualname__ for c in normal.mro()]
    assert native.mro().index(study.NativeTime20Scenario)+1==native.mro().index(ScenarioExitReplay)
    assert native._execute_order is RangeGapOrders._execute_order
    assert benchmark._execute_order is RangeGapOrders._execute_order


def native_fixture(monkeypatch):
    days=pd.bdate_range('2024-01-02',periods=27)
    engine=object.__new__(study.NativeTime20Scenario)
    engine.days=days;engine.positions={d:i for i,d in enumerate(days)}
    engine.holdings={'2330':dict(event_id='e',qty=100,due_index=63)}
    engine.receivables=[]
    engine.cohorts=[dict(event_id='e',stock_id='2330',entry_date=str(days[0].date()))]
    engine.exit_states={};engine.exit_decisions=[]
    engine.exit_signals=SimpleNamespace(price=lambda index,sid:100.)
    calls=[]
    monkeypatch.setattr(Replay,'corporate_day',lambda self,day:calls.append(day) or 0.)
    return engine,days,calls


def test_native20_waits_for_twenty_completed_sessions_and_latches_retry_without_prices(monkeypatch):
    engine,days,calls=native_fixture(monkeypatch)
    for day in days[1:20]:engine.corporate_day(day)
    assert all(r['exit'] is False for r in engine.exit_decisions)
    assert engine.holdings['2330']['due_index']==63
    engine.exit_signals.price=lambda *args:(_ for _ in ()).throw(AssertionError('No new price needed'))
    engine.corporate_day(days[20])
    first=deepcopy(engine.exit_states['e'])
    assert first['signal_date']==str(days[19].date()) and first['target_index']==20
    assert engine.holdings['2330']['due_index']==20
    engine.corporate_day(days[21])
    assert engine.exit_states['e']==first and engine.exit_decisions[-1]['first_signal_date']==first['signal_date']
    assert len(calls)==21


def test_late_stock_rights_inherit_the_latched_native_exit_after_delivery(monkeypatch):
    engine,days,_=native_fixture(monkeypatch)
    engine.corporate_day(days[1])
    engine.holdings={};engine.receivables=[dict(event_id='e',stock_id='2330',qty=10)]
    for day in days[2:21]:engine.corporate_day(day)
    def deliver(self,day):
        self.holdings['2330']=dict(event_id='e',qty=10,due_index=63)
        self.receivables=[]
        return 0.
    monkeypatch.setattr(Replay,'corporate_day',deliver)
    engine.corporate_day(days[22])
    assert engine.holdings['2330']['due_index']==20
    assert engine.exit_states['e']['signal_date']==str(days[19].date())


def test_true_native_replay_funds_odd_and_board_and_exits_next_session_after_20_days():
    days=pd.bdate_range('2024-01-02',periods=65);entry=30
    prices=[100.]*31+[50.]*34  # 50% drop immediately after entry cannot trigger an early native stop.
    close=pd.DataFrame({'2330':prices,'0050':[100.]*len(days)},index=days)
    quotes=pd.DataFrame([dict(date=d,stock_id=s,open=p,close=p,high=p+2,low=p-2,volume=2000000.)
        for s in close for d,p in close[s].items()])
    companies=pd.DataFrame([dict(stock_id=s,name=s,market='TWSE') for s in close])
    feed=SimpleNamespace(get_limits=lambda sid:{str(d.date()):dict(lower=1.,upper=1000.) for d in days},
        get_odd=lambda date,sid,market:dict(odd_shares=1000000,odd_last=float(close.at[pd.Timestamp(date),sid]),
            odd_bid=1.,odd_ask=1000.,bid_qty=10000,ask_qty=10000))
    corp=SimpleNamespace(prepare=lambda sid:None,on_date=lambda sid,day:[])
    class Native(study.NativeTime20Scenario):
        def buy_etf(self,day,reason):pass
    event=dict(event_id='rsi-2330',members=['2330'],signal_date=str(days[entry-1].date()),
        entry_date=str(days[entry].date()),priority=1.)
    engine=Native(quotes,companies,days,[event],feed,corp,start=str(days[entry].date()),
        end=str(days[55].date()),exit_signals=ExitSignals(close,days),mode='loss12')
    account=engine.run()
    assert {t['channel'] for t in account['trades'] if t['side']=='buy'}=={'board','odd'}
    sells=[t for t in account['trades'] if t['side']=='sell']
    assert sells and {t['date'] for t in sells}=={str(days[entry+20].date())}
    assert {t['signal_date'] for t in sells}=={str(days[entry+19].date())}
    assert {t['reason'] for t in sells}=={'time20'}
    assert not engine.holdings and account['daily'][-1]['nav']==account['daily'][-1]['cash']
    assert study.audit_time20(account,days)['first_triggers']==1
    bad=deepcopy(account);bad['trades'][-1]['date']=str(days[entry+19].date())
    with pytest.raises(ValueError,match='preceded'):study.audit_time20(bad,days)


def test_anchor_equality_checks_every_day_and_trade_without_hiding_differences():
    keys=('daily','trades','cash_ledger','corporate_actions','cohorts','holdings',
          'receivables','resource_plans','tick_plans','selection_decisions')
    account={key:[dict(date='2024-01-03',value=1.)] for key in keys}
    reference={'completed':True,'account':deepcopy(account)}
    assert study.compare_legacy_anchor(account,reference)['all_exact']
    for key in keys:
        changed=deepcopy(account);changed[key][0]['value']=2.
        result=study.compare_legacy_anchor(changed,reference)
        assert not result['all_exact']
        assert result['fields'][key]['first_difference_index']==0
        assert result['fields'][key]['reference']['value']==1.
    with pytest.raises(ValueError,match='lacks ledger'):
        study.compare_legacy_anchor({},reference)


def test_imported_runner_source_guard_rejects_mutation_before_any_preflight(tmp_path,monkeypatch):
    original=tmp_path/'runner.py';original.write_text('source-at-import')
    monkeypatch.setattr(study,'RUNNER_SOURCE',original)
    monkeypatch.setattr(study,'IMPORTED_RUNNER_SHA',hashlib.sha256(original.read_bytes()).hexdigest())
    assert study.verify_runner_source()==study.IMPORTED_RUNNER_SHA
    original.write_text('source-changed-after-import')
    provider=SimpleNamespace(verify_sources=lambda:(_ for _ in ()).throw(AssertionError('too late')))
    with pytest.raises(ValueError,match='changed since import'):
        study._run(tmp_path/'out',('poc_priority',),provider)
    assert not (tmp_path/'out').exists()
