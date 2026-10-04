"""Offline integration contracts for the separate executable account runner."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import research_poc_executable_account as study


def event(sid, entry='2026-09-02', red=True):
    return dict(event_id=sid+'-'+str(entry), members=[sid], signal_date='2026-09-01',
                entry_date=entry, priority=.7, red=red)


class RedGate:
    def filter_entries(self, entries):
        return ([e for e in entries if e['red']],
                [dict(event_id=e['event_id'], kept=e['red']) for e in entries])


def test_only_original_red_gate_no_broker_or_score_reordering():
    before=event('2330', '2023-12-29')
    a,b,c=event('2317'),event('2308',red=False),event('2382')
    rows=[before,a,b,c];saved=deepcopy(rows)
    kept,log=study.select_arm_entries('poc_red_executable', rows, RedGate(), study.START, study.END)
    assert kept == [before,a,c]
    assert log == [dict(event_id=e['event_id'],kept=e['red']) for e in [a,b,c]]
    assert rows == saved


@pytest.mark.parametrize('change', ['reverse','duplicate','priority','identity'])
def test_gate_cannot_mutate_candidate_population_or_ranking(change):
    class Corrupt:
        def filter_entries(self, entries):
            if change=='reverse':entries.reverse()
            elif change=='duplicate':entries.append(deepcopy(entries[0]))
            elif change=='priority':entries[0]['priority']=123
            else:entries[0]['event_id']='posthoc-winner'
            return entries,[]
    with pytest.raises(ValueError,match='Red gate changed'):
        study.select_arm_entries('poc_red_executable',[event('2330'),event('2317')],Corrupt(),study.START,study.END)


def test_benchmark_has_separate_unfiltered_inputs_and_unregistered_arm_fails():
    rows=[event('2330')]
    kept,log=study.select_arm_entries('benchmark_executable',rows,None,study.START,study.END)
    assert kept is rows and log==[]
    with pytest.raises(ValueError,match='Unregistered'):
        study.select_arm_entries('poc_combined_guard',rows,None,study.START,study.END)


def test_signal_parity_does_not_require_old_hl2_financial_outcome():
    decisions=[dict(event_id='a',kept=True)]
    reference=dict(entry_gate_decisions=decisions,account={'old_hl2_nav':999999})
    audit=study.compare_signal_gate(deepcopy(decisions),reference)
    assert audit['all_exact'] and audit['account_equality_required'] is False
    reference['account']={'old_hl2_nav':0}
    assert study.compare_signal_gate(decisions,reference)==audit
    with pytest.raises(ValueError,match='sealed red-candle gate'):
        study.compare_signal_gate([],reference)


def test_both_financial_mros_dispatch_new_execution_before_hl2():
    class NewOrders:
        def _plan(self,*args,**kwargs):return 'new_plan'
        def _execute_order(self,*args,**kwargs):return 'new_tape'
    class Era(study.HistoricalOddEra):pass
    stock,benchmark=study.engine_types(NewOrders,Era)
    for cls in (stock,benchmark):
        obj=object.__new__(cls)
        assert obj._plan()=='new_plan'
        assert obj._execute_order()=='new_tape'
        assert cls.mro().index(NewOrders)<cls.mro().index(study.MidpointBenchmark if cls is benchmark else study.MidpointExitReplay)


def test_missing_source_preserves_journal_without_partial_performance():
    engine=SimpleNamespace(daily=[dict(date='2024-01-02',nav=1234)],holdings={'2330':{'qty':10}},
        trades=[{'qty':10}],orders=[],cash_ledger=[],actions=[],holding_rows=[],cohorts=[],
        receivables=[{'amount':3}],resource_plans=[],tick_plans=[],day_plans={('a','buy'):{'date':'2024-01-02'}})
    result=study.preserve_failure(ValueError('Missing official auction'),engine)
    assert result['completed'] is False and result['summary'] is None
    assert result['completed_sessions']==1 and result['last_date']=='2024-01-02'
    assert result['partial_journal']['receivables']==[{'amount':3}]
    engine.trades[0]['qty']=100
    assert result['partial_journal']['trades'][0]['qty']==10
    assert study.preserve_failure(ValueError('Bad initialization'),None)['summary'] is None


def test_historical_auction_audit_uses_original_plan_not_final_engine_day():
    key=('e','buy')
    plan=dict(date='2024-01-02',stock_id='2330',planned_qty=1100,odd_qty=100,
              odd_limit=600,side='buy',signal_date='2023-12-29')
    engine=SimpleNamespace(day_plans={key:plan});saved={}
    initial=study.odd_plan_context(engine,saved,'2024-01-02','2330','TWSE')
    engine.day_plans={('future','sell'):dict(date=study.END,stock_id='2330',odd_qty=999,planned_qty=999)}
    initial.day_plans[key]['odd_qty']=5
    replay=study.odd_plan_context(engine,saved,'2024-01-02','2330','twse')
    assert replay.day_plans[key]['odd_qty']==100
    assert replay.day_plans[key]['signal_date']=='2023-12-29'
    with pytest.raises(ValueError,match='precommitted positive'):
        study.odd_plan_context(engine,saved,'2024-01-03','2330','TWSE')


def test_auction_data_cannot_be_requested_for_zero_or_missing_odd_plan():
    engine=SimpleNamespace(day_plans={('e','buy'):dict(date='2024-01-02',stock_id='2330',planned_qty=1000,odd_qty=0)})
    with pytest.raises(ValueError,match='precommitted positive'):
        study.odd_plan_context(engine,{},'2024-01-02','2330','TWSE')
    with pytest.raises(ValueError,match='existing planned order'):
        study.odd_plan_context(None,{},'2024-01-02','2330','TWSE')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def reference_files(tmp_path,monkeypatch):
    parent=tmp_path/'reference';parent.mkdir()
    source=parent/'finance.py';source.write_text('sealed financial source')
    case=dict(completed=True,entry_gate_decisions=[dict(event_id='a',kept=True)])
    casepath=parent/'poc_red.json';casepath.write_text(json.dumps(case))
    report=dict(all_completed=True,start=study.START,end=study.END,initial_cash=1_000_000,
        cases={'poc_red':dict(completed=True,path='reference/poc_red.json',sha256=digest(casepath))},
        source_sha256={'reference/finance.py':digest(source)})
    reportpath=parent/'report.json';reportpath.write_text(json.dumps(report))
    monkeypatch.setattr(study,'BASE_REPORT',reportpath)
    monkeypatch.setattr(study,'BASE_REPORT_SHA',digest(reportpath))
    monkeypatch.setattr(study,'BASE_CASE_SHA',digest(casepath))
    return tmp_path,source,reportpath,report,case


def test_reference_reuses_bound_gate_without_rerunning_account(reference_files):
    root,source,path,report,case=reference_files
    got,refs=study.load_signal_reference(root)
    assert got==case and refs['reference/finance.py']==digest(source)
    assert refs['reference/report.json']==digest(path)


def test_changed_source_in_reference_is_not_silently_accepted(reference_files):
    root,source,*_=reference_files
    source.write_text('changed')
    with pytest.raises(ValueError,match='Sealed source hash changed'):
        study.load_signal_reference(root)


@pytest.mark.parametrize('field,value',[('all_completed',False),('end','2026-09-09'),('initial_cash',2_000_000)])
def test_reference_scope_cannot_drift(reference_files,monkeypatch,field,value):
    root,source,path,report,case=reference_files
    report[field]=value;path.write_text(json.dumps(report))
    monkeypatch.setattr(study,'BASE_REPORT_SHA',digest(path))
    with pytest.raises(ValueError,match='scope differs'):
        study.load_signal_reference(root)


def test_actual_builder_partition_shape_keeps_terminal_signal_out_of_orders(tmp_path):
    active=event('2330');pending=event('2317',entry=None)
    pending['signal_date']=study.END
    files={'signals.json':dict(entries={'median50m':[active]},pending_signal_file='pending-signals.json'),
        'pending-signals.json':dict(schema='pending_last_close_signals_v1',signal_date=study.END,
            next_session_observed=False,execution_inferred=False,live_qualified=False,entries=[pending])}
    for name,value in files.items():(tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(end=study.END,executable_candidate_count=1,
        pending_candidate_count=1,files_sha256={name:digest(tmp_path/name) for name in files})))
    executable,terminal=study.load_candidate_bundle(tmp_path)
    assert executable==[active] and terminal==[pending]
    assert terminal[0]['entry_date'] is None


def test_run_rejects_new_policy_before_provider_or_account_execution(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'RUN_ROOT',tmp_path)
    with pytest.raises(ValueError,match='two preregistered'):
        study._run(tmp_path/'out',['poc_plus_two_percent'],None)
