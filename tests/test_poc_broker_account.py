"""Entry-gate integration and exact neutral-account contracts, offline only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
from scripts import research_poc_broker_account as study


def event(sid,date='2026-09-02',red=True):
    return dict(event_id=sid+'-'+date,signal_date='2026-09-01',entry_date=date,
                members=[sid],priority=.5,red=red)


class RedGate:
    def __init__(self):self.calls=[]
    def filter_entries(self,entries):
        self.calls.append(deepcopy(entries))
        return [e for e in entries if e['red']],[dict(event_id=e['event_id'],passed=e['red']) for e in entries]


@pytest.mark.parametrize('arm',study.ARMS)
def test_every_arm_applies_red_gate_before_broker_without_touching_outside_window(arm):
    old=event('2330','2023-12-29');a=event('2317');b=event('2308',red=False);c=event('2382')
    entries=[old,a,b,c];red=RedGate();seen=[];evidence=object()
    def gate(got_arm,selected,got_evidence):
        assert got_arm==arm and got_evidence is evidence
        seen.extend(selected)
        kept=selected if arm=='poc_red' else selected[1:]
        return kept,[dict(event_id=e['event_id'],gate='broker') for e in selected]
    selected,redlog,brokerlog=study.select_account_entries(arm,entries,red,evidence,study.START,study.END,gate_fn=gate)
    assert red.calls==[[a,b,c]] and seen==[a,c]
    assert selected==([old,a,c] if arm=='poc_red' else [old,c])
    assert len(redlog)==3 and len(brokerlog)==2
    assert entries==[old,a,b,c]


@pytest.mark.parametrize('mutation',['reverse','duplicate','changed_priority','invented_id'])
def test_broker_gate_cannot_change_order_identity_or_candidate_fields(mutation):
    rows=[event('2330'),event('2317')]
    def gate(arm,selected,evidence):
        if mutation=='reverse':selected.reverse()
        elif mutation=='duplicate':selected.append(selected[0])
        elif mutation=='changed_priority':selected[0]['priority']=999
        else:selected[0]['event_id']='future-winner'
        return selected,[]
    with pytest.raises(ValueError,match='Broker gate changed'):
        study.select_account_entries('poc_persist_guard',rows,RedGate(),{},study.START,study.END,gate_fn=gate)


def test_neutral_broker_gate_must_keep_every_red_candidate():
    with pytest.raises(ValueError,match='Baseline broker gate'):
        study.select_account_entries('poc_red',[event('2330')],RedGate(),{},study.START,study.END,
                                     gate_fn=lambda *args:([],[]))


def test_gate_failure_preserves_separate_ledgers():
    def gate(*args):
        exc=ValueError('Missing bound source')
        exc.decisions=[{'event_id':'x','status':'unknown'}]
        raise exc
    with pytest.raises(ValueError) as caught:
        study.select_account_entries('poc_known5_filter',[event('2330')],RedGate(),{},study.START,study.END,gate_fn=gate)
    assert caught.value.entry_gate_decisions==[dict(event_id='2330-2026-09-02',passed=True)]
    assert caught.value.broker_gate_decisions==[{'event_id':'x','status':'unknown'}]


def case():
    return dict(completed=True,account=dict(daily=[dict(date=study.END,nav=1000000,cash=1000000)],
        trades=[],orders=[],settings={'slots':3}),summary={'final_nav':1000000},
        profile_queries=[{'event_id':'a'}],entry_gate_decisions=[{'event_id':'a','passed':True}])


@pytest.mark.parametrize('key',['account','summary','profile_queries','entry_gate_decisions'])
def test_exact_baseline_checks_more_than_final_nav(key):
    reference=case();actual=deepcopy(reference)
    actual[key]=[] if isinstance(actual[key],list) else {'final_nav':1000000,'changed':True}
    with pytest.raises(ValueError,match='baseline differs'):
        study.compare_account_baseline(actual,reference)


def test_extra_broker_report_fields_do_not_pollute_exact_economic_baseline():
    reference=case();actual=deepcopy(reference)
    actual['broker_gate_decisions']=[{'status':'baseline_pass_through'}]
    audit=study.compare_account_baseline(actual,reference)
    assert audit['all_exact'] and audit['through']==study.END
    assert set(audit['field_sha256'])=={'account','summary','profile_queries','entry_gate_decisions'}
    actual['completed']=False
    with pytest.raises(ValueError,match='Complete accounts'):
        study.compare_account_baseline(actual,reference)


@pytest.fixture
def source_reference(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    path=tmp_path/'parent/report.json';path.parent.mkdir()
    monkeypatch.setattr(study,'BASE_REPORT',path)
    source=tmp_path/'financial.py';source.write_text('frozen financial source')
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    files={};values={}
    for name in ('poc_red','benchmark'):
        value=case();value['summary'].update(start=study.START,end=study.END,initial_cash=1000000)
        p=path.parent/(name+'.json');p.write_text(json.dumps(value));values[name]=value
        files[name]=dict(completed=True,path=str(p.relative_to(tmp_path)),sha256=digest(p),summary=value['summary'])
    monkeypatch.setattr(study,'BASE_CASE_SHA',files['poc_red']['sha256'])
    report=dict(all_completed=True,start=study.START,end=study.END,initial_cash=1000000,
                volume_policy='legacy_total_research',cases=files,source_sha256={'financial.py':digest(source)})
    path.write_text(json.dumps(report));monkeypatch.setattr(study,'BASE_REPORT_SHA',digest(path))
    return path,source,report,values


def test_bound_latest_baseline_and_benchmark_are_reused_without_account_replay(source_reference):
    path,source,report,values=source_reference
    baseline,benchmark,refs=study.load_account_reference(path.parents[1])
    assert baseline==values['poc_red']
    assert benchmark['summary']==values['benchmark']['summary']
    assert benchmark['reused_sealed_account'] and not benchmark['recomputed_this_study']
    assert refs['financial.py']==report['source_sha256']['financial.py']


def test_reference_tamper_halts_before_financial_execution(source_reference):
    path,source,_,_=source_reference
    source.write_text('changed')
    with pytest.raises(ValueError,match='Sealed source hash changed'):
        study.load_account_reference(path.parents[1])


def test_reference_scope_is_fixed(source_reference,monkeypatch):
    path,_,report,_=source_reference
    report['end']='2026-09-09';path.write_text(json.dumps(report))
    monkeypatch.setattr(study,'BASE_REPORT_SHA',hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='scope differs'):
        study.load_account_reference(path.parents[1])
