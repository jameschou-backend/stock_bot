"""Runner contract tests; historical strategy results are never used as fixtures."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from scripts import research_candle_volume_account as runner


def payload():
    return dict(completed=True,account={'daily':[{'nav':1000000.0}], 'settings':{'slots':3}},
                summary={'final_nav':1000000.0},audit={'checked':True},
                profile_queries=[{'event_id':'known'}])


def test_anchor_requires_every_exact_field_and_allows_only_external_metadata():
    expected=payload();actual=deepcopy(expected);actual['rule_audit']={'entry_gate':None}
    assert runner.compare_anchor_payload(actual,expected)['all_exact']


@pytest.mark.parametrize('field',runner.ANCHOR_FIELDS)
def test_anchor_rejects_changed_or_missing_field(field):
    expected=payload();actual=deepcopy(expected);actual[field]={}
    with pytest.raises(ValueError,match=field):runner.compare_anchor_payload(actual,expected)
    actual=deepcopy(expected);actual.pop(field)
    with pytest.raises(ValueError,match=field):runner.compare_anchor_payload(actual,expected)


def test_anchor_rejects_incomplete_results_even_if_financial_values_match():
    actual=payload();actual['completed']=False
    with pytest.raises(ValueError,match='complete'):runner.compare_anchor_payload(actual,payload())


def event(eid,day):return dict(event_id=eid,entry_date=day,priority=1)


def test_red_gate_scope_keeps_boundary_and_existing_event_order():
    entries=[event('old','2023-12-28'),event('boundary','2024-01-02'),
             event('black','2024-01-03'),event('red','2026-09-09'),event('future','2026-09-10')]
    seen=[]
    def gate(rows):
        seen.extend(rows)
        return [r for r in rows if r['event_id']!='black'],[{'id':r['event_id']} for r in rows]
    kept,decisions=runner.select_arm_entries('poc_red',entries,SimpleNamespace(filter_entries=gate),'2024-01-02','2026-09-09')
    assert [r['event_id'] for r in seen]==['boundary','black','red']
    assert [r['event_id'] for r in kept]==['old','boundary','red','future']
    assert len(decisions)==3 and entries[2]['event_id']=='black'


def test_no_red_gate_never_inspects_signal_candles():
    def fail(rows):raise AssertionError('Ungated anchor must not consult red rule')
    entries=[event('same','2024-01-02')]
    for arm in ('original','benchmark','poc_base','poc_dry','poc_dry_weak'):
        result,decisions=runner.select_arm_entries(arm,entries,SimpleNamespace(filter_entries=fail),'2024-01-02','2026-09-09')
        assert result is entries and decisions==[]


@pytest.mark.parametrize('mutation',('reverse','edit','duplicate'))
def test_gate_cannot_reorder_mutate_or_duplicate_candidates(mutation):
    entries=[event('first','2024-01-02'),event('second','2024-01-02')]
    def gate(rows):
        values=deepcopy(rows)
        if mutation=='reverse':values.reverse()
        elif mutation=='edit':values[0]['priority']=2
        else:values.append(values[0])
        return values,[]
    with pytest.raises(ValueError):
        runner.select_arm_entries('poc_red',entries,SimpleNamespace(filter_entries=gate),'2024-01-02','2026-09-09')


def test_gate_data_error_is_not_rejected_candle():
    def unavailable(rows):raise ValueError('Invalid signal candle')
    with pytest.raises(ValueError,match='Invalid signal candle'):
        runner.select_arm_entries('poc_red',[event('bad','2024-01-02')],SimpleNamespace(filter_entries=unavailable),'2024-01-02','2026-09-09')


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,sort_keys=True))
    return runner.sha(path)


def anchor_fixture(tmp_path,monkeypatch):
    cases={};frozen={};refs={}
    for name in ('scripts/research_candle_volume_account.py','skills/candle_volume_rules.py','docs/prereg_candle_volume_account_20261003.md'):
        refs[name]=save(tmp_path/name,{'frozen':name})
    for arm,target in runner.ANCHORS.items():
        ref=payload();rp=Path('.cache/volume-profile-account-20261003/full-v2')/(target+'.json')
        frozen[target]=dict(path=str(rp),sha256=save(tmp_path/rp,ref))
        actual=deepcopy(ref);actual['anchor_parity']=runner.compare_anchor_payload(actual,ref)
        ap=Path('.cache/red-volume-exit-20261003/anchors-v1')/(arm+'.json')
        cases[arm]=dict(path=str(ap),sha256=save(tmp_path/ap,actual),completed=True)
    fr=runner.FROZEN_VP_REPORT.relative_to(runner.ROOT)
    monkeypatch.setitem(runner.FROZEN_BINDINGS,str(fr),save(tmp_path/fr,{'cases':frozen}))
    report=tmp_path/'.cache/red-volume-exit-20261003/anchors-v1/report.json'
    digest=save(report,dict(all_completed=True,cases=cases,source_sha256=refs))
    report.with_suffix('.sha256').write_text(digest+'\n')
    return report


def test_new_variants_require_bound_complete_anchor_reproduction(tmp_path,monkeypatch):
    report=anchor_fixture(tmp_path,monkeypatch)
    refs=runner.require_anchor_report(report,tmp_path)
    assert str(report.relative_to(tmp_path)) in refs
    case=tmp_path/'.cache/red-volume-exit-20261003/anchors-v1/poc_base.json'
    case.write_text('{}')
    with pytest.raises(ValueError,match='changed'):runner.require_anchor_report(report,tmp_path)


def test_new_variants_reject_different_rule_source_since_anchor(tmp_path,monkeypatch):
    report=anchor_fixture(tmp_path,monkeypatch)
    (tmp_path/'skills/candle_volume_rules.py').write_text('changed')
    with pytest.raises(ValueError,match='Anchor source changed'):runner.require_anchor_report(report,tmp_path)


def test_new_variants_cannot_run_before_any_anchor_report(tmp_path):
    with pytest.raises(ValueError,match='all three offline anchors'):
        runner.require_anchor_report(tmp_path/'.cache/red-volume-exit-20261003/anchors-v1/report.json',tmp_path)


def test_frozen_rules_are_only_the_eight_registered_arms():
    assert tuple(runner.ARM_RULES)==runner.ARMS
    assert runner.ARM_RULES['poc_red_dry_weak']==(True,'dry_weak')
    assert runner.ANCHORS['poc_base']=='poc_priority_available'
    assert runner.sha(runner.PREREG)==runner.PREREG_SHA


def test_poc_quality_fallback_restores_only_red_qualified_candidates(monkeypatch):
    import pandas as pd
    from skills.five_axis_replay import FiveAxisReplay
    from skills.volume_profile_account_adapter import AccountCandidateHook
    from skills.volume_profile_selection import select_candidates
    day=pd.Timestamp('2024-01-03')
    candidates=[dict(event_id='black',members=['1234'],priority=3.,signal_date='2024-01-02',entry_date='2024-01-03'),
                dict(event_id='red1',members=['2345'],priority=2.,signal_date='2024-01-02',entry_date='2024-01-03'),
                dict(event_id='red2',members=['3456'],priority=1.,signal_date='2024-01-02',entry_date='2024-01-03')]
    signals=SimpleNamespace(filter_entries=lambda rows:([e for e in rows if e['event_id']!='black'],[]))
    filtered,_=runner.select_arm_entries('poc_red',candidates,signals,'2024-01-02','2026-09-09')
    monkeypatch.setattr(FiveAxisReplay,'corporate_day',lambda self,day:0.)
    engine=object.__new__(AccountCandidateHook)
    engine.__dict__.update(opening_limit=1000000.,cash=1000000.,opening_members=set(),
        released_residuals=set(),holdings={},slots=3,previous_nav=1000000.,residual_budget=1000000/3,
        residual_block=False,prior=lambda day,sid:100.,
        amount20=pd.DataFrame([[100000000.]*3],index=[day],columns=['1234','2345','3456']),
        events={day:filtered},selection_arm='poc_priority_available',selection_decisions=[],
        candidate_selector=select_candidates)
    engine.profile_provider=lambda e:(dict(available=True,poc_up=True) if e['event_id']=='red1'
        else dict(available=False,reason='ordinary_tape_conflict',recoverable=False))
    engine.corporate_day(day)
    assert [e['event_id'] for e in engine.events[day]]==['red1','red2']
    audit=engine.selection_decisions[0]
    assert audit['fallback_to_original'] and audit['reset_state']==audit['initial_state']
    assert audit['discarded_reservation_event_ids']==['red1']
    assert engine.cash==1000000.
