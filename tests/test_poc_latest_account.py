import ast
import json
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from scripts.research_poc_latest_account import (ARM_RULES, ARMS, BOUNDARY, END,
    BoundaryCapture, compare_prefix_payload, executable_entries, merge_overrides,
    load_candidate_bundle, merge_supplements, select_arm_entries)
from scripts.prepare_poc_latest_inputs import partition_signals
from scripts.export_signal_explorer import digest
from scripts import research_poc_latest_account as latest


def event(eid='old', signal='2026-09-08', entry='2026-09-09'):
    return dict(event_id=eid, signal_date=signal, entry_date=entry, members=['2330'], priority=.2)


def prefix_fixture():
    row = dict(date=BOUNDARY, nav=1234., cash=100.)
    position = dict(date=BOUNDARY, stock_id='2330', qty=2, price=567.)
    cohort = dict(event_id='old', entry_date='2026-08-03', due_date=None,
                  due_index=123, exit_date=None, bought_qty=2)
    a = dict(daily=[row], trades=[dict(date=BOUNDARY, qty=2, side='buy')],
             orders=[], cash_ledger=[], corporate_actions=[], holdings=[position],
             resource_plans=[], selection_decisions=[], tick_plans=[],
             settings={'slots': 3, 'slippage': .0045},
             cohorts=[cohort], receivables=[], ending_inventory={'positions': [position]})
    query = dict(event_id='old', signal_date='2026-09-08', stock_id='2330')
    gate = dict(event_id='old', signal_date='2026-09-08', entry_date=BOUNDARY, passed=True)
    expected = dict(completed=True, account=a, profile_queries=[query], entry_gate_decisions=[gate])
    actual = deepcopy(expected)
    actual['account']['daily'].append(dict(date='2026-09-10', nav=2000., cash=20.))
    actual['account']['trades'].append(dict(date='2026-09-10', qty=1, side='sell'))
    actual['account']['cohorts'][0].update(exit_date='2026-09-18', due_date='2026-09-30')
    actual['account']['receivables'].append({'amount': 2., 'pay_date': '2026-10-05'})
    actual['profile_queries'].append(dict(event_id='new', signal_date=BOUNDARY, stock_id='2330'))
    actual['entry_gate_decisions'].append(dict(event_id='new', signal_date=BOUNDARY, entry_date='2026-09-10'))
    state = dict(nav=1234., cash=100., receivables=[], positions=[deepcopy(position)], cohorts=[deepcopy(cohort)])
    state['cohorts'][0]['due_date'] = '2026-09-30'
    return actual, expected, state


def test_end_and_four_arms_do_not_add_strategy_variants():
    assert ARMS == ('original', 'benchmark', 'poc_base', 'poc_red')
    assert END == '2026-10-02'
    assert ARM_RULES == {'original': (False, 'none'), 'benchmark': (False, 'none'),
                         'poc_base': (False, 'none'), 'poc_red': (True, 'none')}


def test_latest_pending_signals_are_retained_without_inventing_next_day():
    old = event()
    new = event('new', BOUNDARY, '2026-09-10')
    pending = event('pending', END, None)
    active, waiting = executable_entries({'entries': [old, new, pending]})
    assert active == [old, new] and waiting == [pending]
    assert waiting[0]['entry_date'] is None
    assert executable_entries({'entries': {'median50m': [old]}}) == ([old], [])


def write_partitioned_bundle(folder,rows,*,end=END):
    """Match prepare_poc_latest_inputs' actual two-file output contract."""
    original=[e for e in rows if e['signal_date']<BOUNDARY]
    calendar=pd.to_datetime(sorted({e['signal_date'] for e in rows}
        |{e['entry_date'] for e in rows if e['entry_date'] is not None}))
    active,pending=partition_signals(original,{'prefix_exact':True,'entries':rows},calendar,end=end)
    folder.mkdir(parents=True,exist_ok=True)
    prepared={'entries':{'median50m':active},'pending_signal_file':'pending-signals.json'}
    terminal=dict(schema='pending_last_close_signals_v1',signal_date=end,entries=pending,
                  next_session_observed=False,execution_inferred=False,live_qualified=False)
    for name,value in [('signals.json',prepared),('pending-signals.json',terminal)]:
        (folder/name).write_text(json.dumps(value))
    manifest=dict(end=end,executable_candidate_count=len(active),pending_candidate_count=len(pending),
        files_sha256={n:digest(folder/n) for n in ('signals.json','pending-signals.json')})
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return active,pending


def test_builder_partition_integration_keeps_all_terminal_candidates(tmp_path):
    rows=[event(),event('new',BOUNDARY,'2026-09-10'),event('pending',END,None)]
    expected=write_partitioned_bundle(tmp_path,rows)
    active,pending=load_candidate_bundle(tmp_path)
    assert (active,pending)==expected and len(active)+len(pending)==3
    assert pending==[rows[-1]] and pending[0]['entry_date'] is None


@pytest.mark.parametrize('mutation',['hash','missing_binding','count','future','duplicate'])
def test_terminal_partition_cannot_be_omitted_tampered_or_invented(tmp_path,mutation):
    write_partitioned_bundle(tmp_path,[event(),event('pending',END,None)])
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    terminal=json.loads((tmp_path/'pending-signals.json').read_text())
    if mutation=='missing_binding':manifest['files_sha256'].pop('pending-signals.json')
    elif mutation=='count':manifest['pending_candidate_count']=0
    else:
        if mutation=='future':terminal['next_session_observed']=True
        elif mutation=='duplicate':terminal['entries'][0]['event_id']='old'
        else:terminal['entries']=[]
        (tmp_path/'pending-signals.json').write_text(json.dumps(terminal))
        if mutation!='hash':manifest['files_sha256']['pending-signals.json']=digest(tmp_path/'pending-signals.json')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):load_candidate_bundle(tmp_path)


@pytest.mark.parametrize('rows', [[event('bad', '2026-09-30', None)],
    [event(), event()], [event('bad', END, '2026-10-05')], [event('bad', BOUNDARY, BOUNDARY)]])
def test_invalid_candidate_dates_or_duplicates_stop(rows):
    with pytest.raises(ValueError):
        executable_entries({'entries': rows})


def test_prefix_ignores_only_new_journals_and_future_calendar_projection():
    actual, expected, state = prefix_fixture()
    result = compare_prefix_payload(actual, expected, state)
    assert result['all_exact']
    assert result['boundary_nav'] == 1234.
    assert result['calendar_projection_only'] == [{'event_id': 'old', 'extended_due_date': '2026-09-30'}]


@pytest.mark.parametrize('field', ['daily', 'trades', 'settings'])
def test_changed_old_financial_evidence_never_passes(field):
    actual, expected, state = prefix_fixture()
    if field == 'daily':
        actual['account']['daily'][0]['cash'] += .01
    elif field == 'trades':
        actual['account']['trades'][0]['qty'] += 1
    else:
        actual['account']['settings']['slots'] = 5
    with pytest.raises(ValueError, match='differs|changed'):
        compare_prefix_payload(actual, expected, state)


@pytest.mark.parametrize('field', ['nav', 'cash', 'receivables', 'positions', 'cohorts'])
def test_changed_boundary_state_is_not_a_fresh_cash_restart(field):
    actual, expected, state = prefix_fixture()
    if field in ('nav', 'cash'):
        state[field] += 1
    elif field == 'receivables':
        state[field].append({'amount': 2})
    elif field == 'positions':
        state[field][0]['qty'] += 1
    else:
        state[field][0]['bought_qty'] += 1
    with pytest.raises(ValueError, match='differs|differ'):
        compare_prefix_payload(actual, expected, state)


def test_old_profile_order_and_red_gate_must_match_before_new_9_9_signal():
    actual, expected, state = prefix_fixture()
    actual['profile_queries'][0]['stock_id'] = '9999'
    with pytest.raises(ValueError, match='profile query sequence'):
        compare_prefix_payload(actual, expected, state)
    actual, expected, state = prefix_fixture()
    actual['entry_gate_decisions'][0]['passed'] = False
    with pytest.raises(ValueError, match='red gate'):
        compare_prefix_payload(actual, expected, state)


def test_due_index_or_past_due_date_cannot_be_hidden_by_normalization():
    actual, expected, state = prefix_fixture()
    state['cohorts'][0]['due_index'] += 1
    with pytest.raises(ValueError, match='cohort content'):
        compare_prefix_payload(actual, expected, state)
    actual, expected, state = prefix_fixture()
    state['cohorts'][0]['due_date'] = '2026-09-08'
    with pytest.raises(ValueError, match='Historical due date'):
        compare_prefix_payload(actual, expected, state)


def test_boundary_is_captured_before_first_new_corporate_action():
    class Parent:
        def __init__(self):
            self.daily = [{'date': BOUNDARY}]
            self.cash = 100
            self.previous_nav = 200
            self.receivables = [{'amount': 3}]
            self.cohorts = [{'event_id': 'old'}]
            self.holding_rows = [{'date': BOUNDARY, 'qty': 1}]

        def corporate_day(self, day):
            self.cash += 3
            self.receivables.clear()
            return 3

    class Account(BoundaryCapture, Parent):
        pass

    account = Account()
    assert account.corporate_day(pd.Timestamp('2026-09-10')) == 3
    assert account.prefix_boundary_state['cash'] == 100
    assert account.prefix_boundary_state['receivables'] == [{'amount': 3}]
    account.corporate_day(pd.Timestamp('2026-09-11'))
    assert account.prefix_boundary_state['cash'] == 100


def test_additive_corporate_terms_cannot_overwrite_sealed_history():
    original = {'2330-old': {'cash': 2}}
    assert merge_overrides(original, {'2330-new': {'cash': 3}})['2330-new']['cash'] == 3
    with pytest.raises(ValueError, match='change old action'):
        merge_overrides(original, {'2330-old': {'cash': 3}})
    old = [{'stock_id': '2330', 'date': '2026-08-01', 'cash_per_share': 2}]
    assert merge_supplements(old, deepcopy(old)) == old
    changed = deepcopy(old)
    changed[0]['cash_per_share'] = 3
    with pytest.raises(ValueError, match='changes old action'):
        merge_supplements(old, changed)


def test_red_gate_retains_order_and_only_scopes_executable_account_entries():
    class Signals:
        def filter_entries(self, entries):
            return [entries[-1]], [{'event_id': e['event_id']} for e in entries]
    old = event('pre', '2023-12-27', '2023-12-28')
    a, b = event('a'), event('b', BOUNDARY, '2026-09-10')
    kept, log = select_arm_entries('poc_red', [old, a, b], Signals(), '2024-01-02', END)
    assert kept == [old, b] and len(log) == 2
    assert select_arm_entries('poc_base', [old, a, b], Signals(), '2024-01-02', END) == ([old, a, b], [])


def test_financial_classes_are_static_copies_with_observation_only_outer_mixin():
    root = Path(__file__).resolve().parents[1]
    def classes(name):
        tree = ast.parse((root / 'scripts' / name).read_text())
        return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    old = classes('research_candle_volume_account.py')
    new = classes('research_poc_latest_account.py')
    for name in ('Era', 'RepairedOriginal', 'Feeds'):
        assert ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False)
    for name in ('CandleVolumeAccount', 'RepairedBenchmark'):
        assert ast.dump(new[name].bases[0]) == "Name(id='BoundaryCapture', ctx=Load())"
        assert [ast.dump(b) for b in new[name].bases[1:]] == [ast.dump(b) for b in old[name].bases]


def test_shared_prefix_cache_hashes_and_reads_each_source_once_per_run(tmp_path,monkeypatch):
    folder=tmp_path/'.cache/red-volume-exit-20261003/anchors-v2'
    folder.mkdir(parents=True)
    source=tmp_path/'source.json';source.write_text('{"fixed":true}')
    items={}
    for arm in ('original','poc_base'):
        path=folder/(arm+'.json')
        path.write_text(json.dumps(dict(completed=True,summary={'end':BOUNDARY})))
        items[arm]=dict(completed=True,path=str(path.relative_to(tmp_path)),sha256=digest(path))
    report=folder/'report.json'
    report.write_text(json.dumps(dict(cases=items,source_sha256={'source.json':digest(source)})))
    monkeypatch.setitem(latest.PREFIX_PARENTS,'anchors-v2',digest(report))
    hashes,reads=Counter(),Counter();original_sha,original_read=latest.sha,latest.read
    def counted_sha(path):
        hashes[Path(path)]+=1
        return original_sha(path)
    def counted_read(path):
        reads[Path(path)]+=1
        return original_read(path)
    monkeypatch.setattr(latest,'sha',counted_sha);monkeypatch.setattr(latest,'read',counted_read)
    cache=latest.RunSourceCache(tmp_path)
    results=[latest.load_prefix_reference(a,tmp_path,verified_cache=cache) for a in items]
    assert all(value['completed'] for value,refs in results)
    assert hashes[source]==hashes[report]==reads[report]==1
    assert max(hashes.values())==1
    latest.load_prefix_reference('original',tmp_path,verified_cache=latest.RunSourceCache(tmp_path))
    assert hashes[source]==hashes[report]==reads[report]==2


def test_metadata_change_rehashes_and_changed_bytes_reject_even_with_restored_mtime(tmp_path,monkeypatch):
    source=tmp_path/'bytes';source.write_bytes(b'fixed');expected=digest(source)
    calls=[];original_sha=latest.sha
    def counted_sha(path):calls.append(path);return original_sha(path)
    monkeypatch.setattr(latest,'sha',counted_sha)
    cache=latest.RunSourceCache(tmp_path);cache.verify(source,expected)
    before=source.stat()
    os.utime(source,ns=(before.st_atime_ns,before.st_mtime_ns+1_000_000_000))
    cache.verify(source,expected)
    assert len(calls)==2  # A metadata-only change rechecks identical content.
    verified=source.stat();source.write_bytes(b'other')
    os.utime(source,ns=(verified.st_atime_ns,verified.st_mtime_ns))
    assert source.stat().st_size==verified.st_size and source.stat().st_mtime_ns==verified.st_mtime_ns
    with pytest.raises(ValueError,match='hash changed'):cache.verify(source,expected)
    assert len(calls)==3


def test_conflicting_hash_or_mid_read_mutation_is_not_cached(tmp_path,monkeypatch):
    source=tmp_path/'bytes';source.write_bytes(b'fixed');expected=digest(source)
    cache=latest.RunSourceCache(tmp_path);cache.verify(source,expected)
    with pytest.raises(ValueError,match='Conflicting sealed'):cache.verify(source,'0'*64)
    def mutate(path):
        source.write_bytes(b'other')
        return expected
    monkeypatch.setattr(latest,'sha',mutate)
    with pytest.raises(ValueError,match='during verification'):
        latest.RunSourceCache(tmp_path).verify(source,expected)


def test_source_merges_resolve_only_new_paths_but_reject_hash_conflicts(tmp_path,monkeypatch):
    root=tmp_path.resolve();calls=[];original=Path.resolve
    def counted(path,*args,**kwargs):
        calls.append(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',counted)
    refs={}
    latest.merge_source_refs(refs,{'first':'hash1'},root)
    latest.merge_source_refs(refs,{'first':'hash1','second':'hash2'},root)
    assert calls==[root/'first',root/'second']
    with pytest.raises(ValueError,match='Conflicting frozen source hash'):
        latest.merge_source_refs(refs,{'first':'different'},root)
    assert refs=={'first':'hash1','second':'hash2'}


def test_source_merge_rejects_path_escape(tmp_path):
    root=tmp_path.resolve();refs={}
    with pytest.raises(ValueError,match='Source escapes repository'):
        latest.merge_source_refs(refs,{'../outside':'hash'},root)
    assert refs=={}


def test_final_path_validation_detects_retargeted_symlink(tmp_path):
    root=(tmp_path/'repo');root.mkdir();root=root.resolve()
    inside=root/'inside';inside.write_text('fixed')
    outside=tmp_path/'outside';outside.write_text('fixed')
    link=root/'link';link.symlink_to(inside)
    refs={};latest.merge_source_refs(refs,{'link':digest(inside)},root)
    latest.validate_source_paths(refs,root)
    link.unlink();link.symlink_to(outside)
    with pytest.raises(ValueError,match='Source escapes repository'):
        latest.validate_source_paths(refs,root)
