from copy import deepcopy
import json

import pytest

from scripts import combine_all_entry_accounts as module
from skills.all_entry_accounts import registry,digest,write


def fixtures(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'BASE',tmp_path)
    code=tmp_path/'code.py';code.write_text('fixed model')
    monkeypatch.setattr(module,'__file__',str(code))
    counts={r['id']:dict(executable=1,pending=0) for r in registry()}
    manifest=dict(counts=counts,files_sha256={k+'.json':k+'-original' for k in counts})
    def run(name,arms,version,*,old_complete=False):
        folder=tmp_path/name;folder.mkdir();m=folder/'manifest.json';msha=write(m,version)
        cases={}
        for arm in arms:
            value=dict(completed=old_complete,reason=name,summary=None,candidate_coverage=counts.get(arm))
            path=folder/(arm+'.json');sha=write(path,value)
            cases[arm]=dict(value,path=str(path.relative_to(tmp_path)),sha256=sha)
        report=dict(schema='all_entry_accounts_v1',start='2024-01-02',end='2026-10-02',initial_cash=1_000_000,
            input_bundle='input',volume_policy='fixed',data_revision='fixed',execution_model=dict(ordinary='fixed'),
            execution_source_snapshot='code.py',source_sha256={'code.py':digest(code)},cases=cases,
            requested_arms=arms,registered_arms=list(counts)+['benchmark'],all_completed=old_complete,
            candidate_manifest=dict(path=str(m.relative_to(tmp_path)),sha256=msha))
        path=folder/'report.json';sha=write(path,report);path.with_suffix('.sha256').write_text(sha+'\n')
        return path
    first=run('first',list(counts)+['benchmark'],manifest,old_complete=True)
    revised=deepcopy(manifest);revised['files_sha256']['entry_peer_narrow.json']='expanded-peer'
    return first,run,revised


def test_new_candidate_version_replaces_old_case_even_when_new_attempt_fails(tmp_path,monkeypatch):
    first,run,revised=fixtures(tmp_path,monkeypatch)
    patch=run('later',['entry_peer_narrow'],revised)
    merged=module.combine([first,patch],tmp_path/'merged.json')
    assert len(merged['cases'])==32 and merged['all_completed'] is False
    assert merged['cases']['entry_peer_narrow']['reason']=='later'
    assert merged['cases']['momentum']['reason']=='first'


def test_each_changed_candidate_file_requires_its_own_new_replay(tmp_path,monkeypatch):
    first,run,revised=fixtures(tmp_path,monkeypatch)
    revised['files_sha256']['momentum.json']='changed-momentum'
    patch=run('later',['entry_peer_narrow'],revised)
    with pytest.raises(ValueError,match='Changed candidates.*momentum'):
        module.combine([first,patch],tmp_path/'merged.json')


def test_different_period_or_execution_model_cannot_share_a_leaderboard(tmp_path,monkeypatch):
    first,run,revised=fixtures(tmp_path,monkeypatch)
    patch=run('later',['entry_peer_narrow'],revised)
    report=json.loads(patch.read_text());report['execution_model']['ordinary']='another_model'
    patch.write_text(json.dumps(report));patch.with_suffix('.sha256').write_text(digest(patch))
    with pytest.raises(ValueError,match='different account models'):
        module.combine([first,patch],tmp_path/'merged.json')


def test_zero_buy_reentry_policy_cannot_be_mixed_with_old_replays(tmp_path,monkeypatch):
    first,run,revised=fixtures(tmp_path,monkeypatch)
    patch=run('later',['entry_peer_narrow'],revised)
    report=json.loads(patch.read_text());report['zero_buy_reentry_policy']='corrected_zero_rejection'
    patch.write_text(json.dumps(report));patch.with_suffix('.sha256').write_text(module.digest(patch))
    with pytest.raises(ValueError,match='different account models'):
        module.combine([first,patch],tmp_path/'merged.json')


def reseal_report(path, report):
    path.write_text(json.dumps(report));path.with_suffix('.sha256').write_text(digest(path))


def test_copying_writer_coverage_is_verified_from_sealed_case(tmp_path,monkeypatch):
    first,run,revised=fixtures(tmp_path,monkeypatch)
    patch=run('later',['entry_peer_narrow'],revised)
    report=json.loads(patch.read_text());report['cases']['entry_peer_narrow'].pop('candidate_coverage')
    reseal_report(patch,report)
    merged=module.combine([first,patch],tmp_path/'merged.json')
    assert merged['cases']['entry_peer_narrow']['candidate_coverage']==revised['counts']['entry_peer_narrow']


@pytest.mark.parametrize('change',['missing_sealed','wrong_sealed','wrong_descriptor','wrong_completion'])
def test_manifest_cannot_fill_missing_or_conflicting_case_evidence(tmp_path,monkeypatch,change):
    first,run,revised=fixtures(tmp_path,monkeypatch)
    patch=run('later',['entry_peer_narrow'],revised)
    report=json.loads(patch.read_text());descriptor=report['cases']['entry_peer_narrow']
    case_path=tmp_path/descriptor['path'];case=json.loads(case_path.read_text())
    if change=='wrong_descriptor':descriptor['candidate_coverage']={'executable':99}
    elif change=='wrong_completion':case['completed']=not descriptor['completed']
    elif change=='missing_sealed':case.pop('candidate_coverage');descriptor.pop('candidate_coverage')
    else:case['candidate_coverage']={'executable':99};descriptor.pop('candidate_coverage')
    descriptor['sha256']=write(case_path,case);reseal_report(patch,report)
    with pytest.raises(ValueError,match='Case (coverage|completion) differs'):
        module.combine([first,patch],tmp_path/'merged.json')
