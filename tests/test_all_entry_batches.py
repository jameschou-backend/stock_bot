"""Batch recovery must never silently rerun or overwrite a sealed account."""
import json
from types import SimpleNamespace

import pytest

from scripts import run_all_entry_batches as module


def setup_batch(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'BASE',tmp_path)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'registry',lambda:[])
    bundle=tmp_path/'input';bundle.mkdir()
    (bundle/'manifest.json').write_text('{}')
    terms=tmp_path/'terms.json';terms.write_text('{}')
    calls=[]
    def worker(command,**kwargs):
        arm=command[command.index('--arms')+1]
        output=module.Path(command[command.index('--output')+1])
        calls.append(arm)
        path=output/'report.json'
        report=dict(requested_arms=[arm],candidate_manifest={'sha256':module.digest(bundle/'manifest.json')},
            cases={arm:dict(completed=False,reason='Missing source')},elapsed_seconds=1.5)
        sha=module.write(path,report);path.with_suffix('.sha256').write_text(sha+'\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',worker)
    monkeypatch.setattr(module,'combine',lambda reports,path:dict(all_completed=False))
    return bundle,terms,calls


def test_progress_advances_and_resume_preserves_failed_attempts(tmp_path,monkeypatch):
    bundle,terms,calls=setup_batch(tmp_path,monkeypatch)
    output=tmp_path/'batch'
    module.run(output,candidate_bundle=bundle,terms=terms)
    assert len(calls)==6
    progress=json.loads((output/'progress.json').read_text())
    assert progress['attempted']==6
    assert all(row['completed'] is False for row in progress['cases'])
    calls.clear()
    module.run(output,candidate_bundle=bundle,terms=terms,resume=True)
    assert calls==[]


def test_resume_rejects_changed_inputs_before_starting_worker(tmp_path,monkeypatch):
    bundle,terms,calls=setup_batch(tmp_path,monkeypatch)
    output=tmp_path/'batch'
    module.run(output,candidate_bundle=bundle,terms=terms)
    calls.clear();terms.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='sealed evidence'):
        module.run(output,candidate_bundle=bundle,terms=terms,resume=True)
    assert calls==[]


def test_resume_rejects_tampered_worker_report(tmp_path,monkeypatch):
    bundle,terms,calls=setup_batch(tmp_path,monkeypatch)
    output=tmp_path/'batch'
    module.run(output,candidate_bundle=bundle,terms=terms)
    calls.clear();(output/'benchmark/report.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum differs'):
        module.run(output,candidate_bundle=bundle,terms=terms,resume=True)
    assert calls==[]
