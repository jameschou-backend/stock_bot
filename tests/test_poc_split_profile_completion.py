import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skills.all_entry_accounts import digest, write
from skills.poc_split_profile_completion import DOCUMENT, DIRECTORY, QUERY, KEY, ExactCompletion
from scripts import run_poc_split_profile_completion as entry


def fixture(root):
    old=root/'failed.json';write(old,dict(query=QUERY,status='provider_error'))
    write(root/DOCUMENT.relative_to(DOCUMENT.parents[1]),dict(schema='poc_split_exact_profile_completion_v1',
        query=QUERY,maximum_new_attempts=1,max_retries=0,immutable_failure_preserved=True,rerun_completed=False,
        previous_failure=dict(path='failed.json',sha256=digest(old))))
    raw=root/'raw';raw.write_bytes(b'exact raw bytes')
    receipt=root/DIRECTORY.relative_to(DOCUMENT.parents[1])/'receipts'/(KEY+'.json')
    write(receipt,dict(query=QUERY,status='received',rows=3191,raw_path='raw',raw_sha256=digest(raw)))
    return old,receipt


def test_exact_completion_keeps_old_failure_and_all_other_query_guards(tmp_path):
    old,receipt=fixture(tmp_path);old_sha=digest(old)
    requested=[]
    previous=SimpleNamespace(refs={},calls=7,get=lambda s,d:requested.append((s,d)) or {'status':'existing'})
    store=ExactCompletion(previous,root=tmp_path)
    assert store.get('4561','2024-05-21')['rows']==3191
    assert store.get('4561','2024-05-22')=={'status':'existing'}
    assert requested==[('4561','2024-05-22')] and store.calls==7
    assert digest(old)==old_sha and json.loads(old.read_text())['status']=='provider_error'
    assert 'failed.json' in store.refs and 'raw' in store.refs
    receipt.write_text('{}')
    with pytest.raises(ValueError,match='hash changed'): store.get('4561','2024-05-21')


def test_changed_original_failure_is_not_hidden(tmp_path):
    old,_=fixture(tmp_path);old.write_text('{}')
    with pytest.raises(ValueError,match='failed receipt changed'):
        ExactCompletion(SimpleNamespace(refs={}),root=tmp_path)


@pytest.mark.parametrize('completed,reason',[(True,'Necessary profiles tick unavailable: '+KEY+' provider_error'),
    (False,'Different missing source'),(False,'Necessary profiles tick unavailable: '+KEY+' provider_error')])
def test_repair_entry_accepts_only_the_exact_incomplete_case(tmp_path,monkeypatch,completed,reason):
    monkeypatch.setattr(entry.single,'BASE',tmp_path)
    p=tmp_path/'report.json'
    write(p,dict(all_completed=completed,split_accounts=dict(arm='poc_priority',mode='slip2'),
        cases=dict(poc_priority=dict(completed=completed,reason=reason))))
    p.with_suffix('.sha256').write_text(digest(p))
    if not completed and reason.endswith(KEY+' provider_error'):
        assert entry.failed_configuration(p)[:2]==('poc_priority','slip2')
    else:
        with pytest.raises(ValueError,match='incomplete'): entry.failed_configuration(p)
