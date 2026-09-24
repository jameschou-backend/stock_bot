import hashlib
import json
import pytest

from app.completion_gaps_ui import load


def seal(path, value):
    path.write_text(json.dumps(value))
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())


def test_gap_summary_checks_evidence_and_never_promotes_live(tmp_path):
    evidence=tmp_path/'proof.json';evidence.write_text('{}')
    path=tmp_path/'report.json'
    value=dict(schema='completion_gaps_v3',live_qualified=False,
               evidence_sha256={'proof.json':hashlib.sha256(evidence.read_bytes()).hexdigest()})
    seal(path,value)
    assert load(path,tmp_path)['live_qualified'] is False
    evidence.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='證據'): load(path,tmp_path)
    value['live_qualified']=True;seal(path,value)
    with pytest.raises(ValueError,match='資格'): load(path,tmp_path)


def test_gap_summary_rejects_changed_seal(tmp_path):
    path=tmp_path/'report.json'
    seal(path,dict(schema='completion_gaps_v3',live_qualified=False,evidence_sha256={}))
    path.write_text('{}')
    with pytest.raises(ValueError,match='摘要已變動'): load(path,tmp_path)
