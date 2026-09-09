import hashlib
import json
from pathlib import Path

from app import event_group_research as service


def test_event_overview_rejects_missing_or_changed_provenance_and_false_qualification(monkeypatch,tmp_path):
    report=json.loads((Path(__file__).resolve().parents[1]/'docs/research_event_groups_20260909.json').read_text())
    monkeypatch.setattr(service,'ROOT',tmp_path)
    assert not service.overview()['available']
    def write(name,content):
        path=tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    for name in service.CODE:
        report['code_sha256'][name]=write(name,'fixture')
    report['preregistration_sha256']=write('docs/prereg_event_groups_20260909.md','frozen')
    audit=json.dumps(report['source_audit'])
    report['source_audit_sha256']=write('docs/event_news_source_audit_20260909.json',audit)
    report['signal_manifest_sha256']=write('.cache/event-group-research/signal-inputs.json',json.dumps(report['inputs']))
    name='.cache/event-group-research/report.summary.json'
    original=json.dumps(report)
    write(name,original)
    assert service.overview()['available']
    report['results'][-1]=report['results'][0]
    write(name,json.dumps(report))
    assert not service.overview()['available']
    report=json.loads(original);report['valid_strategy_evidence']=True
    write(name,json.dumps(report))
    assert not service.overview()['available']
    write(name,original)
    write('docs/event_news_source_audit_20260909.json','{}')
    assert not service.overview()['available']
    write('docs/event_news_source_audit_20260909.json',audit)
    write('skills/event_group_research.py','changed')
    assert not service.overview()['available']
