import hashlib
import json
from pathlib import Path

from app import revenue_research as service


def test_overview_requires_current_code_input_manifest_and_all_contrasts(monkeypatch, tmp_path):
    report = json.loads((Path(__file__).resolve().parents[1]/'docs/research_revenue_20260909.json').read_text())
    monkeypatch.setattr(service, 'ROOT', tmp_path)
    assert not service.overview()['available']
    for name in service.CODE:
        file = tmp_path/name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('fixture')
        report['code_sha256'][name] = hashlib.sha256(file.read_bytes()).hexdigest()
    prereg = tmp_path/'docs/prereg_revenue_20260909.md'
    prereg.parent.mkdir(parents=True, exist_ok=True)
    prereg.write_text('preregistered')
    report['preregistration_sha256'] = hashlib.sha256(prereg.read_bytes()).hexdigest()
    folder = tmp_path/'.cache/revenue-research'
    folder.mkdir(parents=True)
    manifest = folder/'inputs.json'
    manifest.write_text(json.dumps(report['revenue_inputs']))
    path = folder/'report.summary.json'
    original = json.dumps(report)
    path.write_text(original)
    assert service.overview()['available']
    report['results'][-1] = report['results'][0]
    path.write_text(json.dumps(report))
    assert not service.overview()['available']
    path.write_text(original)
    manifest.write_text('{}')
    assert not service.overview()['available']
    manifest.write_text(json.dumps(report['revenue_inputs']))
    (tmp_path/'skills/revenue_research.py').write_text('changed')
    assert not service.overview()['available']
