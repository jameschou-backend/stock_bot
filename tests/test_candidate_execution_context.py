import hashlib
import json
import pandas as pd
import pytest
from skills.candidate_execution_context import load_context


def seed(root):
    def save(path, value):
        p = root/path; p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value));return hashlib.sha256(p.read_bytes()).hexdigest()
    route = '.cache/legacy/execution-source.json'
    refs = {route: save(route, dict(offline=True,path='.cache/inputs'))}
    (root/'.cache/inputs').mkdir()
    manifest = '.cache/legacy/manifest.json'
    digest = save(manifest, dict(files_sha256={}))
    notice_hash = save('notice.json',dict(announced='2025-05-14'))
    save('docs/benchmark_split_evidence_20260914.json',dict(
        schedule_source=dict(local_path='notice.json',sha256=notice_hash,conservative_known_by='2025-05-14'),
        verified_terms=dict(suspension_start='2025-06-11',new_units_listing_date='2025-06-18')))
    path = 'artifacts/forward_simulation/historical_selector_replay_20260925.json'
    digest_pub = save(path,dict(source_sha256=refs,run_manifest=dict(path=manifest,sha256=digest)))
    (root/path).with_suffix('.sha256').write_text(digest_pub)


def test_context_uses_verified_route_and_only_announced_suspension_days(tmp_path):
    seed(tmp_path)
    days = pd.bdate_range('2025-06-10','2025-06-19')
    inputs, rows, exclusion, refs = load_context(tmp_path,days)
    assert inputs == tmp_path/'.cache/inputs'
    assert list(rows.date.dt.strftime('%Y-%m-%d')) == ['2025-06-11','2025-06-12','2025-06-13','2025-06-16','2025-06-17']
    assert rows.volume.eq(0).all() and rows.close.eq(0).all()
    assert exclusion['end']=='2025-06-18'
    assert 'notice.json' in refs


def test_changed_route_fails_before_returning_inputs(tmp_path):
    seed(tmp_path)
    (tmp_path/'.cache/legacy/execution-source.json').write_text('{}')
    with pytest.raises(ValueError,match='Changed execution context'):
        load_context(tmp_path,pd.bdate_range('2025-06-10','2025-06-19'))
