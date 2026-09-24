import hashlib
import json

import pytest

from scripts import publish_priority_optimization as subject


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, 'ROOT', tmp_path)

    def save(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    source = save('raw.json', {'price': 12})
    sources = {'raw.json': hashlib.sha256(source.read_bytes()).hexdigest()}
    save('artifacts/forward_simulation/current_causality_20260925.json', dict(
        passed=True, complete=True, source_and_code_sha256=sources, accepted_candidates=1,
        cases=[{'cutoff': '2022-01-03'}], elapsed_seconds=1))
    save('.cache/oddlot_scope_20260925/summary.json', {'union': {}})
    csv = tmp_path / '.cache/oddlot_scope_20260925/union_stock_date_side.csv'
    csv.write_text('stock_id\n0050\n')
    save('.cache/oddlot_scope_20260925/manifest.json', dict(input_files_sha256=sources,
        output_files_sha256={'union_stock_date_side.csv': hashlib.sha256(csv.read_bytes()).hexdigest()}))
    save('board/summary.json', dict(parent_controls_identical=True, cases={}))
    save('board/identity.json', sources)
    manifest = save('board/manifest.json', {'files_sha256': {}})
    save('board/offline.json', dict(all_cases_identical=True,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
    return tmp_path / 'board', source


def test_publication_rejects_stale_replay_binding(tmp_path, monkeypatch):
    board, _ = fixture(tmp_path, monkeypatch)
    assert subject.build(board)['live_qualified'] is False
    (board / 'manifest.json').write_text('{"files_sha256":{},"changed":true}')
    with pytest.raises(ValueError, match='bind'):
        subject.build(board)


def test_publication_rechecks_inputs_not_just_outputs(tmp_path, monkeypatch):
    board, source = fixture(tmp_path, monkeypatch)
    source.write_text('{"price":999}')
    with pytest.raises(ValueError, match='input or code changed'):
        subject.build(board)
