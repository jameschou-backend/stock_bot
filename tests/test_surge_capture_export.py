import gzip
import json

import pytest

import scripts.export_surge_capture as exporter
from scripts.research_surge_capture import sha


def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, 'ROOT', tmp_path)
    source = tmp_path/'source.json'
    source.write_text('{}')
    left, right = tmp_path/'left', tmp_path/'right'
    for folder in (left, right):
        folder.mkdir()
        (folder/'signals.csv').write_text('stock_id,status\n1234,5\n')
        report = dict(schema='surge_capture_20260930', signal_reconstruction_exact=True,
                      live_qualified=False, new_backtest=False,
                      exports_sha256={'signals.csv': sha(folder/'signals.csv')},
                      source_roots_sha256={'source.json': sha(source)})
        (folder/'report.json').write_text(json.dumps(report))
    return left, right, tmp_path/'publication'


def test_publication_keeps_exact_rows_and_refuses_overwrite(tmp_path, monkeypatch):
    left, right, out = runs(tmp_path, monkeypatch)
    exporter.export(left, right, out)
    assert gzip.decompress((out/'signals.csv.gz').read_bytes()) == (left/'signals.csv').read_bytes()
    assert json.loads((out/'report.json').read_text())['offline_identical']
    with pytest.raises(ValueError, match='new publication'):
        exporter.export(left, right, out)


def test_publication_rejects_tampered_result(tmp_path, monkeypatch):
    left, right, out = runs(tmp_path, monkeypatch)
    (right/'signals.csv').write_text('stock_id,status\n1234,4\n')
    with pytest.raises(ValueError, match='Export changed'):
        exporter.export(left, right, out)
    assert not out.exists()


def test_publication_rejects_changed_source(tmp_path, monkeypatch):
    left, right, out = runs(tmp_path, monkeypatch)
    (tmp_path/'source.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Source root changed'):
        exporter.export(left, right, out)
    assert not out.exists()
