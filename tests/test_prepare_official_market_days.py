"""A stopped acquisition must not look like successful completion to callers."""
import json

import pytest

from scripts import prepare_official_market_days as cli


def test_batch_failure_stops_without_dispatching_later_days(tmp_path, monkeypatch):
    items = [dict(market='TWSE', identity=str(i), date=f'2020-01-0{i}') for i in (1, 2)]
    calls = []
    class FailedClient:
        def __init__(self, *args, **kwargs): pass
        def fetch(self, item):
            calls.append(item)
            return dict(accepted=False, status='origin_stopped', http_status=428)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'initialize', lambda *args: items)
    monkeypatch.setattr(cli, 'OfficialDailyAcquisition', FailedClient)
    monkeypatch.setattr('sys.argv', ['prepare', '--report', str(tmp_path/'report.json'),
        '--cache', str(tmp_path/'cache'), '--mode', 'fetch', '--market', 'TWSE'])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert calls == items[:1]


def test_tampered_audit_never_creates_an_acquisition_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    report = tmp_path/'report.json'
    report.write_text(json.dumps(dict(schema='market_input_validation_v2')))
    report.with_suffix('.sha256').write_text('0'*64)
    with pytest.raises(ValueError, match='hash differs'):
        cli.initialize(report, tmp_path/'cache')
    assert not (tmp_path/'cache').exists()
