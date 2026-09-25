import hashlib
import json
from pathlib import Path

import pytest

from app.backtest_completion_ui import load


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture(root):
    config = {'board_only': True}
    summary = dict(total_return=.1, max_drawdown=-.2)
    case = put(root, 'case.json', dict(completed=True, config=config, summary=summary,
        live_qualified=False, unseen_validation=False))
    corporate = put(root, 'corporate.json', dict(live_qualified=False, cases={
        'capacity_control_board_only': dict(completed=True, config=config, summary=summary,
            path=case['path'], sha256=case['sha256'])}))
    sector = put(root, 'sector.json', dict(live_qualified=False, case_rows=[]))
    source = put(root, 'tape.json', dict(source='original'))
    data = put(root, 'data.json', dict(schema='backtest_data_completion_v1',
        live_qualified=False, strict_data_ready=False, input_sha256={'tape.json': source['sha256']}, code_sha256={},
        case_sources={'account':case}, cases={'account':{}}, coverage_totals={'case_count':1}))
    (root / 'data.sha256').write_text(data['sha256'])
    path = root / 'index.json'
    index = put(root, path.name, dict(format='backtest_completion_v1', live_qualified=False,
        reports=dict(corporate=corporate, sector=sector, data=data)))
    path.with_suffix('.sha256').write_text(index['sha256'])
    return path


def test_changed_account_prevents_showing_cached_returns(tmp_path):
    path = fixture(tmp_path)
    assert load(path, tmp_path)[1]['corporate']['cases']['capacity_control_board_only']['completed']
    (tmp_path / 'case.json').write_text('{}')
    with pytest.raises(ValueError, match='已變動'):
        load(path, tmp_path)


def test_tampered_index_is_rejected(tmp_path):
    path = fixture(tmp_path)
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='指紋'):
        load(path, tmp_path)


def test_blocked_case_cannot_publish_partial_return(tmp_path):
    path = fixture(tmp_path)
    case = put(tmp_path, 'case.json', dict(completed=False, config={}, live_qualified=False,
                                         unseen_validation=False, reason='missing tape'))
    sector = put(tmp_path, 'sector.json', dict(live_qualified=False, case_rows=[dict(
        status='blocked', config={}, result_path=case['path'], result_sha256=case['sha256'],
        total_return=.7, max_drawdown=None, artifact_paths={})]))
    corporate = put(tmp_path, 'corporate.json', dict(live_qualified=False, cases={}))
    index = json.loads(path.read_text())
    index['reports'].update(corporate=corporate, sector=sector)
    data = json.loads((tmp_path / 'data.json').read_text())
    data['case_sources'] = {'account':case}
    data_descriptor = put(tmp_path, 'data.json', data)
    (tmp_path / 'data.sha256').write_text(data_descriptor['sha256'])
    index['reports']['data'] = data_descriptor
    descriptor = put(tmp_path, path.name, index)
    path.with_suffix('.sha256').write_text(descriptor['sha256'])
    with pytest.raises(ValueError, match='中途收益'):
        load(path, tmp_path)


def test_underlying_data_change_is_rejected_even_when_published_report_is_unchanged(tmp_path):
    path = fixture(tmp_path)
    assert load(path, tmp_path)[1]['data']['live_qualified'] is False
    (tmp_path / 'tape.json').write_text('{"source":"changed"}')
    with pytest.raises(ValueError, match='hash mismatch'):
        load(path, tmp_path)


def test_verified_data_must_equal_the_bytes_from_published_descriptor(tmp_path, monkeypatch):
    from app import backtest_completion_ui as ui
    path = fixture(tmp_path)
    seen = []
    def verifier(source, root):
        seen.append((source, root))
        return {'different': True}
    monkeypatch.setattr(ui, 'verify_data_report', verifier)
    with pytest.raises(ValueError, match='核對期間已變動'):
        load(path, tmp_path)
    assert seen == [(tmp_path / 'data.json', tmp_path)]


@pytest.mark.parametrize('mismatch', ['different_accounts', 'duplicate_accounts', 'wrong_count'])
def test_valid_evidence_for_different_scope_cannot_describe_displayed_accounts(tmp_path, mismatch):
    path = fixture(tmp_path)
    data = json.loads((tmp_path / 'data.json').read_text())
    if mismatch == 'different_accounts':
        data['case_sources']['account'] = put(tmp_path, 'other-case.json', {'other':True})
    elif mismatch == 'duplicate_accounts':
        data['case_sources']['duplicate'] = data['case_sources']['account']
        data['cases']['duplicate'] = {}
    else:
        data['coverage_totals']['case_count'] = 2
    descriptor = put(tmp_path, 'data.json', data)
    (tmp_path / 'data.sha256').write_text(descriptor['sha256'])
    index = json.loads(path.read_text())
    index['reports']['data'] = descriptor
    descriptor = put(tmp_path, path.name, index)
    path.with_suffix('.sha256').write_text(descriptor['sha256'])
    with pytest.raises(ValueError, match='範圍不一致'):
        load(path, tmp_path)
