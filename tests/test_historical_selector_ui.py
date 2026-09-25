from copy import deepcopy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import historical_selector_ui as ui


def display_value():
    cases = {}
    for arm in ui.LABELS:
        for stress in ('control', 'combined'):
            cases[arm + '_' + stress] = dict(completed=True, reason=None,
                result={'path': '.cache/example.json', 'sha256': 'a' * 64},
                summary=dict(total_return=.1 if stress == 'combined' and arm != 'benchmark' else .5,
                             max_drawdown=-.2, final_nav=1_100_000))
    return dict(cases=cases, signals={arm: {'candidates': 456} for arm in ui.LABELS if arm != 'benchmark'},
                preparation_calls=90, causality_checks=684)


def test_display_keeps_stress_failure_and_research_scope_visible(monkeypatch):
    monkeypatch.setattr(ui, 'REPORT', Path(__file__))
    monkeypatch.setattr(ui, 'load', display_value)
    app = AppTest.from_string('from app.historical_selector_ui import render\nrender()').run()
    assert not app.exception
    assert any('壓力測試仍落後 0050' in row.value for row in app.warning)
    assert any('只買賣整張' in row.value for row in app.caption)
    assert app.selectbox[0].value == 'combined'
    assert app.dataframe[0].value['累積淨報酬'].tolist() == ['50.00%'] * 5
    app.radio[0].set_value('combined').run()
    assert not app.exception
    assert app.dataframe[0].value['累積淨報酬'].tolist() == ['10.00%'] * 4 + ['50.00%']


def test_no_previous_return_is_shown_when_new_evidence_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, 'REPORT', tmp_path / 'absent.json')
    app = AppTest.from_string('from app.historical_selector_ui import render\nrender()').run()
    assert not app.exception and not app.dataframe
    assert any('不沿用舊報酬' in row.value for row in app.info)


def test_source_failure_hides_new_returns(monkeypatch):
    monkeypatch.setattr(ui, 'REPORT', Path(__file__))
    def changed():
        raise ValueError('source changed')
    monkeypatch.setattr(ui, 'load', changed)
    app = AppTest.from_string('from app.historical_selector_ui import render\nrender()').run()
    assert not app.exception and not app.dataframe
    assert any('不可採信' in row.value for row in app.error)


def test_success_cache_reuses_hashes_but_rejects_changed_leaf(monkeypatch, tmp_path):
    leaf = tmp_path / 'source.txt'; leaf.write_text('verified source')
    path = tmp_path / 'report.json'
    path.write_text(json.dumps({'source_sha256': {'source.txt': ui.sha(leaf)}, 'value': 42}))
    path.with_suffix('.sha256').write_text(ui.sha(path))
    monkeypatch.setattr(ui, 'validate', lambda value, root: value)
    count = []
    real = ui.sha
    def counted(p):
        count.append(Path(p).name)
        return real(p)
    monkeypatch.setattr(ui, 'sha', counted)
    result = ui.load(path, tmp_path)
    result['value'] = 0
    count.clear()
    assert ui.load(path, tmp_path)['value'] == 42
    assert 'source.txt' not in count
    leaf.write_text('tampered source')
    with pytest.raises(ValueError, match='來源已變更'):
        ui.load(path, tmp_path)


def test_blocked_case_never_gets_zero_return_in_table():
    value = display_value()
    value['cases']['combined_control'].update(completed=False, summary=None, reason='missing settlement')
    row = ui.rows(value, 'control')[3]
    assert row['累積淨報酬'] == '—' and row['期末資產'] == '—'
    assert row['限制'] == 'missing settlement'


def validation_fixture(monkeypatch):
    value = display_value()
    value.update(schema='historical_selector_publication_v1', live_qualified=False,
        strict_data_ready=False, unseen_validation=False, execution_policy='board_only',
        initial_cash=1_000_000, start='2022-01-03', end='2026-09-09',
        run_manifest={'path': 'manifest.json', 'sha256': 'm' * 64})
    files = {}
    for name, schema in (('offline_verification', 'historical_selector_offline_v1'),
                         ('causality_verification', 'historical_selector_causality_v1')):
        value[name] = {'path': name + '.json'}
        files[name + '.json'] = dict(schema=schema, passed=True,
            source_sha256={'manifest.json': 'm' * 64}, compared_cases=10, compared_selectors=4,
            complete=True, expected_checks=684, cases=[{'passed': True} for _ in range(684)])
    for name, row in value['cases'].items():
        row['result'] = {'path': name + '.json'}
        row['summary'].update({k: value[k] for k in ('start', 'end', 'initial_cash')})
        files[name + '.json'] = dict(completed=True, config=dict(board_only=True,
            stress=name.rsplit('_', 1)[1], benchmark=name.startswith('benchmark_')),
            live_qualified=False, summary=deepcopy(row['summary']),
            account={'recomputed_summary': deepcopy(row['summary'])})
    monkeypatch.setattr(ui, 'verified_bytes', lambda descriptor, *args: json.dumps(files[descriptor['path']]).encode())
    monkeypatch.setattr(ui, 'summarize', lambda account: account['recomputed_summary'])
    return value, files


def test_verification_from_another_run_cannot_publish_current_returns(monkeypatch):
    value, files = validation_fixture(monkeypatch)
    assert ui.validate(value) is value
    files['offline_verification.json']['source_sha256']['manifest.json'] = 'x' * 64
    with pytest.raises(ValueError, match='未通過離線重現'):
        ui.validate(value)


def test_incomplete_future_checks_cannot_be_called_complete(monkeypatch):
    value, files = validation_fixture(monkeypatch)
    proof = files['causality_verification.json']
    proof.update(expected_checks=0, cases=[])
    value['causality_checks'] = 0
    with pytest.raises(ValueError, match='未來資料隔離檢查尚未完成'):
        ui.validate(value)


def test_publication_cannot_change_returns_without_changing_account(monkeypatch):
    value, files = validation_fixture(monkeypatch)
    value['cases']['combined_control']['summary']['total_return'] = 10.
    files['combined_control.json']['summary']['total_return'] = 10.
    with pytest.raises(ValueError, match='顯示收益與每日帳戶不一致'):
        ui.validate(value)
