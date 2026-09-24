"""Negative controls for the current snapshot audit; no remote data or returns."""
from copy import deepcopy
import json
import socket

import pandas as pd
import pytest

from scripts import audit_current_causality_20260925 as audit
from scripts.prepare_million_signals import build_signals
from tests.test_million_signals import signal_inputs


def test_cutoffs_cover_every_month_transition_and_all_original_quarters():
    index = pd.bdate_range('2021-01-04', '2026-09-09')
    plan = audit.cutoff_plan(index, '2022-01-03', '2026-09-08')
    assert len(plan) == 114
    labels = [label for row in plan for label in row['labels']]
    assert sum(label.startswith('month_first:') for label in labels) == 57
    assert sum(label.startswith('month_last:') for label in labels) == 56
    assert {label for label in labels if label.startswith('original_quarter:')} == {
        'original_quarter:' + requested for requested in audit.CUTS}
    assert not any('month_last:2026-09' == label for label in labels)
    assert next(row for row in plan if 'original_quarter:2022-12-31' in row['labels'])['cutoff'] == '2022-12-30'
    assert plan[-1]['next_session'] == '2026-09-09'


def test_cutoffs_do_not_invent_a_next_trading_session():
    index = pd.bdate_range('2021-01-04', '2026-09-08')
    with pytest.raises(ValueError, match='known next market session'):
        audit.cutoff_plan(index, '2022-01-03', '2026-09-08')


def test_deterministic_comparison_only_ignores_elapsed_seconds():
    original = build_signals(*signal_inputs(), start='2022-01-03', signal_end='2022-01-05')
    changed = deepcopy(original)
    changed['diffusion']['stats']['seconds'] += 100
    assert audit.deterministic(changed) == audit.deterministic(original)
    changed['entries'][0]['priority'] += 1
    assert audit.deterministic(changed) != audit.deterministic(original)
    changed = deepcopy(original)
    changed['diffusion']['events'][0]['status'] = 'incorrect'
    assert audit.deterministic(changed) != audit.deterministic(original)


def test_daily_group_rejection_evidence_is_compared_as_of_its_date():
    result = build_signals(*signal_inputs(), start='2022-01-03', signal_end='2022-01-05')
    result['diffusion']['groups'][0]['leader_rejections'] = [
        {'date': '2022-01-04', 'reason': 'already_broad'},
        {'date': '2022-01-06', 'reason': 'future'}]
    expected = audit.as_of(result, '2022-01-05')
    changed = deepcopy(result)
    changed['diffusion']['groups'][0]['leader_rejections'][1]['reason'] = 'future_mutated'
    assert audit.as_of(changed, '2022-01-05') == expected
    changed['diffusion']['groups'][0]['leader_rejections'][0]['reason'] = 'past_mutated'
    assert audit.as_of(changed, '2022-01-05') != expected


def test_source_hash_mismatch_fails_closed_and_is_persisted(tmp_path, monkeypatch):
    def fail():
        raise ValueError('Sealed source/code changed or missing: test-source')
    monkeypatch.setattr(audit, 'provenance', fail)
    output = tmp_path / 'failed.json'
    report = audit.run(output)
    assert report['status'] == 'failed'
    assert not report['passed'] and not report['complete']
    assert report['cases'] == []
    assert json.loads(output.read_text())['error']['message'].endswith('test-source')


def test_failed_rebuild_is_not_replaced_with_sealed_signals(tmp_path, monkeypatch):
    *frames, companies = signal_inputs()
    monkeypatch.setattr(audit, 'provenance', lambda: {})
    monkeypatch.setattr(audit, 'matrices', lambda path: frames)
    monkeypatch.setattr(audit, 'verify_matrix_recipe', lambda frames: {'passed': True})
    monkeypatch.setattr(audit.pd, 'read_parquet', lambda path: companies)
    monkeypatch.setattr(audit, 'read', lambda path: {'start': '2022-01-03', 'signal_end': '2022-01-05'})
    monkeypatch.setattr(audit, 'cutoff_plan', lambda *args: [{'cutoff': '2022-01-05'}])
    def fail(*args, **kwargs):
        raise RuntimeError('Explicit builder failure')
    monkeypatch.setattr(audit, 'build_signals', fail)
    report = audit.run(tmp_path / 'failed-build.json')
    assert report['stage'] == 'full_rebuild'
    assert report['status'] == 'failed' and not report['passed']
    assert report['error']['message'] == 'Explicit builder failure'
    assert report['cases'] == []


def test_network_is_blocked_and_recorded():
    report = {'network_attempts_blocked': 0}
    before = socket.create_connection
    with audit.offline_only(report):
        with pytest.raises(RuntimeError, match='prohibited'):
            socket.create_connection(('example.com', 443))
    assert report['network_attempts_blocked'] == 1
    assert socket.create_connection is before


def test_existing_evidence_is_never_overwritten(tmp_path):
    output = tmp_path / 'existing.json'
    output.write_text('preserve evidence')
    with pytest.raises(ValueError, match='Output exists'):
        audit.run(output)
    assert output.read_text() == 'preserve evidence'


def test_budget_exhaustion_cannot_report_success(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'provenance', lambda: {})
    clock = iter([0., 2., 3.])
    monkeypatch.setattr(audit.time, 'perf_counter', lambda: next(clock))
    report = audit.run(tmp_path / 'timeout.json', max_seconds=1)
    assert report['status'] == 'failed' and not report['complete']
    assert report['error']['type'] == 'TimeoutError'
