"""Prevent a result report from silently selecting or relabeling replay attempts."""
import json
from pathlib import Path

import pytest

from scripts import analyze_candle_volume_account as analysis


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return analysis.sha(path)


def reports(tmp_path, first_complete, second_mode='dry'):
    paths = []
    for index, complete in enumerate((first_complete, True)):
        case = Path('.cache') / str(index) / 'case.json'
        value = dict(completed=complete, family_rules=dict(red_gate=False,
            volume_exit_mode='dry' if index == 0 else second_mode))
        digest = save(tmp_path / case, value)
        report = tmp_path / '.cache' / str(index) / 'report.json'
        save(report, dict(source_sha256={}, cases={'poc_dry': dict(path=str(case), sha256=digest)}))
        paths.append(report)
    return paths


def test_completed_result_cannot_be_replaced_even_with_completion_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Cannot replace duplicate/completed'):
        analysis.analyze(reports(tmp_path, True),
                         tmp_path / '.cache/red-volume-exit-20261003/check', replace_incomplete=True)


def test_completing_failed_result_cannot_change_strategy_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='changed the registered strategy rules'):
        analysis.analyze(reports(tmp_path, False, 'dry_weak'),
                         tmp_path / '.cache/red-volume-exit-20261003/check', replace_incomplete=True)


def test_failed_result_replacement_requires_explicit_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Cannot replace duplicate/completed'):
        analysis.analyze(reports(tmp_path, False), tmp_path / '.cache/red-volume-exit-20261003/check')


def test_new_manifest_cannot_replace_parent_account_input_version(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, 'ROOT', tmp_path)
    parent = reports(tmp_path, True)[0]
    save(tmp_path / '.cache/market-input-repair-20261002/inputs-v2/manifest.json', {})
    with pytest.raises(ValueError, match='differ from the bound parent account: manifest'):
        analysis.analyze([parent], tmp_path / '.cache/red-volume-exit-20261003/check')
