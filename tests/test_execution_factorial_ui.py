from copy import deepcopy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import execution_factorial_ui as ui
from scripts.research_execution_factorial import analyze
from scripts.research_exit_scenarios import summarize, write, sha


def publication(root):
    account = dict(settings={'initial_cash': 1_000_000},
        daily=[dict(date=day, opening_nav=1_000_000., nav=1_000_000., total_return=0.,
            cash=1_000_000., market_value=0., receivable=0., drawdown=0., stale_holdings=0)
            for day in ('2022-01-03', '2026-09-09')],
        trades=[], orders=[], holdings=[], receivables=[], cohorts=[])
    cases, rows, refs = {}, {}, {}
    names = ['factor_' + str(m) for m in range(8)] + ['noop_depth', 'noop_quote', 'benchmark_control', 'benchmark_combined']
    for name in names:
        cases[name] = dict(completed=True, account=deepcopy(account), summary=summarize(account),
            config={'factor_mask': int(name[7:]) if name.startswith('factor_') else None}, stock_pnl={})
        path = root / (name + '.json')
        write(path, cases[name])
        refs[path.name] = sha(path)
        rows[name] = dict(completed=True, summary=deepcopy(cases[name]['summary']),
            result=dict(path=path.name, sha256=sha(path)))
    manifest = root / 'manifest.json'
    write(manifest, {'files_sha256': refs})
    refs[manifest.name] = sha(manifest)
    proof = root / 'proof.json'
    write(proof, dict(schema='execution_factorial_offline_v1', passed=True, all_completed=True,
        compared_cases=12, source_sha256=refs))
    return dict(schema='execution_factorial_publication_v1', all_completed=True, candidate_count=454,
        labels=list(ui.LABELS), live_qualified=False, strict_data_ready=False, unseen_validation=False,
        cases=rows, analysis=analyze(cases),
        offline_verification=dict(path=proof.name, sha256=sha(proof)),
        run_manifest=dict(path=manifest.name, sha256=sha(manifest)))


def test_validation_requires_all_cases_and_reconciled_analysis(tmp_path):
    value = publication(tmp_path)
    assert ui.validate(value, tmp_path) is value
    value['analysis']['total_return_change'] = 1.
    with pytest.raises(ValueError, match='歸因'):
        ui.validate(value, tmp_path)
    value = publication(tmp_path)
    del value['cases']['factor_4']
    with pytest.raises(ValueError, match='完整因素'):
        ui.validate(value, tmp_path)


def test_mismatched_account_proof_and_qualification_are_rejected(tmp_path):
    value = publication(tmp_path)
    path = tmp_path / 'other.json'
    write(path, json.loads((tmp_path / 'factor_0.json').read_text()))
    value['cases']['factor_0']['result'] = dict(path=path.name, sha256=sha(path))
    with pytest.raises(ValueError, match='同一份'):
        ui.validate(value, tmp_path)
    value['live_qualified'] = True
    with pytest.raises(ValueError, match='資格'):
        ui.validate(value, tmp_path)


def test_source_change_invalidates_verified_display_cache(monkeypatch, tmp_path):
    leaf = tmp_path / 'source.txt'; leaf.write_text('source')
    report = tmp_path / 'report.json'
    write(report, dict(source_sha256={leaf.name: sha(leaf)}))
    report.with_suffix('.sha256').write_text(sha(report))
    monkeypatch.setattr(ui, 'validate', lambda value, root: value)
    ui.load(report, tmp_path)
    leaf.write_text('changed source')
    with pytest.raises(ValueError, match='版本已改變'):
        ui.load(report, tmp_path)


def test_display_all_combinations_and_selected_account_exports(monkeypatch, tmp_path):
    value = publication(tmp_path)
    monkeypatch.setattr(ui, 'ROOT', tmp_path)
    monkeypatch.setattr(ui, 'REPORT', Path(__file__))
    monkeypatch.setattr(ui, 'load', lambda: deepcopy(value))
    app = AppTest.from_string('from app.execution_factorial_ui import render\nrender()').run()
    assert not app.exception
    assert app.dataframe[0].value['情境'].tolist() == list(ui.LABELS)
    assert any('尚未通過未見資料' in row.value for row in app.warning)
    app.selectbox[0].set_value(7).run()
    app.checkbox[0].check().run()
    assert not app.exception
    assert len(app.get('download_button')) == 4


def test_missing_evidence_hides_all_returns(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, 'REPORT', tmp_path / 'absent.json')
    app = AppTest.from_string('from app.execution_factorial_ui import render\nrender()').run()
    assert not app.exception and not app.dataframe
    assert app.info
