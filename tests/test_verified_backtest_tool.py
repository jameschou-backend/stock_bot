from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import socket

import pandas as pd
import pytest

from skills import verified_backtest_tool as tool
from skills.conservative_diversification import ConservativeDiversification
from skills.execution_resources import ResourceBenchmark
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture, ENTRY


@pytest.fixture
def local_runner(tmp_path, monkeypatch):
    days, adjusted, args, kwargs = fixture(entries=[ENTRY])
    events = deepcopy(args[3])
    for row in events:
        row.update(group_cutoff_date=row['signal_date'], trend_decision_date=row['signal_date'])
        liquidity = dict(as_of=row['signal_date'], complete_20_sessions=True, observations=20,
                         adv20_shares=2000000, mean_turnover20_twd=100000000)
        row.update(liquidity_at_signal=liquidity, liquidity_before_entry=deepcopy(liquidity))
    data = SimpleNamespace(days=days, entries=events, start=pd.Timestamp(kwargs['start']), end=pd.Timestamp(kwargs['end']))
    cases = list(tool.configurations('mixed', 'control'))
    identity = dict(recipe='test_recipe', initial_cash=1000000, candidate_count=1, source_sha256={},
                    start=kwargs['start'], end=kwargs['end'])
    originals = {}
    for name, config in cases:
        engine = (ResourceBenchmark(*args, **kwargs, opening_cash_only=True, lock_unused=True)
                  if config['benchmark'] else ConservativeDiversification(*args, **kwargs, position_count=5,
                      exit_signals=ExitSignals(adjusted, days)))
        account = engine.run()
        originals[name] = dict(config=config, account=account, summary=tool.summarize(account),
            resource_plans=engine.resource_plans, slot_decisions=getattr(engine, 'slot_decisions', []),
            completed=True, parent_account_identical=True, live_qualified=False, unseen_validation=False)
    sources = tmp_path / 'sources'
    tool.write(sources / 'overrides.json', {'overrides': {}})
    monkeypatch.setattr(tool, 'ROOT', tmp_path)
    monkeypatch.setattr(tool, 'RUNS', tmp_path / '.cache/backtest-tool/runs')
    monkeypatch.setattr(tool.sealed, 'SOURCES', sources)
    monkeypatch.setattr(tool, 'source_context', lambda: (deepcopy(identity), deepcopy(originals)))
    monkeypatch.setattr(tool.sealed.parent.source, 'inputs', lambda: (data, None))
    calls = []
    def execute(data, config, inputs, additions):
        name = next(name for name, conf in cases if conf == config)
        calls.append(name)
        return deepcopy(originals[name])
    monkeypatch.setattr(tool.sealed, 'run_case', execute)
    return tmp_path, originals, calls, data


def invoke(root, filename, **kwargs):
    return tool.run(output=root / '.cache/backtest-tool' / filename, policy='mixed', stress='control', **kwargs)


def test_complete_accounts_resume_without_reexecuting_and_exports_match(local_runner):
    root, originals, calls, _ = local_runner
    first = invoke(root, 'first.json')
    second = invoke(root, 'second.json')
    assert first['status'] == second['status'] == 'exploratory'
    assert len(calls) == 2
    assert first['metrics']['executed_cases'] == 2 and second['metrics']['reused_cases'] == 2
    assert [r['total_return'] for r in first['case_rows']] == [r['total_return'] for r in second['case_rows']]
    assert len(first['comparisons']) == 1 and all(r['cache_hit'] for r in second['case_rows'])
    for row in second['case_rows']:
        frame = pd.read_csv(root / row['artifact_paths']['daily']['path'])
        assert frame.iloc[-1]['nav'] == originals[row['name']]['summary']['final_nav']
        assert len(frame) == len(originals[row['name']]['account']['daily'])


def test_fresh_reexecutes_and_checks_against_the_cache(local_runner):
    root, _, calls, _ = local_runner
    invoke(root, 'first.json')
    report = invoke(root, 'fresh.json', fresh=True)
    assert len(calls) == 4 and report['metrics']['executed_cases'] == 2
    assert not any(row['cache_hit'] for row in report['case_rows'])


def test_preflight_and_strict_never_execute_an_account(local_runner):
    root, _, calls, _ = local_runner
    assert invoke(root, 'preflight.json', preflight_only=True)['status'] == 'preflight_ready'
    strict = invoke(root, 'strict.json', mode='strict')
    assert strict['status'] == 'blocked' and not calls
    assert all(row['total_return'] is None and not row['artifact_paths'] for row in strict['case_rows'])


def test_interrupted_batch_resumes_the_finished_case(local_runner, monkeypatch):
    root, originals, calls, _ = local_runner
    original = tool.sealed.run_case
    def interrupted(*args):
        if args[1]['benchmark']:
            raise RuntimeError('interrupted')
        return original(*args)
    monkeypatch.setattr(tool.sealed, 'run_case', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        invoke(root, 'first.json')
    assert not (root / '.cache/backtest-tool/first.json').exists()
    monkeypatch.setattr(tool.sealed, 'run_case', original)
    resumed = invoke(root, 'resume.json')
    assert resumed['metrics']['reused_cases'] == 1 and resumed['metrics']['executed_cases'] == 1
    assert len(calls) == 2


def test_bad_cash_ledger_cannot_receive_performance_statistics(local_runner):
    root, originals, _, _ = local_runner
    originals['capacity_control_mixed']['account']['cash_ledger'][1]['cash_change'] -= 1
    with pytest.raises(ValueError, match='reconcile'):
        invoke(root, 'bad.json')
    assert not (root / '.cache/backtest-tool/bad.json').exists()


def test_output_relative_path_is_normalized_and_cannot_clobber_source(local_runner, monkeypatch):
    root, _, _, _ = local_runner
    monkeypatch.chdir(root)
    tool.run(output=Path('.cache/backtest-tool/relative.json'), policy='mixed', stress='control', preflight_only=True)
    assert (root / '.cache/backtest-tool/relative.json').is_file()
    tool.write(root / '.cache/backtest-tool/source.json', {'source': True})
    with pytest.raises(ValueError, match='overwrite'):
        invoke(root, 'source.json')


def test_new_report_cannot_be_written_into_a_sealed_source_folder(local_runner):
    root, _, _, _ = local_runner
    with pytest.raises(ValueError, match='report under'):
        tool.run(output=root / '.cache/sealed-evidence/new-report.json', preflight_only=True)
    assert not (root / '.cache/sealed-evidence').exists()


@pytest.mark.parametrize('preflight_only', [False, True])
def test_source_changes_during_a_warm_run_or_preflight_prevent_publication(local_runner, monkeypatch, preflight_only):
    root, _, _, _ = local_runner
    invoke(root, 'first.json')
    monkeypatch.setattr(tool, 'file_identities', lambda *args: {'changed-source': 'different'})
    with pytest.raises(ValueError, match='Sources changed'):
        invoke(root, 'changed.json', preflight_only=preflight_only)
    assert not (root / '.cache/backtest-tool/changed.json').exists()


def test_network_access_is_explicitly_blocked():
    with tool.offline_only(), pytest.raises(RuntimeError, match='offline'):
        socket.create_connection(('example.com', 443))
