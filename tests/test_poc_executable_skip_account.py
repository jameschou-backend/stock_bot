"""Boundary contracts for the explicitly excluded one-order research account."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from scripts import research_poc_executable_skip as study
from skills import poc_executable_data as provider_module
from skills.poc_executable_replay import ExecutableOrders


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bounded_provider(tmp_path, monkeypatch):
    """Use the real binding method without opening the full historical cache."""
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(provider_module, 'ROOT', tmp_path)
    original = tmp_path / 'strict-prereg.md'
    original.write_text('Frozen acquisition registration')
    supplemental = tmp_path / 'skip-prereg.md'
    supplemental.write_text('Exclude exactly the registered conflicting buy')
    source = tmp_path / 'strict-case.json'
    source.write_text('{"completed":false,"last_date":"2024-10-15"}')
    monkeypatch.setattr(study, 'PREREG', supplemental)
    monkeypatch.setattr(study, 'SKIP_BINDINGS', {'strict-case.json': digest(source)})
    provider = provider_module.ExecutableAccountData.__new__(provider_module.ExecutableAccountData)
    provider.refs = {'strict-prereg.md': digest(original)}
    provider.prereg_path, provider.prereg_sha256 = original, digest(original)
    provider.online = False
    # Binding a new experiment must not replace these acquisition objects or
    # recreate their persistent counters and origin holds under a fresh root.
    provider.profiles = SimpleNamespace(directory=tmp_path/'profiles', calls=155)
    provider.board_ticks = SimpleNamespace(directory=tmp_path/'board', calls=52)
    provider.execution = SimpleNamespace(directory=tmp_path/'financial', calls=8)
    provider.after_hours = SimpleNamespace(directory=tmp_path/'odd', calls=116)
    attempt = tmp_path/'attempt.json'
    attempt.write_text('{"status":"started","request_number":52}')
    hold = tmp_path/'hold.json'
    hold.write_text('{"blocked_until":1791120999,"reason":"original hold"}')
    return provider, original, supplemental, source, attempt, hold


def test_registration_adds_provenance_without_resetting_acquisition(bounded_provider):
    provider, original, supplemental, source, attempt, hold = bounded_provider
    components = {key: getattr(provider, key) for key in
                  ('profiles', 'board_ticks', 'execution', 'after_hours')}
    component_state = {key: deepcopy(vars(value)) for key, value in components.items()}
    files_before = {path: path.read_bytes() for path in (original, source, attempt, hold)}
    assert study.bind_skip_sources(provider) is provider
    assert provider.refs == {'strict-prereg.md': digest(original),
                             'strict-case.json': digest(source),
                             'skip-prereg.md': digest(supplemental)}
    assert provider.prereg_path == supplemental
    assert provider.prereg_sha256 == digest(supplemental)
    assert provider.online is False
    for key, value in components.items():
        assert getattr(provider, key) is value
        assert vars(value) == component_state[key]
    for path, data in files_before.items():
        assert path.read_bytes() == data


def test_changed_strict_evidence_cannot_be_rebound_as_a_skip(bounded_provider):
    provider, original, supplemental, source, *_ = bounded_provider
    source.write_text('{"completed":true,"fabricated":true}')
    with pytest.raises(ValueError, match='Executable source changed'):
        study.bind_skip_sources(provider)
    assert provider.prereg_path == original
    assert provider.prereg_sha256 == digest(original)
    assert 'skip-prereg.md' not in provider.refs


def test_skip_registration_binding_conflict_is_not_silently_overwritten(bounded_provider):
    provider, _, supplemental, *_ = bounded_provider
    provider.refs['skip-prereg.md'] = '0' * 64
    with pytest.raises(ValueError):
        study.bind_skip_sources(provider)
    assert provider.refs['skip-prereg.md'] == '0' * 64


def test_direct_runner_checks_frozen_evidence_before_account_work(bounded_provider, tmp_path, monkeypatch):
    provider, _, _, source, *_ = bounded_provider
    study.bind_skip_sources(provider)
    monkeypatch.setattr(study, 'INPUTS', tmp_path/'inputs')
    monkeypatch.setattr(study, 'RUN_ROOT', tmp_path/'outputs')
    monkeypatch.setattr(study, 'FROZEN_BINDINGS', {})
    provider.bundle = study.INPUTS
    provider.verify_sources = lambda: None
    source.write_text('{"changed_after_binding":true}')
    output = study.RUN_ROOT/'case'
    with pytest.raises(ValueError, match='Changed frozen financial source'):
        study._run(output, ['poc_red_executable'], provider)
    assert not output.exists()


def test_new_registration_cannot_change_after_provider_binding(bounded_provider, tmp_path, monkeypatch):
    provider, _, supplemental, *_ = bounded_provider
    study.bind_skip_sources(provider)
    monkeypatch.setattr(study, 'INPUTS', tmp_path/'inputs')
    monkeypatch.setattr(study, 'RUN_ROOT', tmp_path/'outputs')
    provider.bundle = study.INPUTS
    provider.verify_sources = lambda: None
    supplemental.write_text('Now skip all losing sells')
    output = study.RUN_ROOT/'case'
    with pytest.raises(ValueError, match='Latest preregistration changed'):
        study._run(output, ['poc_red_executable'], provider)
    assert not output.exists()


def test_failed_account_keeps_exclusions_and_original_resource_lock_evidence():
    plan = dict(date='2024-10-16', stock_id='3230', side='buy', planned_qty=4958,
                board_qty=4000, odd_qty=958, reserved_cash=346859.0366666667)
    exclusion = dict(date='2024-10-16', stock_id='3230', side='buy', filled_qty=0,
                     event_id='liquid_universe-2024-10-15-3230')
    engine = SimpleNamespace(
        daily=[dict(date='2024-10-16', nav=1_038_463.11)],
        holdings={'3081': {'qty': 1004}}, trades=[], orders=[deepcopy(exclusion)],
        cash_ledger=[], actions=[], holding_rows=[], cohorts=[], receivables=[],
        resource_plans=[deepcopy(plan)], tick_plans=[deepcopy(plan)],
        day_plans={('event', 'buy'): deepcopy(plan)},
        selection_decisions=[dict(event_id='original-candidate', selected=True)],
        explicit_exclusions=[deepcopy(exclusion)],
        slot_decisions=[dict(date='2024-10-16', available=1, reserved=1)],
        board_decisions=[dict(date='2024-10-16', lock_unused=True)],
        residual_days=[dict(date='2024-10-16', residual_value=0)])
    result = study.preserve_failure(ValueError('Another unapproved data conflict'), engine)
    assert result['completed'] is False and result['summary'] is None
    assert result['last_date'] == '2024-10-16' and result['completed_sessions'] == 1
    journal = result['partial_journal']
    for key in ('resource_plans', 'tick_plans', 'selection_decisions',
                'explicit_exclusions', 'slot_decisions', 'board_decisions', 'residual_days'):
        assert journal[key] == getattr(engine, key)
    assert journal['day_plans'] == [plan]
    assert result['failure_holdings'] == engine.holdings
    engine.explicit_exclusions[0]['filled_qty'] = 999
    engine.resource_plans[0]['reserved_cash'] = 0
    engine.slot_decisions[0]['reserved'] = 0
    engine.day_plans[('event', 'buy')]['planned_qty'] = 0
    engine.holdings['3081']['qty'] = 0
    assert journal['explicit_exclusions'] == [exclusion]
    assert journal['resource_plans'] == [plan]
    assert journal['slot_decisions'][0]['reserved'] == 1
    assert journal['day_plans'] == [plan]
    assert result['failure_holdings']['3081']['qty'] == 1004


def test_setup_failure_cannot_be_presented_as_a_zero_return():
    result = study.preserve_failure(ValueError('Changed source'), None, 'setup')
    assert result['completed'] is False and result['summary'] is None
    assert result['stage'] == 'setup' and result['last_date'] is None
    assert 'partial_journal' not in result


def test_skip_stock_and_untouched_benchmark_dispatch_ahead_of_hl2():
    from skills.poc_executable_skip import ExplicitConflictSkipOrders
    stock, _ = study.engine_types(ExplicitConflictSkipOrders, study.HistoricalOddEra)
    _, benchmark = study.engine_types(ExecutableOrders, study.HistoricalOddEra)
    assert stock._execute_order is ExplicitConflictSkipOrders._execute_order
    assert benchmark._execute_order is ExecutableOrders._execute_order
    assert stock._plan is ExecutableOrders._plan
    assert benchmark._plan is ExecutableOrders._plan
    assert ExplicitConflictSkipOrders not in benchmark.__mro__
    assert stock.__mro__.index(ExecutableOrders) < stock.__mro__.index(study.MidpointExitReplay)
    assert benchmark.__mro__.index(ExecutableOrders) < benchmark.__mro__.index(study.MidpointBenchmark)


def test_frozen_source_bindings_keep_existing_strict_implementation_immutable():
    # Tracked source files are always available in CI; historical cache data is
    # independently checked by the provider rather than required by this test.
    for name in ('scripts/research_poc_executable_account.py', 'skills/poc_executable_replay.py'):
        assert digest(study.ROOT/name) == study.SKIP_BINDINGS[name]
    assert study.SKIP_BINDINGS[str(study.STRICT_CASE.relative_to(study.ROOT))] == study.STRICT_CASE_SHA
    assert study.RUN_ROOT != study.STRICT_CASE.parent.parent
    assert study.PREREG.name == 'prereg_poc_executable_skip_20261004.md'
    prereg = study.PREREG.read_text()
    assert 'liquid_universe-2024-10-15-3230' in prereg
    assert '4,958' in prereg and '4,000' in prereg and '958' in prereg
    assert '任何其他缺件或衝突仍停止' in prereg
    assert '不得自動排除賣單' in prereg
    assert 'posthoc_data_exclusion=true' in prereg
    for flag in ('actual_fill_verified=false', 'live_qualified=false', 'unseen_validation=false'):
        assert flag in prereg


@pytest.mark.parametrize('arm', ['all_conflicts_skipped', 'benchmark_skip', 'poc_red_executable,poc_red_executable'])
def test_runner_does_not_accept_an_implicit_broader_skip_policy(tmp_path, monkeypatch, arm):
    monkeypatch.setattr(study, 'RUN_ROOT', tmp_path)
    with pytest.raises(ValueError, match='two preregistered'):
        study._run(tmp_path/'out', arm.split(','), None)


def test_conflicting_order_remains_in_signal_population_before_planning():
    target = dict(event_id='liquid_universe-2024-10-15-3230', members=['3230'],
                  signal_date='2024-10-15', entry_date='2024-10-16', priority=5)
    other = dict(event_id='next-candidate', members=['2330'],
                 signal_date='2024-10-15', entry_date='2024-10-16', priority=4)
    class OriginalRedGate:
        def filter_entries(self, rows):
            return rows, [dict(event_id=row['event_id'], kept=True) for row in rows]
    rows = [target, other]
    kept, decisions = study.select_arm_entries('poc_red_executable', rows, OriginalRedGate(), study.START, study.END)
    assert kept == rows
    assert [r['event_id'] for r in decisions] == [target['event_id'], other['event_id']]
    assert target['event_id'] == kept[0]['event_id']
