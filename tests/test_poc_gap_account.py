"""New gap scenarios preserve source identity and unfinished financial evidence."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from scripts import research_poc_gap_account as study
from skills import poc_executable_data as data
from skills.poc_gap_execution import DataGapOrders


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    old = tmp_path/'strict.md'; old.write_text('strict acquisition limits')
    new = tmp_path/'gaps.md'; new.write_text('explicit no-fill data gaps')
    frozen = tmp_path/'strict.py'; frozen.write_text('sealed matcher')
    monkeypatch.setattr(study, 'PREREG', new)
    monkeypatch.setattr(study, 'GAP_BINDINGS', {'strict.py': digest(frozen)})
    p = data.ExecutableAccountData.__new__(data.ExecutableAccountData)
    p.refs = {'strict.md': digest(old)}
    p.prereg_path, p.prereg_sha256 = old, digest(old)
    p.board_ticks = object(); p.profiles = object(); p.after_hours = object()
    p.execution = object(); p.online = False
    return p, old, new, frozen


def test_new_policy_binds_without_replacing_data_clients_or_budget(provider):
    p, old, new, frozen = provider
    original = {key: getattr(p, key) for key in ('board_ticks','profiles','after_hours','execution')}
    assert study.bind_gap_sources(p) is p
    assert p.prereg_path == new and p.prereg_sha256 == digest(new)
    assert p.refs == {'strict.md': digest(old), 'strict.py': digest(frozen), 'gaps.md': digest(new)}
    assert all(getattr(p, key) is value for key, value in original.items())
    assert p.online is False


def test_corrupted_strict_matcher_is_fatal_not_an_order_gap(provider):
    p, old, _, frozen = provider
    frozen.write_text('changed matcher')
    with pytest.raises(ValueError, match='source changed'):
        study.bind_gap_sources(p)
    assert p.prereg_path == old


def test_direct_runner_rechecks_source_before_running(provider, tmp_path, monkeypatch):
    p, _, _, frozen = provider
    study.bind_gap_sources(p)
    monkeypatch.setattr(study, 'INPUTS', tmp_path/'inputs')
    monkeypatch.setattr(study, 'RUN_ROOT', tmp_path/'outputs')
    monkeypatch.setattr(study, 'FROZEN_BINDINGS', {})
    p.bundle = study.INPUTS; p.verify_sources = lambda: None
    frozen.write_text('changed after binding')
    output = study.RUN_ROOT/'case'
    with pytest.raises(ValueError, match='Changed frozen financial source'):
        study._run(output, study.ARMS, p)
    assert not output.exists()


def test_failure_keeps_gap_journal_and_real_unsold_holdings():
    gap = dict(date='2024-11-12', stock_id='6151', side='sell', filled_qty=0,
               holding_qty_before=5026, holding_qty_after=5026, retry_sell=True)
    engine = SimpleNamespace(daily=[dict(date='2024-11-12')], holdings={'6151':dict(qty=5026)},
        data_gap_exclusions=[deepcopy(gap)], trades=[], orders=[], cash_ledger=[], actions=[],
        holding_rows=[], cohorts=[], receivables=[], resource_plans=[dict(spent=0)],
        tick_plans=[dict(planned_qty=5026)], day_plans={}, selection_decisions=[],
        slot_decisions=[], board_decisions=[], residual_days=[])
    case = study.preserve_failure(ValueError('Unresolved corporate valuation'), engine)
    assert case['summary'] is None and case['completed'] is False
    assert case['partial_journal']['data_gap_exclusions'] == [gap]
    assert case['failure_holdings']['6151']['qty'] == 5026
    engine.data_gap_exclusions[0]['filled_qty'] = 5026
    engine.holdings['6151']['qty'] = 0
    assert case['partial_journal']['data_gap_exclusions'][0]['filled_qty'] == 0
    assert case['failure_holdings']['6151']['qty'] == 5026


def test_both_strategy_and_benchmark_use_same_gap_executor_before_midpoint():
    stock, benchmark = study.engine_types(DataGapOrders, study.HistoricalOddEra)
    for cls in (stock, benchmark):
        assert cls.mro().index(DataGapOrders) < cls.mro().index(study.MidpointExitReplay if cls is stock else study.MidpointBenchmark)
        assert cls._execute_order is DataGapOrders._execute_order


def test_buy_order_gaps_do_not_prefilter_the_original_red_candle_candidates():
    entries = [dict(event_id='gap', members=['3230'], signal_date='2024-10-15', entry_date='2024-10-16'),
               dict(event_id='normal', members=['2330'], signal_date='2024-10-15', entry_date='2024-10-16')]
    class Red:
        def filter_entries(self, items): return items, []
    kept, _ = study.select_arm_entries('poc_red_executable', entries, Red(), study.START, study.END)
    assert kept == entries


def test_original_matcher_and_runner_remain_sealed():
    for name, expected in study.GAP_BINDINGS.items():
        assert digest(study.ROOT/name) == expected
