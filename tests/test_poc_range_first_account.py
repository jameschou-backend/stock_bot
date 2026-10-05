"""Daily range study preserves frozen financial policy and a controlled factorial design."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import research_poc_range_first_account as study
from skills import poc_executable_data as data
from skills.poc_range_execution import RangeGapOrders


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
    monkeypatch.setattr(study, 'RANGE_BINDINGS', {'strict.py': digest(frozen)})
    p = data.ExecutableAccountData.__new__(data.ExecutableAccountData)
    p.refs = {'strict.md': digest(old)}
    p.prereg_path, p.prereg_sha256 = old, digest(old)
    p.board_ticks = object(); p.profiles = object(); p.after_hours = object()
    p.execution = object(); p.online = False
    return p, old, new, frozen


def test_new_policy_binds_without_replacing_data_clients_or_budget(provider):
    p, old, new, frozen = provider
    original = {key: getattr(p, key) for key in ('board_ticks','profiles','after_hours','execution')}
    assert study.bind_range_sources(p) is p
    assert p.prereg_path == new and p.prereg_sha256 == digest(new)
    assert p.refs == {'strict.md': digest(old), 'strict.py': digest(frozen), 'gaps.md': digest(new)}
    assert all(getattr(p, key) is value for key, value in original.items())
    assert p.online is False


def test_corrupted_strict_matcher_is_fatal_not_an_order_gap(provider):
    p, old, _, frozen = provider
    frozen.write_text('changed matcher')
    with pytest.raises(ValueError, match='source changed'):
        study.bind_range_sources(p)
    assert p.prereg_path == old


def test_direct_runner_rechecks_source_before_running(provider, tmp_path, monkeypatch):
    p, _, _, frozen = provider
    study.bind_range_sources(p)
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
    stock, benchmark = study.engine_types(RangeGapOrders, study.HistoricalOddEra)
    for cls in (stock, benchmark):
        assert cls.mro().index(RangeGapOrders) < cls.mro().index(study.MidpointExitReplay if cls is stock else study.MidpointBenchmark)
        assert cls._execute_order is RangeGapOrders._execute_order


def test_buy_order_gaps_do_not_prefilter_the_original_red_candle_candidates():
    entries = [dict(event_id='gap', members=['3230'], signal_date='2024-10-15', entry_date='2024-10-16'),
               dict(event_id='normal', members=['2330'], signal_date='2024-10-15', entry_date='2024-10-16')]
    class Red:
        def filter_entries(self, items): return items, []
    kept, _, first = study.select_arm_entries('poc_range50_all', entries, Red(), study.START, study.END)
    assert kept == entries
    assert first == []


def test_original_matcher_and_runner_remain_sealed():
    for name, expected in study.RANGE_BINDINGS.items():
        assert digest(study.ROOT/name) == expected


def test_matrix_contains_four_controlled_strategy_arms_and_two_price_matched_benchmarks():
    strategy = [study.ARM_RULES[a] for a in study.ARMS if not study.is_benchmark(a)]
    assert {(r[2], r[3], r[4]) for r in strategy} == {
        (.5, .5, False), (.7, .3, False), (.5, .5, True), (.7, .3, True)}
    assert all(r[:2] == (True, 'none') for r in strategy)
    benchmarks = [study.ARM_RULES[a] for a in study.ARMS if study.is_benchmark(a)]
    assert benchmarks == [(False, 'none', .5, .5, False), (False, 'none', .7, .3, False)]
    with pytest.raises(ValueError, match='Unregistered'):
        study.is_benchmark('best_after_seeing_results')


def test_first_arm_uses_full_history_before_scoping_and_does_not_consult_cash(monkeypatch):
    from skills import poc_first_signal
    events = [dict(event_id='prior', entry_date='2023-12-29'),
              dict(event_id='first', entry_date='2024-01-02')]
    token = object()
    def gate(entries, candle, start, end):
        assert entries is events and candle is token
        assert (start, end) == (study.START, study.END)
        return events[:1], [{'red': True}], [{'passed': False}]
    monkeypatch.setattr(poc_first_signal, 'first_signal_entries', gate)
    result = study.select_arm_entries('poc_range70_30_first', events, token, study.START, study.END)
    assert result == (events[:1], [{'red': True}], [{'passed': False}])


@pytest.fixture
def settlement_document(tmp_path):
    evidence = tmp_path/'official.html'; evidence.write_text('dated issuer terms')
    common = dict(use_scope='account_settlement_only_not_selection', evidence_files=['official.html'])
    rows = {
        '2515-2025-11-10': dict(common, shares_per_share=.0503, pay_date='2025-12-31',
            fractional_cash_per_share=0, entitlement_announcement_date='2025-10-27',
            delivery_announcement_date='2025-12-22'),
        '5876-2026-07-28': dict(common, shares_per_share=.01, pay_date='2026-08-20',
            fractional_cash_per_share=10, entitlement_announcement_date='2026-07-13',
            delivery_announcement_date='2026-08-17', certificate_delivery_date='2026-08-20',
            ordinary_conversion_date='2026-09-11', certificate_trading_modeled=True,
            certificate_stock_id='5876', same_code_fungible_trading_verified=True,
            fractional_cash_rounding='floor_ntd',
            fractional_cash_pay_date=None),
    }
    value = dict(schema='poc_range_settlement_repair_v1', strategy_parameters_changed=False,
                 cash_supplements=[], overrides=rows, source_sha256={'official.html':digest(evidence)})
    path = tmp_path/'docs/poc_range_corporate_terms_20261005.json'; path.parent.mkdir()
    path.write_text(json.dumps(value))
    return tmp_path, path, value, evidence


def test_settlement_patch_binds_sources_and_returns_independent_terms(settlement_document):
    root, path, value, _ = settlement_document
    rows, refs = study.load_range_corporate_terms(root)
    assert refs == {'official.html':value['source_sha256']['official.html'],
                    'docs/poc_range_corporate_terms_20261005.json':digest(path)}
    rows['2515-2025-11-10']['shares_per_share'] = 99
    assert study.load_range_corporate_terms(root)[0]['2515-2025-11-10']['shares_per_share'] == .0503


@pytest.mark.parametrize('mutation', ['source', 'scope', 'early_shares', 'funded_fraction', 'future_entitlement'])
def test_settlement_patch_rejects_corruption_and_early_spend(settlement_document, mutation):
    root, path, value, evidence = settlement_document
    bank = value['overrides']['5876-2026-07-28']
    if mutation == 'source': evidence.write_text('changed official page')
    elif mutation == 'scope': value['overrides']['9999-2025-01-01'] = bank
    elif mutation == 'early_shares': bank['pay_date'] = '2026-08-19'
    elif mutation == 'funded_fraction': bank['fractional_cash_pay_date'] = '2026-08-20'
    else: bank['entitlement_announcement_date'] = '2026-08-01'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): study.load_range_corporate_terms(root)


def test_fungible_bank_certificate_is_delivered_once_and_fraction_is_not_spendable(settlement_document):
    from skills.scenario_exit_replay import FractionalCashActions
    root, _, _, _ = settlement_document
    terms, _ = study.load_range_corporate_terms(root)
    class Provider:
        overrides = terms
        def on_date(self, sid, day):
            return [dict(action_id='5876-2026-07-28-stock', stock_id=sid, date=day,
                kind='stock_dividend', shares_per_share=.01, fractional_cash_per_share=10,
                pay_date='2026-08-20', source='issuer evidence'),
                dict(action_id='5876-2026-07-28-cash', stock_id=sid, date=day,
                     kind='cash_dividend', cash_per_share=1.8, pay_date='2026-08-20')]
    account = SimpleNamespace(holdings={'5876':dict(qty=17517)})
    rows = FractionalCashActions(Provider(), account).on_date('5876','2026-07-28')
    fractional, stock, cash = rows
    assert fractional['gross_cash_amount'] == 1 and fractional['pay_date'] is None
    assert stock['pay_date'] == '2026-08-20' and stock['fractional_cash_per_share'] == 0
    assert 'certificate_restriction' not in stock
    assert terms['5876-2026-07-28']['ordinary_conversion_date'] == '2026-09-11'
    assert cash['pay_date'] == '2026-08-20'
