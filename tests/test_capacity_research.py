"""Tiny sealed research fixtures; no historical simulations, DB or API."""
from copy import deepcopy
from datetime import date
import hashlib
import json
import os
from pathlib import Path

import pytest

from app import capacity_research as service


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def save_report(root, report):
    write(root, service.CACHE + '/report.summary.json', json.dumps(report))


def read_case(root, row):
    return json.loads((root / service.CACHE / row['case_file']).read_text())


def reseal_case(root, report, row, case):
    report['case_files_sha256'][row['case_file']] = write(
        root, service.CACHE + '/' + row['case_file'], json.dumps(case))


def reseal_case_header(root, report, row):
    case = read_case(root, row)
    for key in ('summary', 'valuation_audit', *service.IDENTITY):
        case[key] = deepcopy(row[key])
    reseal_case(root, report, row, case)


def reseal_signals(root, report):
    report['signal_manifest_sha256'] = write(root, service.CACHE + '/signals.json', json.dumps(report['signals']))
    save_report(root, report)


def summary(rule, scenario):
    years = (date(2026, 6, 23) - date(2022, 1, 3)).days / 365.25
    return {
        'mode': 'benchmark' if rule == 'benchmark' else 'events',
        'start': service.START, 'end': service.END,
        'requested_start': service.START, 'requested_end': service.END,
        'initial_nav': 1., 'final_nav': .99, 'final_cash': .99,
        'total_return': -.01, 'cagr': .99 ** (1 / years) - 1,
        'max_drawdown': -.01, 'total_cost': .01, 'turnover': 1.99,
        'mean_active_weight': 0., 'mean_cash_weight': 1.,
        'slots': 6 if rule == 'capacity6' else 3, 'horizon': 63, 'trade_count': 2,
        'completed_cohorts': 0, 'entered_cohorts': 0, 'peak_active_cohorts': 0,
        'blocked_exit_sessions': 0, 'rejected_event_count': 0,
        'unliquidated_position_count': 0, 'unliquidated_positions': [],
        'final_liquidation_complete': True, 'final_nav_is_marked': False,
        'annual_returns': {'2022': -.01, '2023': 0., '2024': 0., '2025': 0., '2026': 0.},
        'slippage_per_side': {'base': .003, 'stress': .0045}[scenario],
        'commission_per_side': .001425, 'stock_sell_tax': .003, 'benchmark_sell_tax': .001,
    }


def curve():
    days = [service.START, '2022-12-30', '2023-12-29', '2024-12-31', '2025-12-31', service.END]
    return [{'date': day, 'nav': .99, 'cash': .99, 'active_value': 0., 'active_weight': 0.,
             'benchmark_value': 0., 'active_cohorts': 0,
             'daily_cost': .01 if i == 0 else 0., 'cumulative_cost': .01, 'market_pnl': 0.}
            for i, day in enumerate(days)]


@pytest.fixture
def sealed(monkeypatch, tmp_path):
    monkeypatch.setattr(service, 'ROOT', tmp_path)
    monkeypatch.setattr(service, 'version', lambda name: 'fixture-runtime')
    monkeypatch.setattr(service.regime_switch_research, 'overview',
                        lambda: {'available': True, 'states': {'diffusion_signal_manifest_sha256': 'sealed-parent-signals'}})
    service._digest.cache_clear()
    service._case_header.cache_clear()
    codes = {name: write(tmp_path, name, '# sealed ' + name) for name in service.CODE}
    protocol = write(tmp_path, 'docs/prereg_capacity_20260910.md', 'Fixed synthetic protocol')
    parent_hash = write(tmp_path, service.PARENT_REPORT, 'sealed parent report')
    stats = {basis: {'original_count': 4, 'trend_count': 3, 'scoreable_trend_count': 2,
                     'score_rejections': {'missing_returns': 1}, 'multiple_event_days': 1,
                     'reordered_days': 1, 'reorder_examples': [{'date': '2023-01-03',
                         'original_order': ['event-A', 'event-B'], 'residual_order': ['event-B', 'event-A']}]}
             for basis in service.BASES}
    signals = {
        'schema': 1, 'code_sha256': codes, 'protocol_sha256': protocol,
        'prefix_invariance_passed': True, 'parent_report_sha256': parent_hash,
        'parent_signal_manifest_sha256': 'sealed-parent-signals',
        'versions': {name: 'fixture-runtime' for name in ('numpy', 'pandas')},
        'files_sha256': {name: write(tmp_path, service.CACHE + '/' + name, 'signal bytes ' + name)
                          for name in ('signals-official.json', 'signals-snapshot.json')},
        'stats': deepcopy(stats), 'elapsed_seconds': .1, 'finmind_requests': 0,
    }
    signal_hash = write(tmp_path, service.CACHE + '/signals.json', json.dumps(signals))
    combos = {(rule, basis, scenario, 0) for rule in service.RULES
              for basis in service.BASES for scenario in service.SCENARIOS}
    combos |= {(rule, 'official', 'stress', 1) for rule in service.RULES}
    benchmark_combos = {('benchmark', basis, scenario, 0)
                        for basis in service.BASES for scenario in service.SCENARIOS}
    results, baselines, cases = [], [], {}
    for rule, basis, scenario, delay in sorted(combos | benchmark_combos):
        filename = f'case-{rule}-{basis}-{scenario}-{delay}.json'
        row = {'rule': rule, 'basis': basis, 'scenario': scenario, 'delay': delay,
               'case_file': filename, 'name': rule, 'summary': summary(rule, scenario),
               'rejection_counts': {},
               'valuation_audit': {'finding_count': 0, 'unresolved_valuation_days': 0, 'findings': []}}
        if rule != 'benchmark':
            row.update(excess_vs_0050=0., contrast_vs_reference=0., reference_rule=service.REFERENCES[rule])
        case = {**deepcopy(row), 'curve': curve(), 'executions': [{}, {}], 'cohorts': [],
                'rejections': [], 'gate_rejections': [], 'score_diagnostics': [], 'score_rejections': []}
        case.pop('rejection_counts')
        cases[filename] = write(tmp_path, service.CACHE + '/' + filename, json.dumps(case))
        (baselines if rule == 'benchmark' else results).append(row)
    report = {
        'schema': 1, 'experiment': 'capacity_priority_20260910',
        'research_only': True, 'live_qualified': False, 'valid_strategy_evidence': False,
        'control_reproduction_passed': True, 'start': service.START, 'end': service.END,
        'signal_end': service.SIGNAL_END, 'code_sha256': codes, 'protocol_sha256': protocol,
        'parent_report_sha256': parent_hash, 'signal_manifest_sha256': signal_hash,
        'signals': signals, 'case_files_sha256': cases, 'signal_stats': stats,
        'results': results, 'baselines': baselines,
        'elapsed_seconds': .2, 'preparation_elapsed_seconds': .1, 'finmind_requests': 0,
        'charts': {rule: [{'date': point['date'], 'nav': point['nav']} for point in curve()]
                   for rule in service.RULES | {'benchmark'}},
        'limitations': ['Synthetic reader fixture, no performance claim.'],
        'cohort_comparisons': [{'rule': r, 'reference': service.REFERENCES[r], 'basis': b,
                               'scenario': c, 'delay': d, 'shared': 0, 'only_rule': [], 'only_reference': []}
                              for r, b, c, d in sorted(combos) if r != 'control3'],
    }
    save_report(tmp_path, report)
    return tmp_path, report


def test_complete_twenty_plus_four_report_is_available_without_research(sealed):
    _, report = sealed
    result = service.overview()
    assert result['available']
    assert result['results'] == report['results']
    assert len(result['results']) == 20 and len(result['baselines']) == 4
    assert len(result['case_files_sha256']) == 24
    assert not result['live_qualified'] and not result['valid_strategy_evidence']
    assert all(r['summary']['slots'] == (6 if r['rule'] == 'capacity6' else 3) for r in result['results'])


@pytest.mark.parametrize('change', [
    'missing_result', 'duplicate_result', 'missing_benchmark', 'duplicate_benchmark',
    'missing_case_hash', 'missing_code_hash', 'wrong_case_link', 'live', 'valid_edge',
    'not_research', 'failed_control', 'wrong_start', 'missing_chart', 'extra_chart',
    'empty_limitations', 'string_limitations', 'wrong_excess', 'wrong_contrast', 'wrong_reference',
    'boolean_delay', 'boolean_schema', 'nonzero_requests', 'boolean_requests', 'bad_duration',
])
def test_incomplete_or_misleading_report_is_fail_closed(sealed, change):
    root, report = sealed
    if change == 'missing_result': report['results'].pop()
    elif change == 'duplicate_result': report['results'][-1] = deepcopy(report['results'][0])
    elif change == 'missing_benchmark': report['baselines'].pop()
    elif change == 'duplicate_benchmark': report['baselines'][-1] = deepcopy(report['baselines'][0])
    elif change == 'missing_case_hash': report['case_files_sha256'].pop(next(iter(report['case_files_sha256'])))
    elif change == 'missing_code_hash': report['code_sha256'].pop(next(iter(report['code_sha256'])))
    elif change == 'wrong_case_link': report['results'][0]['case_file'] = report['results'][1]['case_file']
    elif change == 'live': report['live_qualified'] = True
    elif change == 'valid_edge': report['valid_strategy_evidence'] = True
    elif change == 'not_research': report['research_only'] = False
    elif change == 'failed_control': report['control_reproduction_passed'] = False
    elif change == 'wrong_start': report['start'] = '2021-01-01'
    elif change == 'missing_chart': report['charts'].pop('benchmark')
    elif change == 'extra_chart': report['charts']['extra'] = curve()
    elif change == 'empty_limitations': report['limitations'] = []
    elif change == 'string_limitations': report['limitations'] = 'not a list'
    elif change == 'wrong_excess': report['results'][0]['excess_vs_0050'] = .5
    elif change == 'wrong_contrast': report['results'][0]['contrast_vs_reference'] = .5
    elif change == 'wrong_reference': report['results'][0]['reference_rule'] = 'residual3'
    elif change == 'boolean_delay': report['results'][0]['delay'] = False
    elif change == 'boolean_schema': report['schema'] = True
    elif change == 'nonzero_requests': report['finmind_requests'] = 1
    elif change == 'boolean_requests': report['finmind_requests'] = False
    else: report['elapsed_seconds'] = -1
    save_report(root, report)
    result = service.overview()
    assert not result['available'] and not result['live_qualified'] and 'results' not in result


@pytest.mark.parametrize('name', [
    'skills/residual_priority.py', 'scripts/research_capacity.py', 'docs/prereg_capacity_20260910.md',
    service.CACHE + '/signals.json', service.CACHE + '/signals-official.json',
    service.CACHE + '/signals-snapshot.json', service.PARENT_REPORT,
    service.CACHE + '/case-residual3-official-stress-0.json',
])
def test_provenance_changes_are_detected_after_digest_cache_warmup(sealed, name):
    root, _ = sealed
    assert service.overview()['available']
    path = root / name
    stat, content = path.stat(), path.read_bytes()
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert not service.overview()['available']


@pytest.mark.parametrize('change', ['runtime', 'extra_runtime', 'prefix', 'parent_hash', 'parent_signal_hash',
                                   'signal_keys', 'requests', 'schema', 'preparation_duration'])
def test_resealed_signal_manifest_still_must_match_semantic_contract(sealed, change):
    root, report = sealed
    signals = report['signals']
    if change == 'runtime': signals['versions']['numpy'] = 'different-runtime'
    elif change == 'extra_runtime': signals['versions']['extra'] = 'runtime'
    elif change == 'prefix': signals['prefix_invariance_passed'] = False
    elif change == 'parent_hash': signals['parent_report_sha256'] = 'other-parent'
    elif change == 'parent_signal_hash': signals['parent_signal_manifest_sha256'] = 'other-signals'
    elif change == 'signal_keys': signals['files_sha256'].pop('signals-official.json')
    elif change == 'requests': signals['finmind_requests'] = 3
    elif change == 'schema': signals['schema'] = True
    else: signals['elapsed_seconds'] = 999.
    reseal_signals(root, report)
    assert not service.overview()['available']


def test_parent_became_unavailable_or_changed_after_read(sealed, monkeypatch):
    assert service.overview()['available']
    monkeypatch.setattr(service.regime_switch_research, 'overview', lambda: {'available': False})
    assert not service.overview()['available']
    monkeypatch.setattr(service.regime_switch_research, 'overview',
                        lambda: {'available': True, 'states': {'diffusion_signal_manifest_sha256': 'different'}})
    assert not service.overview()['available']


@pytest.mark.parametrize('change', ['missing_basis', 'missing_key', 'bool_count', 'negative_count', 'bad_order',
                                   'wrong_subset', 'same_order', 'bad_date', 'duplicate_date', 'missing_example',
                                   'wrong_rejection_total', 'list_rejections', 'report_only'])
def test_signal_diagnostics_are_typed_reconciled_and_bound_to_manifest(sealed, change):
    root, report = sealed
    stats = report['signal_stats']['official']
    if change == 'missing_basis': report['signal_stats'].pop('snapshot')
    elif change == 'missing_key': stats.pop('original_count')
    elif change == 'bool_count': stats['original_count'] = True
    elif change == 'negative_count': stats['trend_count'] = -1
    elif change == 'bad_order': stats['trend_count'] = 100
    elif change == 'wrong_subset': stats['reorder_examples'][0]['residual_order'] = ['event-C', 'event-A']
    elif change == 'same_order': stats['reorder_examples'][0]['residual_order'] = ['event-A', 'event-B']
    elif change == 'bad_date': stats['reorder_examples'][0]['date'] = 'tomorrow'
    elif change == 'duplicate_date': stats['reorder_examples'] *= 2
    elif change == 'missing_example': stats['reorder_examples'] = []
    elif change == 'wrong_rejection_total': stats['score_rejections']['missing_returns'] = 2
    elif change == 'list_rejections': stats['score_rejections'] = []
    else: stats['original_count'] = 5
    if change != 'report_only':
        report['signals']['stats'] = deepcopy(report['signal_stats'])
    reseal_signals(root, report)
    assert not service.overview()['available']


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'empty', 'wrong_reference', 'false_shared',
                                   'false_only', 'bad_only_type', 'boolean_shared'])
def test_fifteen_cohort_comparisons_are_complete_and_checked_against_account_ids(sealed, change):
    root, report = sealed
    comparisons = report['cohort_comparisons']
    if change == 'missing': comparisons.pop()
    elif change == 'duplicate': comparisons[-1] = deepcopy(comparisons[0])
    elif change == 'empty': report['cohort_comparisons'] = []
    elif change == 'wrong_reference': comparisons[0]['reference'] = 'residual3'
    elif change == 'false_shared': comparisons[0]['shared'] = 1
    elif change == 'false_only': comparisons[0]['only_rule'] = ['event-invented']
    elif change == 'bad_only_type': comparisons[0]['only_reference'] = {}
    else: comparisons[0]['shared'] = False
    save_report(root, report)
    assert not service.overview()['available']


@pytest.mark.parametrize('change', ['daily_nav', 'cumulative_cost', 'active_value', 'negative_cash', 'active_weight',
                                   'market_pnl', 'missing_year', 'final_nav', 'mdd', 'annual_distribution'])
def test_resealed_case_must_still_reconcile_daily_accounts_and_annual_summary(sealed, change):
    root, report = sealed
    row = next(r for r in report['results'] if r['scenario'] == 'base')
    case = read_case(root, row)
    if change == 'daily_nav': case['curve'][1]['nav'] = 1.1
    elif change == 'cumulative_cost': case['curve'][1]['cumulative_cost'] = .5
    elif change == 'active_value': case['curve'][1]['active_value'] = .1
    elif change == 'negative_cash': case['curve'][1]['cash'] = -.1
    elif change == 'active_weight': case['curve'][1]['active_weight'] = .5
    elif change == 'market_pnl': case['curve'][1]['market_pnl'] = .1
    elif change == 'missing_year': case['curve'].pop(2)
    elif change == 'final_nav': case['curve'][-1]['nav'] = .8
    elif change == 'mdd': row['summary']['max_drawdown'] = -.5
    else: row['summary']['annual_returns'].update({'2022': 0., '2023': -.01})
    case['summary'] = deepcopy(row['summary'])
    reseal_case(root, report, row, case)
    save_report(root, report)
    assert not service.overview()['available']


def test_marked_holdings_and_distinct_audit_dates_are_preserved_and_reconciled(sealed):
    root, report = sealed
    row = report['results'][0]
    row['summary'].update(final_cash=.49, final_liquidation_complete=False, final_nav_is_marked=True,
                          unliquidated_position_count=1, mean_active_weight=(.5 / .99) / 6,
                          mean_cash_weight=(5 + .49 / .99) / 6,
                          unliquidated_positions=[{'stock_id': '1101', 'units': .005, 'mark': 100.,
                                                   'marked_value': .5, 'mark_date': '2026-06-22'}])
    row['valuation_audit'] = {'finding_count': 2, 'unresolved_valuation_days': 1,
                             'findings': [{'stock_id': '1101', 'date': '2023-01-03'},
                                          {'stock_id': '1102', 'date': '2023-01-03'}]}
    case = read_case(root, row)
    case['curve'][-1].update(cash=.49, active_value=.5, active_weight=.5 / .99)
    case.update(summary=deepcopy(row['summary']), valuation_audit=deepcopy(row['valuation_audit']))
    reseal_case(root, report, row, case)
    save_report(root, report)
    result = service.overview()
    assert result['available'] and result['results'][0]['summary']['final_nav_is_marked']
    row['valuation_audit']['unresolved_valuation_days'] = 2
    reseal_case_header(root, report, row)
    save_report(root, report)
    assert not service.overview()['available']


@pytest.mark.parametrize('change', ['missing_diagnostics', 'list_shape', 'execution_count', 'cohort_count',
                                   'rejection_count', 'rejection_breakdown', 'benchmark_diagnostics'])
def test_case_record_shapes_and_counts_are_checked(sealed, change):
    root, report = sealed
    row = report['baselines'][0] if change == 'benchmark_diagnostics' else report['results'][0]
    case = read_case(root, row)
    if change == 'missing_diagnostics': case.pop('score_rejections')
    elif change == 'list_shape': case['score_diagnostics'] = {}
    elif change == 'execution_count': case['executions'].pop()
    elif change == 'cohort_count': case['cohorts'] = [{'event_id': 'invented'}]
    elif change == 'rejection_count': case['rejections'] = [{'reason': 'slots_full'}]
    elif change == 'rejection_breakdown': row['rejection_counts'] = {'slots_full': 1}
    else: case['score_diagnostics'] = [{'event_id': 'impossible'}]
    reseal_case(root, report, row, case)
    save_report(root, report)
    assert not service.overview()['available']


@pytest.mark.parametrize('field,value', [
    ('final_nav', -1.), ('final_cash', -.01), ('total_return', .5), ('cagr', 123.),
    ('max_drawdown', .01), ('total_cost', -.01), ('turnover', -.01),
    ('mean_cash_weight', 1.1), ('mean_active_weight', 1.1), ('initial_nav', True),
    ('slippage_per_side', .123), ('commission_per_side', 0.), ('stock_sell_tax', 0.),
    ('benchmark_sell_tax', .003), ('horizon', 30), ('slots', 10), ('mode', 'benchmark'),
    ('final_liquidation_complete', False), ('final_nav_is_marked', True),
])
def test_fees_and_accounting_numbers_are_cross_checked(sealed, field, value):
    root, report = sealed
    report['results'][0]['summary'][field] = value
    reseal_case_header(root, report, report['results'][0])
    save_report(root, report)
    assert not service.overview()['available']


@pytest.mark.parametrize('change', ['missing_year', 'wrong_product', 'boolean', 'list', 'not_finite'])
def test_annual_results_require_finite_complete_compounding(sealed, change):
    root, report = sealed
    annual = report['results'][0]['summary']['annual_returns']
    if change == 'missing_year': annual.pop('2025')
    elif change == 'wrong_product': annual['2025'] = .5
    elif change == 'boolean': annual['2025'] = True
    elif change == 'list': report['results'][0]['summary']['annual_returns'] = list(annual)
    else: annual['2025'] = float('nan')
    reseal_case_header(root, report, report['results'][0])
    save_report(root, report)
    assert not service.overview()['available']


def test_load_case_accepts_verified_identity_and_keeps_original_detail(sealed):
    _, _ = sealed
    report = service.overview()
    for row in report['results'] + report['baselines']:
        result = service.load_case(report, row)
        assert result['available']
        assert result['summary'] == row['summary']
        assert result['valuation_audit'] == row['valuation_audit']
        assert result['curve']


@pytest.mark.parametrize('name', ['../outside.json', '/tmp/outside.json', 'folder/case.json'])
def test_load_case_rejects_nonbasenames_even_in_a_supplied_report(sealed, name):
    _, _ = sealed
    report = service.overview()
    row = report['results'][0]
    row['case_file'] = name
    assert not service.load_case(report, row)['available']


def test_load_case_requires_membership_available_report_and_unchanged_bytes(sealed):
    root, _ = sealed
    report = service.overview()
    row = report['results'][0]
    assert not service.load_case({**report, 'available': False}, row)['available']
    assert not service.load_case(report, {**row, 'rule': 'unlisted'})['available']
    path = root / service.CACHE / row['case_file']
    path.write_text(path.read_text() + ' ')  # JSON equivalent but different sealed artifact.
    assert not service.load_case(report, row)['available']


@pytest.mark.parametrize('change', ['summary', 'audit', 'rule', 'basis', 'scenario', 'delay'])
def test_load_case_rejects_rehashed_details_that_disagree_with_displayed_case(sealed, change):
    root, _ = sealed
    report = service.overview()
    row = report['results'][0]
    path = root / service.CACHE / row['case_file']
    case = json.loads(path.read_text())
    if change == 'summary': case['summary']['total_return'] = .5
    elif change == 'audit': case['valuation_audit']['finding_count'] = 100
    elif change == 'delay': case['delay'] = 7
    else: case[change] = 'different'
    report['case_files_sha256'][row['case_file']] = write(root, service.CACHE + '/' + row['case_file'], json.dumps(case))
    assert not service.load_case(report, row)['available']


def test_cache_is_bounded_and_warm_overview_only_reuses_valid_file_digests(sealed):
    root, _ = sealed
    assert service.overview()['available']
    before = service._digest.cache_info()
    assert service.overview()['available']
    after = service._digest.cache_info()
    assert after.misses == before.misses and after.hits > before.hits
    for i in range(140):
        path = root / f'cache-entry-{i}'
        path.write_text(str(i))
        service.sha(path)
    assert service._digest.cache_info().currsize == 128


def test_same_size_mtime_different_paths_and_changed_sizes_do_not_share_hashes(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.write_text('a'); second.write_text('b')
    stat = first.stat()
    os.utime(second, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert service.sha(first) != service.sha(second)
    before = service.sha(first)
    first.write_text('longer')
    os.utime(first, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert service.sha(first) != before


def test_file_mutation_during_hash_read_is_not_accepted(tmp_path, monkeypatch):
    path = tmp_path / 'changing'
    path.write_text('original')

    def mutate(filename, size, modified):
        Path(filename).write_text('changed while hash read')
        return 'a' * 64

    monkeypatch.setattr(service, '_digest', mutate)
    with pytest.raises(ValueError, match='changed while reading'):
        service.sha(path)


def test_missing_malformed_or_nonfinite_reports_are_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'ROOT', tmp_path)
    assert not service.overview()['available']
    for payload in ('{', 'null', '[]', '{"schema": NaN}', '{"schema": Infinity}'):
        write(tmp_path, service.CACHE + '/report.summary.json', payload)
        assert not service.overview()['available']


@pytest.mark.parametrize('field,value', [('max_drawdown', -.02), ('total_cost', .02),
                                       ('turnover', 2.5), ('mean_cash_weight', .5)])
def test_legal_report_only_numbers_cannot_disagree_with_sealed_case_header(sealed, field, value):
    root, report = sealed
    assert service.overview()['available']  # Warm the small-header cache first.
    # Each new value is individually legal; the sealed component account did
    # not change. The overview itself must reject it, before opening details.
    report['results'][0]['summary'][field] = value
    save_report(root, report)
    assert not service.overview()['available']


@pytest.mark.parametrize('change', ['nav', 'date', 'missing_point', 'reordered'])
def test_chart_must_match_the_sealed_default_case_curve(sealed, change):
    root, report = sealed
    assert service.overview()['available']
    chart = report['charts']['residual3']
    if change == 'nav': chart[0]['nav'] = .98
    elif change == 'date': chart[0]['date'] = '2022-01-04'
    elif change == 'missing_point': chart.pop()
    else: chart.reverse()
    save_report(root, report)
    assert not service.overview()['available']


def test_small_case_headers_are_cached_without_retaining_full_transaction_records(sealed):
    root, original = sealed
    assert service.overview()['available']
    cold = service._case_header.cache_info()
    assert cold.misses == 24 and cold.currsize == 24 and cold.maxsize == 128
    assert service.overview()['available']
    warm = service._case_header.cache_info()
    assert warm.misses == cold.misses and warm.hits == cold.hits + 24
    row = original['results'][0]
    header, chart_digest = service._case_header(
        str(root / service.CACHE / row['case_file']), original['case_files_sha256'][row['case_file']])
    assert set(header) == {'summary', 'valuation_audit', 'rule', 'basis', 'scenario', 'delay', 'rejection_counts', 'cohort_ids'}
    assert len(chart_digest) == 64
    # Detailed records are still loaded from their original sealed case on
    # demand; they are not stored in the overview's reusable header cache.
    detail = service.load_case(service.overview(), row)
    assert detail['available'] and 'curve' in detail and 'executions' in detail


def test_wrong_case_fingerprint_is_rejected_during_cold_header_read(sealed):
    root, report = sealed
    row = report['results'][0]
    with pytest.raises(ValueError, match='Case changed during header verification'):
        service._case_header(str(root / service.CACHE / row['case_file']), '0' * 64)


def test_last_signal_may_have_a_next_year_entry_ordering_example(sealed):
    root, report = sealed
    for stats in report['signal_stats'].values():
        stats['reorder_examples'][0]['date'] = '2026-01-02'
    report['signals']['stats'] = deepcopy(report['signal_stats'])
    reseal_signals(root, report)
    assert service.overview()['available']


def test_cohort_comparisons_use_actual_sealed_intersections_and_differences(sealed):
    root, report = sealed
    event_ids = {'control3': ['shared', 'control-only'], 'capacity6': ['shared', 'capacity-only'],
                 'matched3': ['shared'], 'residual3': ['shared', 'residual-only']}
    for row in report['results']:
        ids = event_ids[row['rule']]
        row['summary'].update(entered_cohorts=len(ids), completed_cohorts=len(ids), peak_active_cohorts=len(ids))
        case = read_case(root, row)
        case['cohorts'] = [{'event_id': event} for event in ids]
        case['summary'] = deepcopy(row['summary'])
        reseal_case(root, report, row, case)
    for comparison in report['cohort_comparisons']:
        ours, theirs = set(event_ids[comparison['rule']]), set(event_ids[comparison['reference']])
        comparison.update(shared=len(ours & theirs), only_rule=sorted(ours - theirs), only_reference=sorted(theirs - ours))
    save_report(root, report)
    assert service.overview()['available']
    report['cohort_comparisons'][0]['only_reference'] = ['invented']
    save_report(root, report)
    assert not service.overview()['available']


def test_residual_contrast_uses_matched_subset_even_when_control_return_differs(sealed):
    root, report = sealed
    navs = {'control3': 1.1, 'capacity6': 1.2, 'matched3': .9, 'residual3': 1.05}
    years = (date.fromisoformat(service.END) - date.fromisoformat(service.START)).days / 365.25
    for row in report['results']:
        final_nav = navs[row['rule']]
        row['summary'].update(final_nav=final_nav, final_cash=final_nav, total_return=final_nav - 1,
                              cagr=final_nav ** (1 / years) - 1, max_drawdown=min(-.01, final_nav - 1))
        row['summary']['annual_returns']['2022'] = final_nav - 1
        row['excess_vs_0050'] = final_nav - .99
        row['contrast_vs_reference'] = final_nav - navs[row['reference_rule']]
        case = read_case(root, row)
        case['summary'] = deepcopy(row['summary'])
        for i, point in enumerate(case['curve'][1:], 1):
            point.update(nav=final_nav, cash=final_nav, market_pnl=final_nav - .99 if i == 1 else 0.)
        reseal_case(root, report, row, case)
        if (row['basis'], row['scenario'], row['delay']) == ('official', 'stress', 0):
            report['charts'][row['rule']] = [{'date': p['date'], 'nav': p['nav']} for p in case['curve']]
    save_report(root, report)
    assert service.overview()['available']
    residual = next(r for r in report['results'] if r['rule'] == 'residual3')
    assert residual['contrast_vs_reference'] == pytest.approx(.15)
    residual['contrast_vs_reference'] = navs['residual3'] - navs['control3']
    save_report(root, report)
    assert not service.overview()['available']
