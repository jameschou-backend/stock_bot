"""Small sealed-reader fixtures: no history, DB, API, or research jobs."""
from copy import deepcopy
from datetime import date
import hashlib
import json
import os
from pathlib import Path

import pytest

from app import regime_switch_research as service


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def save_report(root, report):
    write(root, service.CACHE + '/report.summary.json', json.dumps(report))


def reseal_case_header(root, report, row):
    """Keep synthetic case/header aligned when testing independent validation."""
    filename = row['case_file']
    case = json.loads((root / service.CACHE / filename).read_text())
    for key in ('summary', 'valuation_audit', 'rule', 'basis', 'scenario', 'delay'):
        case[key] = deepcopy(row[key])
    report['case_files_sha256'][filename] = write(root, service.CACHE + '/' + filename, json.dumps(case))


def summary(rule, scenario):
    mode = {'always': 'events', 'entry_only': 'events', 'idle_cash': 'idle_cash',
            'exit_cash': 'exit_cash', 'mix_always': 'fixed_mix',
            'mix_entry': 'fixed_mix', 'benchmark': 'benchmark'}[rule]
    years = (date(2026, 6, 23) - date(2022, 1, 3)).days / 365.25
    return {
        'mode': mode, 'start': '2022-01-03', 'end': '2026-06-23',
        'requested_start': '2022-01-03', 'requested_end': '2026-06-23',
        'initial_nav': 1., 'final_nav': .99, 'final_cash': .99,
        'total_return': -.01, 'cagr': .99 ** (1 / years) - 1,
        'max_drawdown': -.01, 'total_cost': .01, 'turnover': 1.99,
        'mean_active_weight': 0., 'mean_cash_weight': 1.,
        'slots': 3, 'horizon': 63, 'trade_count': 2,
        'completed_cohorts': 0, 'entered_cohorts': 0, 'peak_active_cohorts': 0,
        'blocked_exit_sessions': 0, 'rejected_event_count': 0,
        'unliquidated_position_count': 0, 'unliquidated_positions': [],
        'final_liquidation_complete': True, 'final_nav_is_marked': False,
        'annual_returns': {'2022': -.01, '2023': 0., '2024': 0., '2025': 0., '2026': 0.},
        'slippage_per_side': {'base': .003, 'stress': .0045}[scenario],
        'commission_per_side': .001425, 'stock_sell_tax': .003, 'benchmark_sell_tax': .001,
    }


@pytest.fixture
def sealed(monkeypatch, tmp_path):
    monkeypatch.setattr(service, 'ROOT', tmp_path)
    monkeypatch.setattr(service, 'version', lambda name: 'fixture-runtime')
    monkeypatch.setattr(service.diffusion_research, 'overview',
                        lambda: {'available': True, 'signal_manifest_sha256': 'sealed-parent'})
    service._digest.cache_clear()
    service._case_header.cache_clear()
    codes = {name: write(tmp_path, name, '# sealed ' + name) for name in service.CODE}
    protocol = write(tmp_path, 'docs/prereg_regime_switch_20260910.md', 'Fixed synthetic protocol')
    states = {
        'schema': 1, 'code_sha256': codes, 'protocol_sha256': protocol,
        'prefix_invariance_passed': True, 'diffusion_signal_manifest_sha256': 'sealed-parent',
        'versions': {name: 'fixture-runtime' for name in ('numpy', 'pandas', 'scipy')},
        'files_sha256': {name: write(tmp_path, service.CACHE + '/' + name, 'state bytes ' + name)
                          for name in ('states-official.parquet', 'states-snapshot.parquet')},
        'prior_probe_files_sha256': {name: write(tmp_path, name, 'prior probe bytes ' + name)
                                      for name in service.PROBE},
    }
    state_hash = write(tmp_path, service.CACHE + '/states.json', json.dumps(states))
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
               'valuation_audit': {'finding_count': 0, 'unresolved_valuation_days': 0, 'findings': []}}
        if rule != 'benchmark':
            row.update(excess_vs_0050=0., excess_vs_entry_only=0., excess_vs_mix_entry=0.)
        case = {**row, 'curve': [{'date': '2022-01-03', 'nav': .99, 'cash': .99,
                                 'daily_cost': .01, 'cumulative_cost': .01, 'market_pnl': 0.},
                                {'date': '2026-06-23', 'nav': .99, 'cash': .99,
                                 'daily_cost': 0., 'cumulative_cost': .01, 'market_pnl': 0.}],
                'executions': [], 'cohorts': [], 'rejections': [], 'gate_rejections': []}
        cases[filename] = write(tmp_path, service.CACHE + '/' + filename, json.dumps(case))
        (baselines if rule == 'benchmark' else results).append(row)
    report = {
        'schema': 1, 'experiment': 'regime_switch_20260910',
        'research_only': True, 'live_qualified': False, 'valid_strategy_evidence': False,
        'control_reproduction_passed': True, 'start': '2022-01-03', 'end': '2026-06-23',
        'signal_end': '2025-12-31', 'code_sha256': codes, 'protocol_sha256': protocol,
        'state_manifest_sha256': state_hash, 'states': states, 'case_files_sha256': cases,
        'results': results, 'baselines': baselines,
        'charts': {rule: [{'date': '2022-01-03', 'nav': .99}, {'date': '2026-06-23', 'nav': .99}]
                   for rule in service.RULES | {'benchmark'}},
        'limitations': ['Synthetic reader fixture, no performance claim.'],
        'leave_one_year_diagnostic': [{'omitted_year': year, 'diagnostic_only': True,
                                      'compounded_remaining_years': {rule: 0. if year == '2022' else -.01
                                                                    for rule in service.RULES | {'benchmark'}}}
                                     for year in ('2022', '2023', '2024', '2025')],
    }
    save_report(tmp_path, report)
    return tmp_path, report


def test_complete_thirty_plus_four_report_is_available_without_research(sealed):
    _, report = sealed
    result = service.overview()
    assert result['available']
    assert result['results'] == report['results']
    assert len(result['results']) == 30 and len(result['baselines']) == 4
    assert len(result['case_files_sha256']) == 34
    assert not result['live_qualified'] and not result['valid_strategy_evidence']


@pytest.mark.parametrize('change', [
    'missing_result', 'duplicate_result', 'missing_benchmark', 'duplicate_benchmark',
    'missing_case_hash', 'missing_code_hash', 'wrong_case_link', 'live', 'valid_edge',
    'not_research', 'failed_control', 'wrong_start', 'missing_chart', 'empty_limitations',
    'nondiagnostic_leave_year', 'wrong_excess',
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
    elif change == 'empty_limitations': report['limitations'] = []
    elif change == 'nondiagnostic_leave_year': report['leave_one_year_diagnostic'][0]['diagnostic_only'] = False
    else: report['results'][0]['excess_vs_mix_entry'] = .5
    save_report(root, report)
    result = service.overview()
    assert not result['available'] and not result['live_qualified']
    assert 'results' not in result


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


def test_marked_holdings_and_daily_audit_counts_are_preserved_and_reconciled(sealed):
    root, report = sealed
    row = report['results'][0]
    row['summary'].update(final_cash=.49, final_liquidation_complete=False, final_nav_is_marked=True,
                          unliquidated_position_count=1,
                          unliquidated_positions=[{'stock_id': '1101', 'units': .005, 'mark': 100.,
                                                   'marked_value': .5, 'mark_date': '2026-06-22'}])
    row['valuation_audit'] = {'finding_count': 2, 'unresolved_valuation_days': 1,
                             'findings': [{'stock_id': '1101', 'date': '2023-01-03'},
                                          {'stock_id': '1102', 'date': '2023-01-03'}]}
    reseal_case_header(root, report, row)
    save_report(root, report)
    result = service.overview()
    assert result['available']
    assert result['results'][0]['summary']['final_nav_is_marked']
    row['summary']['unliquidated_positions'][0]['marked_value'] = .6
    reseal_case_header(root, report, row)
    save_report(root, report)
    assert not service.overview()['available']
    row['summary']['unliquidated_positions'][0]['marked_value'] = .5
    row['valuation_audit']['unresolved_valuation_days'] = 2
    reseal_case_header(root, report, row)
    save_report(root, report)
    assert not service.overview()['available']


@pytest.mark.parametrize('name', [
    'skills/regime_state.py', 'skills/regime_mix.py', 'scripts/research_regime_switch.py',
    'docs/prereg_regime_switch_20260910.md', service.CACHE + '/states.json',
    service.CACHE + '/states-official.parquet', service.CACHE + '/states-snapshot.parquet',
    '.cache/regime-probe-20260910/protocol.md', '.cache/regime-probe-20260910/probe.py',
    '.cache/regime-probe-20260910/report.json',
    service.CACHE + '/case-exit_cash-official-stress-0.json',
])
def test_provenance_changes_are_detected_after_digest_cache_warmup(sealed, name):
    root, _ = sealed
    assert service.overview()['available']
    path = root / name
    stat, content = path.stat(), path.read_bytes()
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])  # Same length, new mtime.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert not service.overview()['available']


@pytest.mark.parametrize('change', ['runtime', 'prefix', 'parent_hash', 'probe_keys'])
def test_resealed_state_manifest_still_must_match_its_semantic_contract(sealed, change):
    root, report = sealed
    states = report['states']
    if change == 'runtime': states['versions']['numpy'] = 'different-runtime'
    elif change == 'prefix': states['prefix_invariance_passed'] = False
    elif change == 'parent_hash': states['diffusion_signal_manifest_sha256'] = 'other-parent'
    else: states['prior_probe_files_sha256'].pop(next(iter(states['prior_probe_files_sha256'])))
    report['state_manifest_sha256'] = write(root, service.CACHE + '/states.json', json.dumps(states))
    save_report(root, report)
    assert not service.overview()['available']


def test_parent_became_unavailable_or_changed_after_read(sealed, monkeypatch):
    assert service.overview()['available']
    monkeypatch.setattr(service.diffusion_research, 'overview', lambda: {'available': False})
    assert not service.overview()['available']
    monkeypatch.setattr(service.diffusion_research, 'overview',
                        lambda: {'available': True, 'signal_manifest_sha256': 'new-parent'})
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
    chart = report['charts']['exit_cash']
    if change == 'nav': chart[0]['nav'] = .98
    elif change == 'date': chart[0]['date'] = '2022-01-04'
    elif change == 'missing_point': chart.pop()
    else: chart.reverse()
    save_report(root, report)
    assert not service.overview()['available']


@pytest.mark.parametrize('change', ['missing_year', 'duplicate_year', 'unexpected_year',
                                   'wrong_compounding', 'missing_policy', 'extra_policy'])
def test_leave_one_year_diagnostic_has_exact_years_and_recomputed_products(sealed, change):
    root, report = sealed
    diagnostic = report['leave_one_year_diagnostic']
    if change == 'missing_year': diagnostic.pop()
    elif change == 'duplicate_year': diagnostic[-1] = deepcopy(diagnostic[0])
    elif change == 'unexpected_year': diagnostic[-1]['omitted_year'] = '2026'
    elif change == 'wrong_compounding': diagnostic[0]['compounded_remaining_years']['always'] = .5
    elif change == 'missing_policy': diagnostic[0]['compounded_remaining_years'].pop('benchmark')
    else: diagnostic[0]['compounded_remaining_years']['unexpected'] = 0.
    save_report(root, report)
    assert not service.overview()['available']


def test_small_case_headers_are_cached_without_retaining_full_transaction_records(sealed):
    root, original = sealed
    assert service.overview()['available']
    cold = service._case_header.cache_info()
    assert cold.misses == 34 and cold.currsize == 34 and cold.maxsize == 128
    assert service.overview()['available']
    warm = service._case_header.cache_info()
    assert warm.misses == cold.misses and warm.hits == cold.hits + 34
    row = original['results'][0]
    header, chart_digest = service._case_header(
        str(root / service.CACHE / row['case_file']), original['case_files_sha256'][row['case_file']])
    assert set(header) == {'summary', 'valuation_audit', 'rule', 'basis', 'scenario', 'delay'}
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
