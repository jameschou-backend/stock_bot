"""Small sealed fixtures; no history, database, network, or portfolio jobs."""
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from app import diffusion_research as service


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def summary(mode, scenario):
    return {'mode': mode, 'start': service.START, 'end': service.END, 'slots': 3, 'horizon': 63,
            'total_return': -.01, 'cagr': -.002, 'max_drawdown': -.01, 'total_cost': .01,
            'turnover': 1.99, 'mean_active_weight': 0., 'initial_nav': 1.,
            'final_nav': .99, 'final_cash': .99, 'trade_count': 2,
            'completed_cohorts': 0, 'entered_cohorts': 0, 'peak_active_cohorts': 0,
            'blocked_exit_sessions': 0, 'rejected_event_count': 0,
            'unliquidated_position_count': 0, 'unliquidated_positions': [],
            'final_liquidation_complete': True, 'final_nav_is_marked': False,
            'annual_returns': {'2022': -.01, '2023': 0., '2024': 0., '2025': 0., '2026': 0.},
            'slippage_per_side': {'base': .003, 'stress': .0045}[scenario],
            'commission_per_side': .001425, 'stock_sell_tax': .003, 'benchmark_sell_tax': .001}


@pytest.fixture
def sealed(monkeypatch, tmp_path):
    monkeypatch.setattr(service, 'ROOT', tmp_path)
    service._sha_cached.cache_clear()
    source = 'def prepare_inputs():\n    return "fixture"\n\ndef save_matrix(path, frame):\n    return None\n'
    hashes = {name: write(tmp_path, name, source if name.endswith('research_diffusion.py') else '# fixture\n')
              for name in service.CODE}
    spec_hash = write(tmp_path, 'docs/prereg_diffusion_20260910.md', 'fixed protocol')
    inputs = {'schema': 1,
              'files_sha256': {name: write(tmp_path, service.CACHE + '/' + name, name + ' frozen input')
                               for name in service.INPUT_FILES},
              'parent_files_sha256': {name: write(tmp_path, name, name + ' parent input')
                                      for name in service.PARENT_FILES},
              'parent_price_manifest_sha256': write(tmp_path, '.cache/event-group-research/signal-inputs.json', '{}'),
              'input_transform_sha256': service.input_transform_hash(tmp_path / 'scripts/research_diffusion.py'),
              'transform_versions': service.runtime_versions(('numpy', 'pandas', 'duckdb'))}
    input_hash = write(tmp_path, service.CACHE + '/inputs.json', json.dumps(inputs))
    signals = {'schema': 1, 'code_sha256': hashes, 'preregistration_sha256': spec_hash,
               'input_manifest_sha256': input_hash,
               'versions': service.runtime_versions(('numpy', 'pandas', 'scipy')),
               'files_sha256': {name: write(tmp_path, service.CACHE + '/' + name, '{}')
                                for name in service.SIGNAL_FILES}}
    signal_hash = write(tmp_path, service.CACHE + '/signals.json', json.dumps(signals))
    combos = {(r, b, b, c, 0) for r in service.RULES for b in service.BASES for c in service.SCENARIOS}
    combos |= {(r, 'official', 'official', 'stress', 1) for r in service.RULES}
    combos |= {(r, 'snapshot', 'official', 'stress', 0) for r in service.RULES}
    results = [{'rule': r, 'basis': b, 'signal_basis': sb, 'scenario': c, 'delay': d,
                'name': r, 'summary': summary('events', c), 'cohorts': [], 'rejections': [],
                'signal_count': 0, 'excess_vs_0050': 0., 'excess_vs_basket': None if r == 'leader_now' else 0.,
                'basket_comparison_paired': r != 'leader_now',
                'valuation_audit': {'unresolved_valuation_days': 0, 'finding_count': 0, 'findings': []}}
               for r, b, sb, c, d in sorted(combos)]
    report = {'schema': 1, 'experiment': 'diffusion_20260910', 'research_only': True,
              'live_qualified': False, 'valid_strategy_evidence': False,
              'start': service.START, 'end': service.END, 'signal_end': '2025-12-31',
              'code_sha256': hashes, 'preregistration_sha256': spec_hash,
              'input_manifest_sha256': input_hash, 'signal_manifest_sha256': signal_hash,
              'inputs': inputs, 'signal_info': signals, 'results': results,
              'baselines': [{'basis': b, 'scenario': c, 'summary': summary('benchmark', c),
                             'cohorts': [], 'rejections': []}
                            for b in sorted(service.BASES) for c in sorted(service.SCENARIOS)],
              'events': {'official': [], 'snapshot': []}, 'groups': {'official': [], 'snapshot': []},
              'limitations': ['Synthetic fixture, no performance claim.']}
    write(tmp_path, service.CACHE + '/report.summary.json', json.dumps(report))
    return tmp_path, report


def test_accepts_complete_sealed_report_without_starting_research(sealed):
    _, report = sealed
    result = service.overview()
    assert result['available']
    assert result['results'] == report['results']
    assert len(result['results']) == 24
    assert len(result['baselines']) == 4
    assert not result['live_qualified'] and not result['valid_strategy_evidence']


@pytest.mark.parametrize('mutation', [
    lambda report: report['results'].pop(),
    lambda report: report['results'].__setitem__(-1, copy.deepcopy(report['results'][0])),
    lambda report: report['baselines'].pop(),
    lambda report: report['baselines'].__setitem__(-1, copy.deepcopy(report['baselines'][0])),
    lambda report: report.__setitem__('live_qualified', True),
    lambda report: report.__setitem__('valid_strategy_evidence', True),
    lambda report: report['results'][0]['summary'].__setitem__('final_liquidation_complete', False),
    lambda report: report['results'][0].__setitem__('excess_vs_0050', .05),
    lambda report: next(row for row in report['results'] if row['rule'] == 'leader_now').__setitem__('excess_vs_basket', .1),
    lambda report: report['results'][0]['summary']['annual_returns'].__setitem__('2023', .1),
    lambda report: report['results'][0]['valuation_audit'].__setitem__('unresolved_valuation_days', 1),
    lambda report: report['results'][0]['valuation_audit'].__setitem__('finding_count', 1),
    lambda report: report['results'][0].__setitem__('basket_comparison_paired', False),
    lambda report: report['results'][0]['summary'].__setitem__('slippage_per_side', 0.),
    lambda report: report['results'][0].__setitem__('delay', False),
    lambda report: report['events'].pop('snapshot'),
    lambda report: report['code_sha256'].pop('skills/diffusion_signals.py'),
])
def test_rejects_incomplete_duplicate_or_misleading_reports(sealed, mutation):
    root, report = sealed
    mutation(report)
    write(root, service.CACHE + '/report.summary.json', json.dumps(report))
    result = service.overview()
    assert not result['available']
    assert not result['live_qualified']
    assert 'results' not in result


@pytest.mark.parametrize('name', [
    'scripts/research_diffusion.py', 'skills/diffusion_signals.py',
    'docs/prereg_diffusion_20260910.md',
    service.CACHE + '/inputs.json', service.CACHE + '/signals.json',
    service.CACHE + '/signals-official.json', service.CACHE + '/signals-snapshot.json',
    service.CACHE + '/close-official.parquet', service.CACHE + '/volume.parquet',
    '.cache/growth-flow-research/quotes.parquet', '.cache/event-group-research/signal-inputs.json',
])
def test_changed_sources_raw_inputs_and_signals_invalidate_even_after_cache_warmup(sealed, name):
    root, _ = sealed
    assert service.overview()['available']
    path = root / name
    old = path.stat()
    content = path.read_bytes()
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
    assert not service.overview()['available']


def test_rejects_changed_runtime_and_transform_even_with_resealed_manifests(sealed):
    root, report = sealed
    report['inputs']['input_transform_sha256'] = '0' * 64
    report['input_manifest_sha256'] = write(root, service.CACHE + '/inputs.json', json.dumps(report['inputs']))
    report['signal_info']['input_manifest_sha256'] = report['input_manifest_sha256']
    report['signal_manifest_sha256'] = write(root, service.CACHE + '/signals.json', json.dumps(report['signal_info']))
    write(root, service.CACHE + '/report.summary.json', json.dumps(report))
    assert not service.overview()['available']


def test_runtime_version_change_is_explicitly_unavailable(sealed, monkeypatch):
    assert service.overview()['available']
    monkeypatch.setattr(service, 'runtime_versions', lambda names: {name: 'changed' for name in names})
    assert not service.overview()['available']


def test_cache_reuses_digests_and_is_bounded(sealed):
    root, _ = sealed
    assert service.overview()['available']
    before = service._sha_cached.cache_info()
    assert service.overview()['available']
    after = service._sha_cached.cache_info()
    assert after.misses == before.misses
    assert after.hits > before.hits
    for i in range(50):
        path = root / f'extra-{i}'
        path.write_text(str(i))
        service.sha(path)
    assert service._sha_cached.cache_info().currsize == 32


def test_path_and_size_changes_cannot_reuse_another_digest(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.write_text('a'); b.write_text('b')
    stat = a.stat()
    os.utime(b, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert service.sha(a) != service.sha(b)
    before = service.sha(a)
    a.write_text('longer')
    os.utime(a, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert service.sha(a) != before


def test_missing_malformed_and_nonfinite_json_are_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'ROOT', tmp_path)
    assert not service.overview()['available']
    for text in ('{', 'null', '{"schema": NaN}'):
        write(tmp_path, service.CACHE + '/report.summary.json', text)
        assert not service.overview()['available']


def test_explicit_unliquidated_mark_is_accepted_and_preserved(sealed):
    root, report = sealed
    row = report['results'][0]
    row['summary'].update(final_cash=.49, final_liquidation_complete=False, final_nav_is_marked=True,
                          unliquidated_position_count=1,
                          unliquidated_positions=[{'stock_id': '1101', 'marked_value': .5}])
    write(root, service.CACHE + '/report.summary.json', json.dumps(report))
    result = service.overview()
    assert result['available']
    assert not result['results'][0]['summary']['final_liquidation_complete']
    assert result['results'][0]['summary']['final_cash'] != result['results'][0]['summary']['final_nav']


def test_two_price_findings_on_one_day_are_not_two_unresolved_days(sealed):
    root, report = sealed
    report['results'][0]['valuation_audit'] = {
        'unresolved_valuation_days': 1, 'finding_count': 2,
        'findings': [{'date': '2023-01-03', 'stock_id': '1101'},
                     {'date': '2023-01-03', 'stock_id': '1102'}]}
    write(root, service.CACHE + '/report.summary.json', json.dumps(report))
    assert service.overview()['available']
    report['results'][0]['valuation_audit']['unresolved_valuation_days'] = 2
    write(root, service.CACHE + '/report.summary.json', json.dumps(report))
    assert not service.overview()['available']


def test_unconfirmed_leaders_cannot_claim_paired_basket_control(sealed):
    root, report = sealed
    row = next(row for row in report['results'] if row['rule'] == 'leader_now')
    row['basket_comparison_paired'] = True
    write(root, service.CACHE + '/report.summary.json', json.dumps(report))
    assert not service.overview()['available']
