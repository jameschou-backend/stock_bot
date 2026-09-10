"""Read a sealed diffusion report without network, database, or backtest work."""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = '.cache/diffusion-research'
CODE = {'scripts/research_diffusion.py', 'skills/diffusion_signals.py', 'skills/diffusion_portfolio.py'}
INPUT_FILES = {'close-official.parquet', 'close-snapshot.parquet', 'trade-flags.parquet',
               'volume.parquet', 'turnover.parquet', 'companies.parquet'}
PARENT_FILES = {'.cache/event-group-research/' + name for name in
                ('close-official.parquet', 'close-snapshot.parquet', 'trade-flags.parquet')} | {
                    '.cache/growth-flow-research/quotes.parquet', '.cache/growth-flow-research/companies.parquet'}
SIGNAL_FILES = {'signals-official.json', 'signals-snapshot.json'}
RULES = {'leader_now', 'leader_after', 'follower_after', 'basket_after'}
BASES = {'official', 'snapshot'}
SCENARIOS = {'base', 'stress'}
START, END = '2022-01-03', '2026-06-23'


@lru_cache(maxsize=32)
def _sha_cached(resolved_path, size, mtime_ns):
    digest = hashlib.sha256()
    with Path(resolved_path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha(path):
    """Cache at most 32 file digests and invalidate when path/size/mtime changes."""
    path = Path(path).resolve(strict=True)
    before = path.stat()
    result = _sha_cached(str(path), before.st_size, before.st_mtime_ns)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Research file changed while verifying it')
    return result


def runtime_versions(names):
    return {name: version(name) for name in names}


def input_transform_hash(path):
    """Match inspect.getsource without importing or executing the research CLI."""
    text = Path(path).read_text()
    lines = text.splitlines(keepends=True)
    functions = {node.name: ''.join(lines[node.lineno - 1:node.end_lineno])
                 for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    return hashlib.sha256((functions['prepare_inputs'] + functions['save_matrix']).encode()).hexdigest()


def _read(path):
    # Python's default decoder otherwise accepts NaN/Infinity in JSON.
    def invalid(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def _verify_hashes(mapping, expected, base):
    if not isinstance(mapping, dict) or set(mapping) != expected:
        raise ValueError('Incomplete provenance')
    for name, fingerprint in mapping.items():
        if not isinstance(fingerprint, str) or sha(base / name) != fingerprint:
            raise ValueError('Research file changed: ' + name)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _summary(row, mode):
    summary = row['summary']
    required = ('total_return', 'cagr', 'max_drawdown', 'total_cost', 'turnover',
                'mean_active_weight', 'initial_nav', 'final_nav', 'final_cash')
    if any(not _finite(summary[key]) for key in required):
        raise ValueError('Invalid portfolio numbers')
    if (summary['mode'] != mode or summary['start'] != START or summary['end'] != END
            or summary['slots'] != 3 or summary['horizon'] != 63
            or summary['initial_nav'] != 1 or summary['final_nav'] <= 0
            or summary['total_return'] <= -1 or summary['cagr'] <= -1
            or not -1 <= summary['max_drawdown'] <= 0
            or not 0 <= summary['mean_active_weight'] <= 1 + 1e-10
            or summary['total_cost'] < 0 or summary['turnover'] < 0 or summary['final_cash'] < 0
            or not math.isclose(summary['final_nav'], 1 + summary['total_return'], abs_tol=1e-12)):
        raise ValueError('Inconsistent portfolio summary')
    if (summary['slippage_per_side'] != {'base': .003, 'stress': .0045}[row['scenario']]
            or summary['commission_per_side'] != .001425
            or summary['stock_sell_tax'] != .003 or summary['benchmark_sell_tax'] != .001):
        raise ValueError('Changed cost assumptions')
    count_fields = ('trade_count', 'completed_cohorts', 'entered_cohorts', 'peak_active_cohorts',
                    'blocked_exit_sessions', 'rejected_event_count', 'unliquidated_position_count')
    if any(not _integer(summary[key]) for key in count_fields):
        raise ValueError('Invalid portfolio counts')
    cohorts, rejections = row['cohorts'], row['rejections']
    if not isinstance(cohorts, list) or not isinstance(rejections, list):
        raise ValueError('Missing portfolio ledger')
    if (len(cohorts) != summary['entered_cohorts']
            or len({item['event_id'] for item in cohorts}) != len(cohorts)
            or sum(item['status'] == 'completed' for item in cohorts) != summary['completed_cohorts']
            or any(item['status'] not in {'completed', 'open'} for item in cohorts)
            or len(rejections) != summary['rejected_event_count'] or summary['peak_active_cohorts'] > 3):
        raise ValueError('Incomplete portfolio ledger')
    positions = summary['unliquidated_positions']
    if (not isinstance(positions, list) or len(positions) != summary['unliquidated_position_count']
            or summary['final_liquidation_complete'] is not (not positions)
            or summary['final_nav_is_marked'] is not bool(positions)):
        raise ValueError('Missing final liquidation warning')
    if any(not _finite(item['marked_value']) or item['marked_value'] <= 0 for item in positions):
        raise ValueError('Invalid outstanding holding value')
    if not math.isclose(summary['final_cash'] + sum(item['marked_value'] for item in positions),
                        summary['final_nav'], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('Final cash and outstanding holdings do not reconcile')
    annual = summary['annual_returns']
    if (set(annual) != {'2022', '2023', '2024', '2025', '2026'}
            or any(not _finite(value) or value <= -1 for value in annual.values())
            or not math.isclose(math.prod(1 + value for value in annual.values()),
                                summary['final_nav'], rel_tol=1e-9, abs_tol=1e-12)):
        raise ValueError('Incomplete annual returns')


def overview():
    """Return the sealed summary or an explicit unavailable/rebuild message."""
    missing = {'available': False, 'research_only': True, 'live_qualified': False,
               'valid_strategy_evidence': False,
               'note': '尚無完整的族群擴散試驗；先執行 make prepare-diffusion，再執行 make research-diffusion。'}
    cache = ROOT / CACHE
    path = cache / 'report.summary.json'
    if not path.exists():
        return missing
    try:
        report = _read(path)
        if (report['schema'] != 1 or report['experiment'] != 'diffusion_20260910'
                or report['research_only'] is not True or report['live_qualified'] is not False
                or report['valid_strategy_evidence'] is not False
                or report['start'] != START or report['end'] != END or report['signal_end'] != '2025-12-31'):
            raise ValueError('Unsupported diffusion experiment')
        _verify_hashes(report['code_sha256'], CODE, ROOT)
        if sha(ROOT / 'docs/prereg_diffusion_20260910.md') != report['preregistration_sha256']:
            raise ValueError('Changed protocol')
        if sha(cache / 'inputs.json') != report['input_manifest_sha256']:
            raise ValueError('Changed input manifest')
        inputs = _read(cache / 'inputs.json')
        if inputs != report['inputs'] or inputs['schema'] != 1:
            raise ValueError('Incomplete input provenance')
        _verify_hashes(inputs['files_sha256'], INPUT_FILES, cache)
        _verify_hashes(inputs['parent_files_sha256'], PARENT_FILES, ROOT)
        if sha(ROOT / '.cache/event-group-research/signal-inputs.json') != inputs['parent_price_manifest_sha256']:
            raise ValueError('Changed parent price provenance')
        if (inputs['input_transform_sha256'] != input_transform_hash(ROOT / 'scripts/research_diffusion.py')
                or inputs['transform_versions'] != runtime_versions(('numpy', 'pandas', 'duckdb'))):
            raise ValueError('Input transformation or runtime changed')
        if sha(cache / 'signals.json') != report['signal_manifest_sha256']:
            raise ValueError('Changed signal manifest')
        signals = _read(cache / 'signals.json')
        if (signals != report['signal_info'] or signals['schema'] != 1
                or signals['code_sha256'] != report['code_sha256']
                or signals['preregistration_sha256'] != report['preregistration_sha256']
                or signals['input_manifest_sha256'] != report['input_manifest_sha256']
                or signals['versions'] != runtime_versions(('numpy', 'pandas', 'scipy'))):
            raise ValueError('Incomplete or changed signal provenance')
        _verify_hashes(signals['files_sha256'], SIGNAL_FILES, cache)

        expected = {(r, b, b, c, 0) for r in RULES for b in BASES for c in SCENARIOS}
        expected |= {(r, 'official', 'official', 'stress', 1) for r in RULES}
        expected |= {(r, 'snapshot', 'official', 'stress', 0) for r in RULES}
        results = report['results']
        key = lambda row: (row['rule'], row['basis'], row['signal_basis'], row['scenario'], row['delay'])
        if (len(results) != 24 or any(not _integer(row['delay']) for row in results)
                or {key(row) for row in results} != expected):
            raise ValueError('Incomplete or duplicate fixed contrasts')
        baselines = report['baselines']
        if len(baselines) != 4 or {(row['basis'], row['scenario']) for row in baselines} != {
                (b, c) for b in BASES for c in SCENARIOS}:
            raise ValueError('Incomplete benchmark controls')
        for row in baselines:
            _summary(row, 'benchmark')
        indexed = {key(row): row for row in results}
        benchmark = {(row['basis'], row['scenario']): row for row in baselines}
        for row in results:
            _summary(row, 'events')
            if not _integer(row['signal_count']):
                raise ValueError('Invalid signal count')
            audit = row['valuation_audit']
            if (not _integer(audit['unresolved_valuation_days']) or not _integer(audit['finding_count'])
                    or not isinstance(audit['findings'], list)
                    or audit['finding_count'] != len(audit['findings'])
                    or audit['unresolved_valuation_days'] != len({finding['date'] for finding in audit['findings']})):
                raise ValueError('Incomplete valuation audit')
            if row['basket_comparison_paired'] is not (row['rule'] != 'leader_now'):
                raise ValueError('Incorrect basket pairing claim')
            bm = benchmark[row['basis'], row['scenario']]['summary']['total_return']
            if not _finite(row['excess_vs_0050']) or not math.isclose(
                    row['excess_vs_0050'], row['summary']['total_return'] - bm, abs_tol=1e-12):
                raise ValueError('Inconsistent benchmark comparison')
            if row['rule'] == 'leader_now':
                if row['excess_vs_basket'] is not None:
                    raise ValueError('Unconfirmed leaders do not have a matched basket control')
            else:
                basket_key = ('basket_after', *key(row)[1:])
                delta = row['summary']['total_return'] - indexed[basket_key]['summary']['total_return']
                if not _finite(row['excess_vs_basket']) or not math.isclose(row['excess_vs_basket'], delta, abs_tol=1e-12):
                    raise ValueError('Inconsistent matched basket comparison')
        if (set(report['events']) != BASES or set(report['groups']) != BASES
                or any(not isinstance(report[key][basis], list) for key in ('events', 'groups') for basis in BASES)
                or not isinstance(report['limitations'], list) or not report['limitations']):
            raise ValueError('Incomplete signal coverage or caveats')
        return {**report, 'available': True}
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, OverflowError, ImportError):
        return {**missing, 'note': '族群擴散試驗不完整，或來源／程式／環境已變更；請重新執行 make prepare-diffusion 與 make research-diffusion。'}
