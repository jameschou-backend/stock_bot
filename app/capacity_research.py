"""Fail-closed reader for sealed capacity and same-day priority research."""
from collections import Counter
from datetime import date
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import re

from app import regime_switch_research

ROOT = Path(__file__).resolve().parents[1]
CACHE = '.cache/capacity-research'
CODE = {'scripts/research_capacity.py', 'skills/residual_priority.py'}
RULES = {'control3', 'capacity6', 'matched3', 'residual3'}
BASES = {'official', 'snapshot'}
SCENARIOS = {'base', 'stress'}
START, END, SIGNAL_END = '2022-01-03', '2026-06-23', '2025-12-31'
IDENTITY = ('rule', 'basis', 'scenario', 'delay')
PARENT_REPORT = '.cache/regime-switch-research/report.summary.json'
REFERENCES = {'control3': 'control3', 'capacity6': 'control3',
              'matched3': 'control3', 'residual3': 'matched3'}


@lru_cache(maxsize=128)
def _digest(path, size, modified):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = _digest(str(path), before.st_size, before.st_mtime_ns)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Research artifact changed while reading')
    return digest


def _json(payload):
    def invalid(value):
        raise ValueError('Non-finite number: ' + value)
    return json.loads(payload, parse_constant=invalid)


def read(path):
    return _json(Path(path).read_text())


def hashes(mapping, expected, base):
    if not isinstance(mapping, dict) or set(mapping) != expected:
        raise ValueError('Incomplete research provenance')
    for name, fingerprint in mapping.items():
        if sha(base / name) != fingerprint:
            raise ValueError('Changed research file: ' + name)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Invalid research date')
    date.fromisoformat(value)
    return value


def _ids(value):
    return (isinstance(value, list) and all(isinstance(v, str) and v for v in value)
            and len(set(value)) == len(value))


def _counts(value):
    return (isinstance(value, dict)
            and all(isinstance(k, str) and k and _count(v) for k, v in value.items()))


def _close(left, right):
    return _finite(left) and _finite(right) and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-11)


def _chart_digest(curve):
    if not isinstance(curve, list) or len(curve) < 2:
        raise ValueError('Missing account curve')
    points = []
    for row in curve:
        day = _date(row['date'])
        if (not _finite(row['nav']) or row['nav'] <= 0
                or points and day <= points[-1]['date']):
            raise ValueError('Invalid account curve')
        points.append({'date': day, 'nav': row['nav']})
    return hashlib.sha256(json.dumps(points, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _summary(row):
    s = row['summary']
    if (s['start'] != START or s['end'] != END or not _finite(s['initial_nav']) or s['initial_nav'] != 1
            or any(not _finite(s[k]) for k in ('total_return', 'cagr', 'max_drawdown', 'total_cost',
                'turnover', 'mean_active_weight', 'mean_cash_weight', 'final_nav', 'final_cash'))
            or s['final_nav'] <= 0 or s['final_cash'] < 0 or s['total_cost'] < 0 or s['turnover'] < 0
            or not -1 <= s['max_drawdown'] <= 0
            or not 0 <= s['mean_cash_weight'] <= 1 + 1e-10
            or not 0 <= s['mean_active_weight'] <= 1 + 1e-10
            or not _close(s['total_return'] + 1, s['final_nav'])):
        raise ValueError('Invalid strategy numbers')
    years = (date.fromisoformat(END) - date.fromisoformat(START)).days / 365.25
    if not _close(s['cagr'], s['final_nav'] ** (1 / years) - 1):
        raise ValueError('Annualized return does not match the account period')
    if (s['mode'] != ('benchmark' if row['rule'] == 'benchmark' else 'events')
            or not _count(s['horizon']) or s['horizon'] != 63
            or not _count(s['slots']) or s['slots'] != (6 if row['rule'] == 'capacity6' else 3)
            or s['commission_per_side'] != .001425 or s['stock_sell_tax'] != .003
            or s['benchmark_sell_tax'] != .001
            or s['slippage_per_side'] != {'base': .003, 'stress': .0045}[row['scenario']]):
        raise ValueError('Changed portfolio assumptions')
    for field in ('trade_count', 'completed_cohorts', 'entered_cohorts', 'peak_active_cohorts',
                  'blocked_exit_sessions', 'rejected_event_count', 'unliquidated_position_count'):
        if not _count(s[field]):
            raise ValueError('Invalid account count')
    if not s['completed_cohorts'] <= s['entered_cohorts'] or s['peak_active_cohorts'] > s['slots']:
        raise ValueError('Invalid cohort count')
    annual = s['annual_returns']
    if (not isinstance(annual, dict) or set(annual) != {'2022', '2023', '2024', '2025', '2026'}
            or any(not _finite(v) or v <= -1 for v in annual.values())
            or not _close(math.prod(1 + v for v in annual.values()), s['final_nav'])):
        raise ValueError('Annual compounding mismatch')
    positions = s['unliquidated_positions']
    if (not isinstance(positions, list) or len(positions) != s['unliquidated_position_count']
            or s['final_liquidation_complete'] is not (not positions)
            or s['final_nav_is_marked'] is not bool(positions)):
        raise ValueError('Missing liquidation warning')
    for position in positions:
        if (not isinstance(position['stock_id'], str)
                or not re.fullmatch(r'\d{4}', position['stock_id'])
                or any(not _finite(position[k]) or position[k] <= 0 for k in ('units', 'mark', 'marked_value'))
                or not _close(position['units'] * position['mark'], position['marked_value'])
                or _date(position['mark_date']) > END):
            raise ValueError('Invalid marked holding')
    if (len({p['stock_id'] for p in positions}) != len(positions)
            or not _close(s['final_cash'] + sum(p['marked_value'] for p in positions), s['final_nav'])):
        raise ValueError('Cash and marked positions do not reconcile')
    audit = row['valuation_audit']
    findings = audit['findings']
    if (not isinstance(findings, list) or not _count(audit['finding_count'])
            or not _count(audit['unresolved_valuation_days'])
            or audit['finding_count'] != len(findings)
            or audit['unresolved_valuation_days'] != len({_date(f['date']) for f in findings})):
        raise ValueError('Incomplete valuation audit')


def _account_curve(case):
    """Cold-read account reconciliation; retain no full curve in the header cache."""
    curve, summary = case['curve'], case['summary']
    fingerprint = _chart_digest(curve)
    if curve[0]['date'] != START or curve[-1]['date'] != END:
        raise ValueError('Account curve period differs')
    previous, cost, peak, drawdown = 1., 0., 1., 0.
    active_weights, cash_weights, year_ends = [], [], {}
    for point in curve:
        fields = ('nav', 'cash', 'active_value', 'benchmark_value', 'active_weight',
                  'daily_cost', 'cumulative_cost', 'market_pnl')
        if (any(not _finite(point[k]) for k in fields)
                or any(point[k] < 0 for k in ('cash', 'active_value', 'benchmark_value', 'daily_cost'))
                or not _count(point['active_cohorts']) or point['active_cohorts'] > summary['slots']):
            raise ValueError('Invalid account observations')
        cost += point['daily_cost']
        nav = point['nav']
        if (not _close(nav, point['cash'] + point['active_value'] + point['benchmark_value'])
                or not _close(nav, previous + point['market_pnl'] - point['daily_cost'])
                or not _close(cost, point['cumulative_cost'])
                or not _close(point['active_weight'], point['active_value'] / nav)):
            raise ValueError('Daily account does not reconcile')
        peak = max(peak, nav)
        drawdown = min(drawdown, nav / peak - 1)
        active_weights.append(point['active_weight'])
        cash_weights.append(point['cash'] / nav)
        year_ends[point['date'][:4]] = nav
        previous = nav
    if set(year_ends) != set(summary['annual_returns']):
        raise ValueError('Account curve is missing a reported year')
    previous = 1.
    for year, nav in year_ends.items():
        if not _close(summary['annual_returns'][year], nav / previous - 1):
            raise ValueError('Annual returns differ from the account curve')
        previous = nav
    expected = {'final_nav': curve[-1]['nav'], 'final_cash': curve[-1]['cash'], 'total_cost': cost,
                'max_drawdown': drawdown, 'mean_active_weight': sum(active_weights) / len(curve),
                'mean_cash_weight': sum(cash_weights) / len(curve)}
    if any(not _close(summary[key], value) for key, value in expected.items()):
        raise ValueError('Account summary differs from its curve')
    return fingerprint


@lru_cache(maxsize=128)
def _case_header(path, fingerprint):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != fingerprint:
        raise ValueError('Case changed during header verification')
    case = _json(payload)
    for field in ('executions', 'cohorts', 'rejections', 'gate_rejections', 'score_diagnostics', 'score_rejections'):
        if not isinstance(case[field], list) or any(not isinstance(row, dict) for row in case[field]):
            raise ValueError('Invalid detailed account records')
    cohort_ids = [row['event_id'] for row in case['cohorts']]
    rejection_counts = dict(Counter(row['reason'] for row in case['rejections']))
    if (not _ids(cohort_ids) or not _counts(rejection_counts)
            or len(case['executions']) != case['summary']['trade_count']
            or len(case['cohorts']) != case['summary']['entered_cohorts']
            or len(case['rejections']) != case['summary']['rejected_event_count']):
        raise ValueError('Account records do not match their counts')
    if case['rule'] == 'benchmark' and any(case[k] for k in ('gate_rejections', 'score_diagnostics', 'score_rejections')):
        raise ValueError('Benchmark cannot have signal diagnostics')
    header = {key: case[key] for key in ('summary', 'valuation_audit', *IDENTITY)}
    header['rejection_counts'] = rejection_counts
    header['cohort_ids'] = sorted(cohort_ids)
    return header, _account_curve(case)


def _signal_stats(stats):
    if not isinstance(stats, dict) or set(stats) != BASES:
        raise ValueError('Incomplete signal statistics')
    for values in stats.values():
        for key in ('original_count', 'trend_count', 'scoreable_trend_count', 'multiple_event_days', 'reordered_days'):
            if not _count(values[key]):
                raise ValueError('Invalid signal count')
        examples = values['reorder_examples']
        if (not values['scoreable_trend_count'] <= values['trend_count'] <= values['original_count']
                or not values['reordered_days'] <= values['multiple_event_days'] <= values['scoreable_trend_count'] // 2
                or not _counts(values['score_rejections'])
                or sum(values['score_rejections'].values()) != values['trend_count'] - values['scoreable_trend_count']
                or not isinstance(examples, list) or len(examples) != min(10, values['reordered_days'])):
            raise ValueError('Signal counts do not reconcile')
        dates = []
        for example in examples:
            dates.append(_date(example['date']))
            original, residual = example['original_order'], example['residual_order']
            if (not START <= dates[-1] <= END or not _ids(original) or not _ids(residual)
                    or len(original) < 2 or set(original) != set(residual) or original == residual):
                raise ValueError('Invalid same-day ordering diagnostic')
        if dates != sorted(set(dates)):
            raise ValueError('Invalid ordering diagnostic dates')


def overview():
    missing = {'available': False, 'research_only': True, 'live_qualified': False,
               'valid_strategy_evidence': False,
               'note': '尚無完整的部位容量與排序研究；先執行 make prepare-capacity，再執行 make research-capacity。'}
    folder = ROOT / CACHE
    if not (folder / 'report.summary.json').exists():
        return missing
    try:
        report = read(folder / 'report.summary.json')
        if (type(report['schema']) is not int or report['schema'] != 1
                or report['experiment'] != 'capacity_priority_20260910'
                or report['research_only'] is not True or report['live_qualified'] is not False
                or report['valid_strategy_evidence'] is not False
                or report['control_reproduction_passed'] is not True
                or (report['start'], report['end'], report['signal_end']) != (START, END, SIGNAL_END)
                or not _count(report['finmind_requests']) or report['finmind_requests'] != 0
                or any(not _finite(report[k]) or report[k] < 0 for k in ('elapsed_seconds', 'preparation_elapsed_seconds'))):
            raise ValueError('Unsupported research')
        hashes(report['code_sha256'], CODE, ROOT)
        if sha(ROOT / 'docs/prereg_capacity_20260910.md') != report['protocol_sha256']:
            raise ValueError('Changed protocol')
        if sha(folder / 'signals.json') != report['signal_manifest_sha256']:
            raise ValueError('Changed signal manifest')
        signals = read(folder / 'signals.json')
        if (signals != report['signals'] or type(signals['schema']) is not int or signals['schema'] != 1
                or signals['code_sha256'] != report['code_sha256']
                or signals['protocol_sha256'] != report['protocol_sha256']
                or signals['prefix_invariance_passed'] is not True
                or signals['versions'] != {name: version(name) for name in ('numpy', 'pandas')}
                or not _count(signals['finmind_requests']) or signals['finmind_requests'] != 0
                or not _finite(signals['elapsed_seconds']) or signals['elapsed_seconds'] < 0
                or signals['elapsed_seconds'] != report['preparation_elapsed_seconds']):
            raise ValueError('Changed signal source or runtime')
        hashes(signals['files_sha256'], {'signals-official.json', 'signals-snapshot.json'}, folder)
        parent = regime_switch_research.overview()
        if (parent['available'] is not True
                or report['parent_report_sha256'] != sha(ROOT / PARENT_REPORT)
                or signals['parent_report_sha256'] != report['parent_report_sha256']
                or signals['parent_signal_manifest_sha256'] != parent['states']['diffusion_signal_manifest_sha256']):
            raise ValueError('Parent research changed')
        _signal_stats(report['signal_stats'])
        if report['signal_stats'] != signals['stats']:
            raise ValueError('Displayed signal statistics changed')
        expected = {(r, b, c, 0) for r in RULES for b in BASES for c in SCENARIOS}
        expected |= {(r, 'official', 'stress', 1) for r in RULES}
        key = lambda r: tuple(r[k] for k in IDENTITY)
        rows, benchmarks = report['results'], report['baselines']
        if (not isinstance(rows, list) or len(rows) != 20 or {key(r) for r in rows} != expected
                or any(not _count(r['delay']) for r in rows)):
            raise ValueError('Incomplete twenty contrasts')
        if (not isinstance(benchmarks, list) or len(benchmarks) != 4
                or {key(r) for r in benchmarks} != {('benchmark', b, c, 0) for b in BASES for c in SCENARIOS}
                or any(not _count(r['delay']) for r in benchmarks)):
            raise ValueError('Incomplete benchmark controls')
        cases = {f'case-{r}-{b}-{c}-{d}.json' for r, b, c, d in expected}
        cases |= {f'case-benchmark-{b}-{c}-0.json' for b in BASES for c in SCENARIOS}
        hashes(report['case_files_sha256'], cases, folder)
        if (not isinstance(report['charts'], dict) or set(report['charts']) != RULES | {'benchmark'}
                or not isinstance(report['limitations'], list) or not report['limitations']
                or any(not isinstance(v, str) or not v for v in report['limitations'])):
            raise ValueError('Missing charts or limitations')
        headers = {}
        for row in rows + benchmarks:
            _summary(row)
            if not isinstance(row['name'], str) or not row['name'] or not _counts(row['rejection_counts']):
                raise ValueError('Invalid displayed case metadata')
            if row['case_file'] != 'case-{}-{}-{}-{}.json'.format(*key(row)):
                raise ValueError('Incorrect case link')
            header, chart_hash = _case_header(str(folder / row['case_file']), report['case_files_sha256'][row['case_file']])
            if any(row[k] != v for k, v in header.items() if k != 'cohort_ids'):
                raise ValueError('Displayed summary differs from the sealed account')
            headers[key(row)] = header
            if (row['basis'], row['scenario'], row['delay']) == ('official', 'stress', 0):
                if _chart_digest(report['charts'][row['rule']]) != chart_hash:
                    raise ValueError('Displayed chart differs from the sealed account')
        indexed = {key(row): row for row in rows + benchmarks}
        for row in rows:
            rule, basis, scenario, delay = key(row)
            benchmark = indexed[('benchmark', basis, scenario, 0)]['summary']['total_return']
            reference = indexed[(REFERENCES[rule], basis, scenario, delay)]['summary']['total_return']
            if (row['reference_rule'] != REFERENCES[rule]
                    or not _close(row['excess_vs_0050'], row['summary']['total_return'] - benchmark)
                    or not _close(row['contrast_vs_reference'], row['summary']['total_return'] - reference)):
                raise ValueError('Inconsistent comparison')
        comparisons = report['cohort_comparisons']
        comparison_keys = {(r, b, c, d) for r, b, c, d in expected if r != 'control3'}
        if (not isinstance(comparisons, list) or len(comparisons) != 15
                or {key(r) for r in comparisons} != comparison_keys):
            raise ValueError('Incomplete cohort comparisons')
        for row in comparisons:
            rule, basis, scenario, delay = key(row)
            ours = set(headers[(rule, basis, scenario, delay)]['cohort_ids'])
            theirs = set(headers[(REFERENCES[rule], basis, scenario, delay)]['cohort_ids'])
            if (not _count(delay) or row['reference'] != REFERENCES[rule] or not _count(row['shared'])
                    or not _ids(row['only_rule']) or not _ids(row['only_reference'])
                    or row['shared'] != len(ours & theirs) or set(row['only_rule']) != ours - theirs
                    or set(row['only_reference']) != theirs - ours):
                raise ValueError('Cohort comparison differs from the sealed accounts')
        return {**report, 'available': True}
    except (OSError, ValueError, KeyError, TypeError, ImportError, OverflowError, AttributeError):
        return {**missing, 'note': '部位容量與排序研究不完整，或來源／程式已變更；請重新準備訊號與研究結果。'}


def load_case(report, row):
    """Load one sealed member of the verified report; never an arbitrary path."""
    try:
        if report.get('available') is not True or row not in report['results'] + report['baselines']:
            raise ValueError('Case is not in the verified report')
        name = row['case_file']
        if not isinstance(name, str) or Path(name).name != name or sha(ROOT / CACHE / name) != report['case_files_sha256'][name]:
            raise ValueError('Case changed')
        header, _ = _case_header(str(ROOT / CACHE / name), report['case_files_sha256'][name])
        if any(row[k] != value for k, value in header.items() if k != 'cohort_ids'):
            raise ValueError('Case summary differs from displayed result')
        payload = (ROOT / CACHE / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != report['case_files_sha256'][name]:
            raise ValueError('Case changed while reading detail')
        return {**_json(payload), 'available': True}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {'available': False, 'note': '這份成交紀錄已變更或不完整，請重新研究後再查看。'}
