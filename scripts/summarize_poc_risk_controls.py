#!/usr/bin/env python3
"""Publish all fixed follow-ups, original controls, same-cost 0050 and execution gaps."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_poc_broker_account import Sources, account_metrics
from scripts.summarize_top3_validation import windows
from scripts.research_poc_risk_controls import BASE, SPEC
from skills.poc_risk_controls import VARIANTS, audit_controls
from skills.all_entry_accounts import digest, write
from skills.top3_validation import MODES, verify_costs
from app.account_analytics import realized_metrics
import pandas as pd

NAMES = {'poc_up_red': 'POC 硬篩＋紅 K', 'poc_priority': 'POC 優先＋紅 K', 'benchmark': '0050 含息基準'}
IDENTIFIER = 'poc_risk_controls_20261014'


class PublicationSources(Sources):
    """Check shared source bytes once before publication, retaining all bindings.

Reports and account files are verified as they are loaded. Their overlapping
source inventories are collected with conflict checks, then fully verified
together before build returns. No result may be published before finish().
"""
    def __init__(self, root):
        super().__init__(root)
        self.pending = {}

    def bind(self, name, expected):
        if name in self.pending and self.pending[name] != expected:
            raise ValueError('Conflicting publication source '+name)
        return super().bind(name, expected)

    def closure(self, report, report_name):
        refs = report.get('source_sha256')
        if not refs:
            raise ValueError('Missing source closure '+report_name)
        for name, expected in refs.items():
            snapshot = report.get('source_snapshots', {}).get(name)
            if snapshot and Path(name).suffix in ('.py', '.md'):
                if snapshot['sha256'] != expected:
                    raise ValueError('Source snapshot differs from original hash')
                self.snapshot_resolutions.append(dict(report=report_name, original=name,
                    path=snapshot['path'], sha256=expected))
                name = snapshot['path']
            known = self.pending.get(name, self.used.get(name))
            if known is not None and known != expected:
                raise ValueError('Conflicting publication source '+name)
            self.pending[name] = expected

    def finish(self):
        expected = dict(self.pending)
        expected.update(self.used)
        for name, sha in expected.items():
            self.bind(name, sha)
        self.pending.clear()



def attempt_paths(name):
    """Source snapshots are evidence copies, never independent account attempts."""
    for directory, children, files in os.walk(BASE):
        children[:] = [child for child in children if child != 'source-snapshots']
        if name in files:
            yield Path(directory)/name


def execution_coverage(account):
    orders = [r for r in account['orders'] if r['side'] == 'buy']
    board = [r for r in orders if r['channel'] == 'board']
    odd = [r for r in orders if r['channel'] == 'odd']
    return dict(board_buy_orders=len(board),
        board_complete_sequence=sum(r.get('tape_evidence', {}).get('tick_sequence_complete') is True for r in board),
        odd_buy_orders=len(odd), odd_chronological_verified=sum(r.get('odd_tick_verified') is True for r in odd),
        missing_chronology=[dict(date=r['date'], stock_id=r['stock_id'], channel=r['channel'],
            required='完整普通盤逐筆序列與日總量核對' if r['channel']=='board' else '盤中零股逐筆時間、價格與成交量',
            current_evidence=r.get('execution_evidence')) for r in board+odd
            if not (r.get('tape_evidence', {}).get('tick_sequence_complete') is True if r['channel']=='board'
                    else r.get('odd_tick_verified') is True)],
        actual_fill_verified=False,
        scope='封存基準買單的證據標示；不是供應商全部歷史資料的覆蓋率')


def build(paths, root=ROOT):
    root = Path(root)
    sources, cases, accounts, refs, attempts = PublicationSources(root), {}, {}, [], []
    spec_ref = dict(path=str(SPEC.relative_to(ROOT)), sha256=digest(SPEC))
    sources.bind(**dict(name=spec_ref['path'], expected=spec_ref['sha256']))
    for name, expected in json.loads(SPEC.read_text())['original_publications'].items():
        sources.bind(name, expected)

    def load(name, sha):
        return json.loads(sources.bind(name, sha).read_text())

    def add(arm, variant, mode, descriptor):
        key = f'{arm}:{variant}:{mode}'
        if key in cases:
            raise ValueError('Duplicate published configuration; select no best-of retries')
        case = load(descriptor['path'], descriptor['sha256'])
        row = dict(id=arm, variant=variant, mode=mode, name=NAMES[arm],
            path=descriptor['path'], sha256=descriptor['sha256'], completed=case['completed'],
            reason=case.get('reason'), summary=case.get('summary'), metrics=None, analytics=None)
        if row['completed']:
            account = case['account']
            verify_costs(account, mode)
            metrics = account_metrics(case)
            if (metrics['start'], metrics['end'], metrics['initial_cash'], metrics['trading_days']) != (
                    '2024-01-02', '2026-10-02', 1000000, 666):
                raise ValueError('Optimization account scope differs')
            settings = account['settings']
            if (settings['commission'], settings['slippage']) != MODES[mode]:
                raise ValueError('Optimization cost mode differs')
            row.update(metrics=metrics, analytics=realized_metrics(account),
                execution_coverage=execution_coverage(account),
                admission=case.get('risk_controls'), data_gap_exclusions=len(case.get('data_gap_exclusions', [])),
                profile_data_exclusions=len(case.get('profile_data_exclusions', [])),
                selection_fallback_days=case.get('selection_fallback_days', 0))
            accounts[key] = account
        cases[key] = row

    for identifier, arm in [('top3_validation_20261012', 'poc_up_red'),
                            ('legacy_return_comparison_20261012', 'poc_priority')]:
        path = root/'artifacts/forward_simulation'/(identifier+'.json')
        sha = path.with_suffix('.sha256').read_text().strip()
        publication = load(str(path.relative_to(root)), sha)
        refs.append(dict(path=str(path.relative_to(root)), sha256=sha, reused_baseline=True))
        for mode in MODES:
            add(arm, 'original', mode, publication['cases'][f'{arm}:{mode}'])
            if arm == 'poc_up_red':
                add('benchmark', 'original', mode, publication['cases'][f'benchmark:{mode}'])
            else:
                descriptor = publication['cases'][f'benchmark:{mode}']
                other = load(descriptor['path'], descriptor['sha256'])
                verify_costs(other['account'], mode)
                left, right = other['account']['daily'], accounts[f'benchmark:original:{mode}']['daily']
                if (len(left) != len(right) or any(a['date'] != b['date'] or abs(a['nav']-b['nav']) > .011
                        for a, b in zip(left, right))):
                    raise ValueError('The two original studies do not share the same 0050 path')

    for supplied in paths:
        path = Path(supplied).resolve()
        sha = path.with_suffix('.sha256').read_text().strip()
        report = load(str(path.relative_to(root)), sha)
        sources.closure(report, str(path.relative_to(root)))
        metadata = report['risk_controls']
        arm, mode = metadata['arm'], metadata['mode']
        if (arm not in ('poc_up_red', 'poc_priority') or mode not in MODES
                or metadata['variant'] not in VARIANTS or metadata['preregistration'] != spec_ref
                or set(report['cases']) != {arm}):
            raise ValueError('Unregistered optimization attempt')
        case = load(report['cases'][arm]['path'], report['cases'][arm]['sha256'])
        if any(case['risk_controls'].get(k) != v for k, v in metadata.items()):
            raise ValueError('Case and report optimization metadata differ')
        add(arm, metadata['variant'], mode, report['cases'][arm])
        if case['completed']:
            adjusted_path = '.cache/poc-latest-20261003/inputs-v1/close-official.parquet'
            sources.bind(adjusted_path, report['source_sha256'][adjusted_path])
            adjusted = pd.read_parquet(root/adjusted_path).set_index('date')
            check = dict(audit_controls(case['account'], adjusted), passed=True, variant=metadata['variant'])
            cases[f"{arm}:{metadata['variant']}:{mode}"]['control_audit'] = check
            exchanges = [a for a in case['account']['corporate_actions'] if a['kind'] == 'share_exchange']
            if exchanges:
                addendum = root/'docs/poc_followup_merger_black_20261014.json'
                load(str(addendum.relative_to(root)), digest(addendum))
            cases[f"{arm}:{metadata['variant']}:{mode}"]['corporate_completion'] = dict(
                share_exchanges=len(exchanges), posthoc_model_completion=bool(exchanges),
                note='2867換股保留持有天數與停損；2884黑K從9/8普通股可交易日起重新累積。畸零款未列入可用現金。' if exchanges else None)
        trial_path = path.parent/'trials.jsonl'
        trial_sha = digest(trial_path)
        trial_rows = [json.loads(r) for r in sources.bind(str(trial_path.relative_to(root)), trial_sha).read_text().splitlines()]
        if len(trial_rows) != 1 or trial_rows[0].get('risk_controls') != metadata:
            raise ValueError('Expected one bound attempt per configuration')
        attempts.extend(trial_rows)
        refs.append(dict(path=str(path.relative_to(root)), sha256=sha,
            trials_path=str(trial_path.relative_to(root)), trials_sha256=trial_sha))
    expected = {f'{a}:{v}:{m}' for a in ('poc_up_red', 'poc_priority')
                for v in ('original', *VARIANTS) for m in MODES} | {f'benchmark:original:{m}' for m in MODES}
    if set(cases) != expected:
        raise ValueError('Publish the complete fixed matrix, including incomplete accounts')
    for key, row in cases.items():
        if not row['completed']:
            continue
        base_key = f"benchmark:original:{row['mode']}"
        base = cases[base_key]['metrics']
        row['excess_return'] = row['metrics']['total_return'] - base['total_return']
        row['annual_excess'] = [dict(year=r['year'], excess=r['total_return']-b['total_return'],
            partial_year=r['partial_year']) for r, b in zip(row['metrics']['annual'], base['annual'])]
        row['rolling252'] = windows(accounts[key]['daily'], accounts[base_key]['daily'])
    for key, row in cases.items():
        if row['variant'] not in VARIANTS or not row['completed']:
            continue
        original = accounts[f"{row['id']}:original:{row['mode']}"]
        candidate = accounts[key]
        fields = {
            'daily': ('date', 'cash', 'nav', 'market_value', 'receivable', 'drawdown', 'holdings'),
            'trades': ('date', 'signal_date', 'event_id', 'stock_id', 'side', 'qty',
                       'reference_price', 'cash_change', 'total_cost'),
        }
        row['same_funded_path_as_original'] = all(
            [{k: r.get(k) for k in names} for r in candidate[section]]
            == [{k: r.get(k) for k in names} for r in original[section]]
            for section, names in fields.items())
    decisions = {}
    for arm in ('poc_up_red', 'poc_priority'):
        for variant in VARIANTS:
            improved, historical = True, True
            for mode in MODES:
                row, original, benchmark = (cases[f'{arm}:{variant}:{mode}'],
                    cases[f'{arm}:original:{mode}'], cases[f'benchmark:original:{mode}'])
                if not row['completed']:
                    improved = historical = False
                    continue
                new, old, base = row['metrics'], original['metrics'], benchmark['metrics']
                improved &= new['total_return'] > old['total_return'] and new['max_drawdown'] >= old['max_drawdown']
                historical &= new['total_return'] > 0 and row['excess_return'] > 0 and new['max_drawdown'] >= base['max_drawdown']
                row['change_from_original'] = {k: new[k]-old[k] for k in ('total_return', 'max_drawdown', 'trade_count')}
            baseline = cases[f'{arm}:{variant}:baseline']
            original = cases[f'{arm}:original:baseline']
            primary = baseline['completed'] and baseline['metrics']['total_return'] > original['metrics']['total_return'] and baseline['metrics']['max_drawdown'] >= original['metrics']['max_drawdown']
            decisions[f'{arm}:{variant}'] = dict(improvement_pass=bool(primary), all_cost_improvement=bool(improved), historical_stress_pass=bool(historical),
                unseen_validation=False, causal_fill_verified=False, paper_enabled=False, live_qualified=False)
    failures = []
    selected_reports = {Path(p).resolve() for p in paths}
    for path in sorted(attempt_paths('report.json')):
        if path.resolve() in selected_reports:
            continue
        sha = path.with_suffix('.sha256').read_text().strip()
        earlier = load(str(path.relative_to(root)), sha)
        if earlier.get('all_completed') is not False:
            raise ValueError('An unpublished completed attempt exists; no best-of selection')
        sources.closure(earlier, str(path.relative_to(root)))
        failed = [dict(arm=a, completed=r['completed'], reason=r.get('reason')) for a, r in earlier['cases'].items()]
        failures.append(dict(path=str(path.relative_to(root)), sha256=sha, cases=failed))
    # Setup failures cannot disappear just because they have no completed report.
    for path in sorted([*attempt_paths('failure.json'), *attempt_paths('setup-failure.json'), *attempt_paths('process-failures.json')]):
        failure = load(str(path.relative_to(root)), digest(path))
        archived_source = failure.get('source')
        if isinstance(archived_source, dict):
            sources.bind(archived_source['path'], archived_source['sha256'])
        failures.append(failure)
    execution_path = root/'.cache/poc-followup-20261014/execution-audit-v1.json'
    execution_sha = execution_path.with_suffix('.sha256').read_text().strip()
    execution = load(str(execution_path.relative_to(root)), execution_sha)
    sources.closure(execution, str(execution_path.relative_to(root)))
    if execution.get('passed') is not False or execution.get('whole_account_replayed') is not False:
        raise ValueError('Frozen-order diagnostics cannot qualify a causal account')
    requirements_path = root/'docs/poc_followup_execution_requirements_20261014.json'
    requirements = load(str(requirements_path.relative_to(root)), digest(requirements_path))
    sources.closure(requirements, str(requirements_path.relative_to(root)))
    execution_summary = dict(requirements={k: v for k, v in requirements.items() if k not in ('coordinates', 'source_sha256')}, path=str(execution_path.relative_to(root)), sha256=execution_sha,
        completed=execution['completed'], passed=False, whole_account_replayed=False,
        unique_ordinary_coordinates=execution['unique_ordinary_coordinates'],
        finmind_requests=execution['finmind_requests'], blockers=execution['blockers'],
        cases={k: {name: value for name, value in row.items() if name != 'rows'}
               for k, row in execution['cases'].items()})
    # Every collected source, including directly loaded files, is checked here.
    # The CLI writes nothing unless this final verification succeeds.
    sources.finish()
    result = dict(schema='poc_risk_controls_publication_v1', start='2024-01-02', end='2026-10-02',
        created_at=datetime.now(timezone.utc).isoformat(), initial_cash=1000000, position_count=3,
        buy_fraction=.5, sell_fraction=.3, preregistration=spec_ref,
        modes={k: dict(commission=v[0], slippage=v[1]) for k, v in MODES.items()},
        cases=cases, decisions=decisions, execution_audit=execution_summary, expected_new_accounts=16,
        completed_new_accounts=sum(r['completed'] for r in cases.values() if r['variant'] in VARIANTS),
        trial_accounting=dict(published_attempts=len(attempts), earlier_failures=failures,
            all_attempts=attempts), reports=refs, checked_source_count=len(sources.used),
        execution_status='blocked_unverified_chronological_fills',
        variants=list(VARIANTS), stage_order=list(VARIANTS),
        live_qualified=False, paper_enabled=False, actual_fill_verified=False, unseen_validation=False,
        reporting_sources={str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), ROOT/'app/account_analytics.py', ROOT/'scripts/summarize_poc_broker_account.py', ROOT/'skills/poc_risk_controls.py')})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.reports)
    path = ROOT/'artifacts/forward_simulation'/(IDENTIFIER+'.json')
    if path.exists():
        raise ValueError('Optimization publication is immutable; use a new version')
    write(path, result)
    path.with_suffix('.sha256').write_text(digest(path)+'\n')
    print(json.dumps(dict(path=str(path), completed=result['completed_new_accounts'])))
