#!/usr/bin/env python3
"""Publish every fixed cooldown trial alongside sealed originals and 0050."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_poc_broker_account import Sources, account_metrics
from scripts.summarize_top3_validation import windows
from scripts.research_poc_optimization import BASE, SPEC
from skills.all_entry_accounts import digest, write
from skills.top3_validation import MODES, verify_costs
from app.account_analytics import realized_metrics

NAMES = {'poc_up_red': 'POC 硬篩＋紅 K', 'poc_priority': 'POC 優先＋紅 K', 'benchmark': '0050 含息基準'}
IDENTIFIER = 'poc_optimization_20261013'


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
    sources, cases, accounts, refs, attempts = Sources(root), {}, {}, [], []
    spec_ref = dict(path=str(SPEC.relative_to(ROOT)), sha256=digest(SPEC))
    sources.bind(**dict(name=spec_ref['path'], expected=spec_ref['sha256']))

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
                admission=case.get('optimization'), data_gap_exclusions=len(case.get('data_gap_exclusions', [])),
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
        metadata = report['optimization']
        arm, mode = metadata['arm'], metadata['mode']
        if (arm not in ('poc_up_red', 'poc_priority') or mode not in MODES
                or metadata['variant'] != 'cooldown5' or metadata['preregistration'] != spec_ref
                or set(report['cases']) != {arm}):
            raise ValueError('Unregistered optimization attempt')
        case = load(report['cases'][arm]['path'], report['cases'][arm]['sha256'])
        if any(case['optimization'].get(k) != v for k, v in metadata.items()):
            raise ValueError('Case and report optimization metadata differ')
        add(arm, 'cooldown5', mode, report['cases'][arm])
        trial_path = path.parent/'trials.jsonl'
        trial_sha = digest(trial_path)
        trial_rows = [json.loads(r) for r in sources.bind(str(trial_path.relative_to(root)), trial_sha).read_text().splitlines()]
        if len(trial_rows) != 1 or trial_rows[0].get('optimization') != metadata:
            raise ValueError('Expected one bound attempt per configuration')
        attempts.extend(trial_rows)
        refs.append(dict(path=str(path.relative_to(root)), sha256=sha,
            trials_path=str(trial_path.relative_to(root)), trials_sha256=trial_sha))
    expected = {f'{a}:{v}:{m}' for a in ('poc_up_red', 'poc_priority')
                for v in ('original', 'cooldown5') for m in MODES} | {f'benchmark:original:{m}' for m in MODES}
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
    decisions = {}
    for arm in ('poc_up_red', 'poc_priority'):
        improved, historical = True, True
        for mode in MODES:
            row, original, benchmark = (cases[f'{arm}:cooldown5:{mode}'],
                cases[f'{arm}:original:{mode}'], cases[f'benchmark:original:{mode}'])
            if not row['completed']:
                improved = historical = False
                continue
            new, old, base = row['metrics'], original['metrics'], benchmark['metrics']
            improved &= new['total_return'] > old['total_return'] and new['max_drawdown'] >= old['max_drawdown']
            historical &= new['total_return'] > 0 and row['excess_return'] > 0 and new['max_drawdown'] >= base['max_drawdown']
            row['change_from_original'] = {k: new[k]-old[k] for k in ('total_return', 'max_drawdown', 'trade_count')}
            old_events = {c['event_id'] for c in accounts[f'{arm}:original:{mode}']['cohorts']}
            new_events = {c['event_id'] for c in accounts[f'{arm}:cooldown5:{mode}']['cohorts']}
            row['funded_path_changes'] = dict(common=len(old_events & new_events),
                original_only=len(old_events-new_events), cooldown_only=len(new_events-old_events),
                note='比較實際入金批次；差異同時受候選抑制與後續資金、名額影響，不是逐筆損益歸因。')
        decisions[arm] = dict(improvement_pass=bool(improved), historical_stress_pass=bool(historical),
            unseen_validation=False, causal_fill_verified=False, paper_enabled=False, live_qualified=False)
    failures = []
    selected_reports = {Path(p).resolve() for p in paths}
    for path in sorted(BASE.glob('*/report.json')):
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
    for path in sorted(BASE.glob('*/failure.json')):
        failures.append(load(str(path.relative_to(root)), digest(path)))
    result = dict(schema='poc_optimization_publication_v1', start='2024-01-02', end='2026-10-02',
        created_at=datetime.now(timezone.utc).isoformat(), initial_cash=1000000, position_count=3,
        buy_fraction=.5, sell_fraction=.3, preregistration=spec_ref,
        modes={k: dict(commission=v[0], slippage=v[1]) for k, v in MODES.items()},
        cases=cases, decisions=decisions, expected_new_accounts=8,
        completed_new_accounts=sum(r['completed'] for r in cases.values() if r['variant']=='cooldown5'),
        trial_accounting=dict(published_attempts=len(attempts), earlier_failures=failures,
            all_attempts=attempts), reports=refs, checked_source_count=len(sources.used),
        execution_status='blocked_unverified_chronological_fills',
        next_stages=json.loads(SPEC.read_text())['next_fixed_order'],
        live_qualified=False, paper_enabled=False, actual_fill_verified=False, unseen_validation=False,
        reporting_sources={str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), ROOT/'app/account_analytics.py', ROOT/'scripts/summarize_poc_broker_account.py')})
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
