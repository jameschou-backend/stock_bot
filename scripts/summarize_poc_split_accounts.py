#!/usr/bin/env python3
"""Publish eight new sleeves, four summed portfolios and twelve reused controls."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_poc_split_accounts import BASE, SPEC
from scripts.summarize_poc_risk_controls import PublicationSources, execution_coverage, NAMES
from scripts.summarize_poc_broker_account import account_metrics
from scripts.summarize_top3_validation import windows
from scripts.replay_million import summarize
from skills.all_entry_accounts import digest, write
from skills.top3_validation import MODES, verify_costs
from skills.poc_split_accounts import audit_funding
from skills.poc_split_aggregation import ARMS, combine
from app.account_analytics import realized_metrics

IDENTIFIER = 'poc_split_accounts_20261015'


def build(paths, root=ROOT):
    root = Path(root).resolve()
    sources = PublicationSources(root)
    cases, accounts, reports, attempts, failures = {}, {}, [], [], []
    spec_ref = dict(path=str(SPEC.relative_to(ROOT)), sha256=digest(SPEC))

    def load(name, sha):
        return json.loads(sources.bind(name, sha).read_text())

    prereg = load(spec_ref['path'], spec_ref['sha256'])
    for name, expected in prereg['original_publications'].items(): sources.bind(name, expected)
    prior_name = 'artifacts/forward_simulation/poc_risk_controls_20261014.json'
    prior = load(prior_name, prereg['original_publications'][prior_name])

    def add(arm, variant, mode, descriptor, case=None):
        key = f'{arm}:{variant}:{mode}'
        if key in cases: raise ValueError('Duplicate configuration; no best-of selection')
        case = case or load(descriptor['path'], descriptor['sha256'])
        if case.get('completed') is not True: raise ValueError('All split accounts must complete before publication')
        account = case['account']
        verify_costs(account, mode)
        metrics = account_metrics(case)
        initial = 500000 if variant == 'split50' else 1000000
        if (metrics['start'], metrics['end'], metrics['initial_cash'], metrics['trading_days']) != (
                '2024-01-02', '2026-10-02', initial, 666):
            raise ValueError('Split comparison scope differs')
        if (account['settings']['commission'], account['settings']['slippage']) != MODES[mode]:
            raise ValueError('Split cost mode differs')
        row = dict(id=arm, variant=variant, mode=mode,
            name='兩套原版各50萬・不再平衡' if arm == 'combined' else NAMES[arm],
            **descriptor, completed=True, summary=case['summary'], metrics=metrics,
            analytics=realized_metrics(account), reused=variant == 'original',
            data_gap_exclusions=len(case.get('data_gap_exclusions', [])),
            profile_data_exclusions=len(case.get('profile_data_exclusions', [])),
            selection_fallback_days=case.get('selection_fallback_days', 0))
        if variant == 'split50':
            row.update(funding_audit=audit_funding(account), execution_coverage=execution_coverage(account))
        if variant == 'combined':
            row.update(aggregation_audit=account['aggregation_audit'], shared_capacity=account['shared_capacity'])
        cases[key], accounts[key] = row, account

    for mode in MODES:
        for arm in (*ARMS, 'benchmark'):
            row = prior['cases'][f'{arm}:original:{mode}']
            add(arm, 'original', mode, {k: row[k] for k in ('path', 'sha256')})
    reports.append(dict(path=prior_name, sha256=prereg['original_publications'][prior_name], reused_accounts=12))
    for supplied in paths:
        path = Path(supplied).resolve()
        name = str(path.relative_to(root))
        sha = path.with_suffix('.sha256').read_text().strip()
        report = load(name, sha)
        metadata = report['split_accounts']
        arm, mode = metadata['arm'], metadata['mode']
        if (arm not in ARMS or mode not in MODES or metadata['variant'] != 'split50'
                or metadata['preregistration'] != spec_ref or report['initial_cash'] != 500000
                or metadata['initial_cash'] != 500000 or metadata['rebalance'] is not False
                or metadata['cash_transfers'] is not False or report['all_completed'] is not True
                or set(report['cases']) != {arm}):
            raise ValueError('Unregistered or incomplete split report')
        sources.closure(report, name)
        descriptor = {k: report['cases'][arm][k] for k in ('path', 'sha256')}
        case = load(descriptor['path'], descriptor['sha256'])
        if case.get('split_accounts') != metadata: raise ValueError('Split case metadata differs')
        add(arm, 'split50', mode, descriptor, case)
        trial_path = path.parent/'trials.jsonl'
        trial_sha = digest(trial_path)
        trial_rows = [json.loads(r) for r in sources.bind(str(trial_path.relative_to(root)), trial_sha).read_text().splitlines()]
        if len(trial_rows) != 1 or trial_rows[0].get('split_accounts') != metadata:
            raise ValueError('Each new sleeve requires exactly one recorded attempt')
        attempts.extend(trial_rows)
        reports.append(dict(path=name, sha256=sha, trials_path=str(trial_path.relative_to(root)), trials_sha256=trial_sha))
    expected = {f'{a}:{v}:{m}' for a in ARMS for v in ('original', 'split50') for m in MODES}
    expected |= {f'benchmark:original:{m}' for m in MODES}
    if set(cases) != expected: raise ValueError('Missing a fixed split configuration')

    for mode in MODES:
        for arm in ARMS:
            old = {c['event_id'] for c in accounts[f'{arm}:original:{mode}']['cohorts']}
            new = {c['event_id'] for c in accounts[f'{arm}:split50:{mode}']['cohorts']}
            cases[f'{arm}:split50:{mode}']['funded_path_comparison'] = dict(
                original_cohorts=len(old), sleeve_cohorts=len(new), common_cohorts=len(old & new),
                only_original=len(old-new), only_sleeve=len(new-old))
    derived = {}
    for mode in MODES:
        sleeve_accounts = {a: accounts[f'{a}:split50:{mode}'] for a in ARMS}
        account = combine(sleeve_accounts, mode)
        refs = {a: {k: cases[f'{a}:split50:{mode}'][k] for k in ('path', 'sha256')} for a in ARMS}
        case = dict(completed=True, account=account, summary=summarize(account), sleeve_sources=refs,
            live_qualified=False, actual_fill_verified=False, unseen_validation=False, paper_enabled=False)
        target = BASE/'combined'/f'{mode}.json'
        derived[mode] = (target, case)
        add('combined', 'combined', mode, dict(path=str(target.relative_to(ROOT)), sha256=None), case)
        for field in ('data_gap_exclusions', 'profile_data_exclusions', 'selection_fallback_days'):
            cases[f'combined:combined:{mode}'][field] = sum(cases[f'{a}:split50:{mode}'][field] for a in ARMS)
    for key, row in cases.items():
        benchmark_key = f"benchmark:original:{row['mode']}"
        base = cases[benchmark_key]['metrics']
        row['excess_return'] = row['metrics']['total_return']-base['total_return']
        row['rolling252'] = windows(accounts[key]['daily'], accounts[benchmark_key]['daily'])
    comparisons = {}
    for mode in MODES:
        combined = cases[f'combined:combined:{mode}']['metrics']
        comparisons[mode] = {}
        for arm in (*ARMS, 'benchmark'):
            old = cases[f'{arm}:original:{mode}']['metrics']
            comparisons[mode][arm] = dict(
                return_change=combined['total_return']-old['total_return'],
                return_sacrifice=old['total_return']-combined['total_return'],
                drawdown_improvement=combined['max_drawdown']-old['max_drawdown'],
                drawdown_relative_reduction=(combined['max_drawdown']-old['max_drawdown'])/abs(old['max_drawdown']),
                profit_change=combined['profit']-old['profit'])
    selected = {Path(p).resolve() for p in paths}
    for directory, children, files in os.walk(BASE):
        children[:] = [c for c in children if c != 'source-snapshots']
        for filename in ('report.json', 'failure.json', 'setup-failure.json'):
            if filename not in files: continue
            path = Path(directory)/filename
            if path.resolve() in selected: continue
            failure = load(str(path.relative_to(root)), digest(path))
            if filename == 'report.json':
                if failure.get('all_completed') is not False: raise ValueError('Unpublished completed attempt')
                sources.closure(failure, str(path.relative_to(root)))
            failures.append(dict(path=str(path.relative_to(root)), sha256=digest(path), reason=failure.get('reason'),
                cases={a: dict(completed=r['completed'], reason=r.get('reason')) for a, r in failure.get('cases', {}).items()}))
    reporting = [Path(__file__), ROOT/'skills/poc_split_aggregation.py', ROOT/'skills/poc_split_accounts.py',
        ROOT/'scripts/replay_million.py', ROOT/'scripts/summarize_poc_broker_account.py',
        ROOT/'scripts/summarize_poc_risk_controls.py', ROOT/'scripts/summarize_top3_validation.py', ROOT/'app/account_analytics.py', ROOT/'app/poc_split_accounts_ui.py']
    for p in reporting: sources.bind(str(p.relative_to(ROOT)), digest(p))
    sources.finish()
    for mode, (path, case) in derived.items():
        if path.exists():
            if json.loads(path.read_text()) != case: raise ValueError('Immutable combined account differs')
        else: write(path, case)
        sha = digest(path)
        cases[f'combined:combined:{mode}']['sha256'] = sha
        sources.bind(str(path.relative_to(ROOT)), sha)
    return dict(schema='poc_split_accounts_publication_v1', created_at=datetime.now(timezone.utc).isoformat(),
        start='2024-01-02', end='2026-10-02', initial_cash=1000000, sleeve_initial_cash=500000,
        position_count=6, slots_per_sleeve=3, rebalance=False, cash_transfers=False, buy_fraction=.5, sell_fraction=.3,
        cases=cases, comparisons=comparisons, preregistration=spec_ref,
        modes={k: dict(commission=v[0], slippage=v[1]) for k, v in MODES.items()},
        expected_new_accounts=8, completed_new_accounts=8, reused_accounts=12,
        reports=reports, trial_accounting=dict(published_attempts=len(attempts), all_attempts=attempts, earlier_failures=failures),
        checked_source_count=len(sources.used), reporting_sources={str(p.relative_to(ROOT)): digest(p) for p in reporting},
        execution_audit=prior['execution_audit'], execution_audit_reused=True,
        execution_audit_scope='reused_original_account_diagnostics_not_new_sleeve_fill_verification',
        historical_period_already_researched=True, daily_range_proxy=True, intraday_odd_daily_proxy=True,
        intraday_odd_sequence_verified=False, complete_historical_universe=False, posthoc_data_exclusion=True,
        live_qualified=False, paper_enabled=False, actual_fill_verified=False, unseen_validation=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', nargs='+', type=Path, required=True)
    args = p.parse_args()
    path = ROOT/'artifacts/forward_simulation'/(IDENTIFIER+'.json')
    if path.exists(): raise ValueError('Split publication is immutable')
    from app.poc_split_accounts_ui import validate
    result = validate(build(args.reports))
    write(path, result)
    path.with_suffix('.sha256').write_text(digest(path)+'\n')
    print(json.dumps(dict(path=str(path), completed=result['completed_new_accounts'], checked_sources=result['checked_source_count'])))
