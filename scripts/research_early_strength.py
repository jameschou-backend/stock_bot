#!/usr/bin/env python3
"""Offline six-arm study: prior strength transitions and post-launch confirmation."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from app.file_lock import file_lock
from scripts.research_first_bar import inputs
from scripts.research_exit_scenarios import read, write, sha
from scripts.research_surge_anatomy import clean
from scripts.research_theme_catalyst import source_closure, annotate_annual_periods, CACHE
from scripts.audit_current_causality_20260925 import verify_hashes
from skills.first_bar import features, episodes, START, END
from skills.early_strength import EXCLUDED, event_features, decisions, orders, outcomes, summarize
from skills.sector_account_replay import run_case
from skills.verified_backtest_tool import offline_only
from skills.trial_registry import append_trial_registry

SPEC = ROOT / 'docs/prereg_early_strength_20260927.md'
PUBLICATION = ROOT / 'artifacts/forward_simulation/early_strength_20260927.json'


def causal_checks(frames, ohlc, companies, baseline_events, baseline_decisions, baseline_orders):
    close, _, raw, volume = frames
    results = []
    for date in ('2023-12-29', '2024-12-31', '2026-03-31'):
        cutoff = pd.Timestamp(date)
        for mode in ('truncate', 'mutate'):
            changed = []
            for frame in (close, raw, volume):
                f = frame.loc[:cutoff].copy() if mode == 'truncate' else frame.copy()
                if mode == 'mutate': f.loc[f.index > cutoff] *= 1.8
                changed.append(f)
            bars = {}
            for k, frame in ohlc.items():
                f = frame.loc[:cutoff].copy() if mode == 'truncate' else frame.copy()
                if mode == 'mutate': f.loc[f.index > cutoff] *= 1.8
                bars[k] = f
            computed = features(*changed, bars, companies)
            ev = event_features(episodes(computed), changed[0])
            past = lambda t: t[t.signal_date.le(date)].reset_index(drop=True)
            pd.testing.assert_frame_equal(past(baseline_events), past(ev))
            choice = decisions(ev, computed, *changed, bars)
            # Only final decisions known before the cutoff; a scheduled order
            # whose next session was removed is compared below by entry date.
            settled = lambda t: t[t.decision_date.notna() & t.decision_date.lt(date)].reset_index(drop=True)
            pd.testing.assert_frame_equal(settled(baseline_decisions), settled(choice))
            actual_orders = orders(choice, computed)
            for arm, records in baseline_orders.items():
                before = lambda values: [e for e in values if e['entry_date'] <= date]
                if before(records) != before(actual_orders[arm]):
                    raise ValueError('Future information changed earlier entry orders')
            results.append(dict(cutoff=date, mode=mode, passed=True, events=len(past(ev))))
            print('causality', date, mode, flush=True)
    return results


def run(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT / '.cache'):
        raise ValueError('Use a new immutable directory below .cache')
    tick = time.perf_counter()
    with file_lock(ROOT / '.cache/early-strength.lock', timeout=0), offline_only():
        frames, ohlc, _, data, expected = inputs()
        close, quality, raw, volume = frames
        closure, overrides = source_closure(CACHE); expected.update(closure)
        for path in (SPEC, Path(__file__), ROOT/'skills/early_strength.py', ROOT/'tests/test_early_strength.py',
                ROOT/'scripts/research_first_bar.py', ROOT/'skills/first_bar.py', ROOT/'skills/surge_anatomy.py',
                ROOT/'scripts/research_theme_catalyst.py', ROOT/'scripts/research_surge_anatomy.py'):
            expected[path] = sha(path)
        computed = features(close, raw, volume, ohlc, data.companies)
        ev = event_features(episodes(computed), close)
        choice = decisions(ev, computed, close, raw, volume, ohlc)
        entries = orders(choice, computed)
        print('events', len(ev), 'signals', {k:len(v) for k,v in entries.items()}, flush=True)
        checks = causal_checks(frames, ohlc, data.companies, ev, choice, entries)
        print('calculating separate outcome labels', flush=True)
        table = outcomes(ev, choice, close, quality)
        statistics = summarize(table); annual = summarize(table, annual=True)
        output.mkdir(parents=True)
        for name, frame in [('events', ev), ('decisions', choice), ('outcomes', table),
                ('statistics', statistics), ('annual', annual)]:
            frame.to_csv(output / (name+'.csv'), index=False, encoding='utf-8-sig')
        write(output/'entries.json', entries)
        cases = {}; jobs = []
        for board in (False, True):
            channel = 'board' if board else 'mixed'
            for stress in ('control', 'combined'):
                jobs.append((f'benchmark_{channel}_{stress}', data,
                    dict(arm='benchmark', stress=stress, benchmark=True, board_only=board, position_count=0)))
                for arm, rows in entries.items():
                    jobs.append((f'{arm}_{channel}_{stress}', replace(data, entries_by_arm={'relative_strength': rows}),
                        dict(arm='relative_strength', stress=stress, benchmark=False, board_only=board, position_count=5)))
        for name, selected, config in jobs:
            trial = dict(timestamp=datetime.now(timezone.utc).isoformat(), source='early_strength', case=name,
                output=str(output.relative_to(ROOT)), preregistration_sha256=sha(SPEC))
            append_trial_registry(dict(trial, status='started'))
            result = run_case(selected, config, CACHE, overrides); result['strategy_case'] = name
            row = {k:v for k,v in result.items() if k in ('completed', 'summary', 'reason', 'candidate_count', 'audit', 'execution')}
            if result['completed']:
                annotate_annual_periods(result['summary'])
                row['summary'] = result['summary']
                row['average_cash_fraction'] = sum(d['cash']/d['nav'] for d in result['account']['daily'])/len(result['account']['daily'])
                for key in ('trades', 'daily', 'cash_ledger'):
                    pd.DataFrame(result['account'][key]).to_csv(output/(name+'-'+key+'.csv'), index=False, encoding='utf-8-sig')
            write(output/(name+'.json'), clean(result))
            row['result'] = dict(path=str((output/(name+'.json')).relative_to(ROOT)), sha256=sha(output/(name+'.json')))
            cases[name] = row
            append_trial_registry(dict(trial, status='completed' if result['completed'] else 'blocked', reason=result.get('reason')))
            print(name, 'completed' if result['completed'] else result['reason'], flush=True)
        for name, row in cases.items():
            if name.startswith('benchmark'): continue
            bm = cases['benchmark_'+'_'.join(name.split('_')[-2:])]
            row['benchmark_return'] = bm['summary']['total_return'] if bm['completed'] else None
            row['excess_return'] = row['summary']['total_return']-row['benchmark_return'] if row['completed'] and bm['completed'] else None
        verify_hashes(expected)
        report = clean(dict(schema='early_strength_v1', completed=True, live_qualified=False, adopted=False,
            unseen_validation=False, historical_first_publication_verified=False, historical_membership_verified=False,
            start=START, end=END, events=len(ev), main_events=int((~ev.named_case).sum()),
            excluded_named_stocks=sorted(EXCLUDED), signal_counts={k:len(v) for k,v in entries.items()},
            statistics=statistics.to_dict('records'), cases=cases, all_accounts_completed=all(c['completed'] for c in cases.values()),
            price_statistics_are_account_returns=False, causality_checks=checks,
            source_sha256={str(p.relative_to(ROOT)):h for p,h in expected.items()},
            elapsed_seconds=round(time.perf_counter()-tick, 3), finmind_requests=0, network_calls=0, database_writes=0,
            limitations=['Previously researched history and current-company cohort, not unseen or full historical membership.',
                'Relative strength is measured before the launch; its thresholds are hypotheses, not optimal values.',
                'Confirmation necessarily removes some early failures but can miss surges or pay a higher entry.',
                'Opportunity means include unselected or untriggered cash; they are not account performance.',
                'Shared endpoint shortens holding time for delayed entries; matched entry benchmarks are separate.',
                'Missing or immature observations remain unknown; no assumed fills or silent data substitution.',
                'No new news, institutional, chip-concentration or broker filter was added.']))
        write(output/'report.json', report)
        write(output/'manifest.json', dict(schema='early_strength_manifest_v1', source_sha256=report['source_sha256'],
            files_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(dict(events=report['events'], elapsed_seconds=report['elapsed_seconds'], all_accounts_completed=report['all_accounts_completed']))
    return report


def publish(first, second):
    folders = [Path(first).resolve(), Path(second).resolve()]
    if folders[0] == folders[1] or any(not p.is_relative_to(ROOT/'.cache') for p in folders):
        raise ValueError('Two distinct immutable cache outputs required')
    normalized = []; manifests = []
    for folder in folders:
        report = read(folder/'report.json'); manifest = read(folder/'manifest.json')
        if (report['schema'] != 'early_strength_v1' or manifest['schema'] != 'early_strength_manifest_v1'
                or manifest['source_sha256'] != report['source_sha256'] or report['completed'] is not True
                or any(report.get(k) is not False for k in ('live_qualified', 'adopted', 'unseen_validation', 'price_statistics_are_account_returns'))):
            raise ValueError('Unbound or unsupported study report')
        if manifest['files_sha256'].get('report.json') != sha(folder/'report.json'):
            raise ValueError('Report missing from manifest or changed')
        verify_hashes({ROOT/p:h for p,h in manifest['source_sha256'].items()})
        for name, digest in manifest['files_sha256'].items():
            if Path(name).name != name or sha(folder/name) != digest: raise ValueError('Changed study output')
        r = copy.deepcopy(report); r.pop('elapsed_seconds')
        for case in r['cases'].values(): case['result']['path'] = Path(case['result']['path']).name
        normalized.append(r); manifests.append(manifest)
    files = lambda m: {k:v for k,v in m['files_sha256'].items() if k != 'report.json'}
    if normalized[0] != normalized[1] or files(manifests[0]) != files(manifests[1]):
        raise ValueError('Independent early-strength runs differ')
    write(PUBLICATION, dict(schema='early_strength_publication_v1', live_qualified=False, adopted=False,
        report=dict(path=str((folders[0]/'report.json').relative_to(ROOT)), sha256=sha(folders[0]/'report.json')),
        reproducibility=dict(passed=True, files_sha256=files(manifests[0]),
            runs=[dict(path=str((p/'manifest.json').relative_to(ROOT)), sha256=sha(p/'manifest.json')) for p in folders])))
    PUBLICATION.with_suffix('.sha256').write_text(sha(PUBLICATION)+'\n')
    print('Published identical six-arm results')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--output', type=Path); g.add_argument('--publish', nargs=2, type=Path)
    args = p.parse_args()
    if args.publish: publish(*args.publish)
    else: run(args.output)
