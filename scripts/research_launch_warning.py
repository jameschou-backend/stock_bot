#!/usr/bin/env python3
"""Offline post-launch warning study with explicit entry-evidence preflight."""
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
from scripts.research_theme_catalyst import source_closure, CACHE
from scripts.audit_current_causality_20260925 import verify_hashes
from skills.first_bar import features, episodes, START, END
from skills.launch_warning import ARMS, EXCLUDED, observations, decisions, outcomes, summarize
from skills.sector_account_replay import run_case
from skills.verified_backtest_tool import offline_only
from skills.trial_registry import append_trial_registry

SPEC = ROOT/'docs/prereg_launch_warning_20260927.md'
PARENT = ROOT/'artifacts/forward_simulation/early_strength_20260927.json'
PUBLICATION = ROOT/'artifacts/forward_simulation/launch_warning_20260927.json'


def parent_inputs():
    if sha(PARENT) != PARENT.with_suffix('.sha256').read_text().strip(): raise ValueError('Parent index changed')
    pub = read(PARENT); report_path = ROOT/pub['report']['path']
    if sha(report_path) != pub['report']['sha256']: raise ValueError('Parent report changed')
    report = read(report_path); expected = {ROOT/p:h for p,h in report['source_sha256'].items()}
    expected[PARENT] = sha(PARENT); expected[report_path] = sha(report_path)
    for name in ('events.csv', 'entries.json'):
        expected[report_path.parent/name] = pub['reproducibility']['files_sha256'][name]
    verify_hashes(expected)
    return pd.read_csv(report_path.parent/'events.csv', dtype={'stock_id':str}), read(report_path.parent/'entries.json')['all_first'], expected


def causal_checks(events, frames, bars, companies, baseline_obs, baseline_decisions):
    close, _, raw, volume = frames; results = []
    for date in ('2023-12-29', '2024-12-31', '2026-03-31'):
        stop = pd.Timestamp(date)
        for mode in ('truncate', 'mutate'):
            changed = []
            for frame in (close, raw, volume):
                f = frame.loc[:stop].copy() if mode == 'truncate' else frame.copy()
                if mode == 'mutate': f.loc[f.index > stop] *= 2.7
                changed.append(f)
            q = {}
            for key, frame in bars.items():
                f = frame.loc[:stop].copy() if mode == 'truncate' else frame.copy()
                if mode == 'mutate': f.loc[f.index > stop] *= 2.7
                q[key] = f
            ev = episodes(features(*changed, q, companies)); ev['named_case'] = ev.stock_id.isin(EXCLUDED)
            past = lambda f: f[f.signal_date.le(date)].reset_index(drop=True)
            pd.testing.assert_frame_equal(past(events), past(ev))
            obs = observations(ev, *changed, q, companies)
            past_obs = lambda f: f[f.date.le(date)].reset_index(drop=True)
            pd.testing.assert_frame_equal(past_obs(baseline_obs), past_obs(obs))
            choice = decisions(ev, obs, changed[0])
            columns = ['event_id','stock_id','launch_date','horizon','arm','state','signal_date','signal_offset','reason']
            settled = lambda f: f[f.state.isin(['unknown','triggered']) & f.signal_date.le(date)][columns].reset_index(drop=True)
            pd.testing.assert_frame_equal(settled(baseline_decisions), settled(choice))
            for key in ('exit_date', 'delayed_exit_date'):
                scheduled = lambda f: f[f.state.eq('triggered') & f[key].le(date)][columns+[key]].reset_index(drop=True)
                pd.testing.assert_frame_equal(scheduled(baseline_decisions), scheduled(choice))
            results.append(dict(cutoff=date, mode=mode, passed=True, observation_rows=len(past_obs(obs))))
            print('causality', date, mode, flush=True)
    return results


def run(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.cache'): raise ValueError('Use a new immutable cache directory')
    tick = time.perf_counter()
    with file_lock(ROOT/'.cache/launch-warning.lock', timeout=0), offline_only():
        parent_events, entry_rows, expected = parent_inputs()
        frames, bars, _, data, inherited = inputs(); expected.update(inherited)
        closure, overrides = source_closure(CACHE); expected.update(closure)
        for path in (SPEC, Path(__file__), ROOT/'skills/launch_warning.py', ROOT/'tests/test_launch_warning.py',
                     ROOT/'skills/million_replay.py', ROOT/'scripts/research_first_bar.py', ROOT/'skills/first_bar.py',
                     ROOT/'skills/early_strength.py', ROOT/'skills/surge_anatomy.py'):
            expected[path] = sha(path)
        for arm in ARMS:
            append_trial_registry(dict(source='launch_warning', status='started', arm=arm,
                timestamp=datetime.now(timezone.utc).isoformat(), output=str(output.relative_to(ROOT)), preregistration_sha256=sha(SPEC)))
        close, quality, raw, volume = frames
        ev = episodes(features(close, raw, volume, bars, data.companies)); ev['named_case'] = ev.stock_id.isin(EXCLUDED)
        pd.testing.assert_frame_equal(ev, parent_events[ev.columns])
        obs = observations(ev, close, raw, volume, bars, data.companies)
        choice = decisions(ev, obs, close)
        print('events', len(ev), 'warning observations', len(obs), flush=True)
        checks = causal_checks(ev, frames, bars, data.companies, obs, choice)
        table = outcomes(ev, choice, close, quality)
        statistics = summarize(table); annual = summarize(table, annual=True)
        output.mkdir(parents=True)
        for name, frame in [('events', ev), ('observations', obs), ('decisions', choice), ('outcomes', table),
                            ('statistics', statistics), ('annual', annual)]:
            frame.to_csv(output/(name+'.csv'), index=False, encoding='utf-8-sig')
        preflight = {}
        selected = replace(data, entries_by_arm={'relative_strength':entry_rows})
        for stress in ('control', 'combined'):
            config = dict(arm='relative_strength', stress=stress, benchmark=False, board_only=False, position_count=5)
            result = run_case(selected, config, CACHE, overrides)
            path = output/('entry-preflight-'+stress+'.json'); write(path, clean(result))
            trades = len(result.get('account', result.get('partial_account', {})).get('trades', []))
            preflight[stress] = dict(completed=result['completed'], reason=result.get('reason'), trade_count=trades,
                blocked_before_any_fill=not result['completed'] and trades == 0,
                path=str(path.relative_to(ROOT)), sha256=sha(path))
            print('entry preflight', stress, preflight[stress]['reason'], 'trades', trades, flush=True)
        verify_hashes(expected)
        report = clean(dict(schema='launch_warning_v1', completed=True, start=START, end=END,
            events=len(ev), main_events=int((~ev.named_case).sum()), observations=len(obs),
            arms=list(ARMS), statistics=statistics.to_dict('records'), entry_preflight=preflight,
            all_new_exit_accounts_blocked_before_entry=all(p['blocked_before_any_fill'] for p in preflight.values()),
            full_account_returns_computed=False, strategy_net_return=None, live_qualified=False, adopted=False,
            unseen_validation=False, historical_membership_verified=False, fee_results_are_account_returns=False,
            finmind_requests=0, network_calls=0, database_writes=0, causality_checks=checks,
            source_sha256={str(p.relative_to(ROOT)):h for p,h in expected.items()},
            elapsed_seconds=round(time.perf_counter()-tick, 3),
            limitations=['Known history and present-day industry snapshot; no untouched or full historical theme membership.',
                'Industry peers exclude the stock; observed price times volume is activity, not net money flow.',
                'Same reference entry and shared endpoint, with cash after an early exit; no reinvestment or overlapping capital.',
                'Price exits are next-session closing references, not certified historical fills.',
                'Proportional fee sensitivity omits rounding, minimum fees, share integers and corporate cash timing.',
                'Unknown necessary features do not silently become hold signals; group coverage can change the paired subset.',
                'Only early failure in ten sessions is targeted; normal later pullbacks are not retrospectively relabeled.',
                'Entry preflight does not claim six new account policies were executed.']))
        write(output/'report.json', report)
        write(output/'manifest.json', dict(schema='launch_warning_manifest_v1', source_sha256=report['source_sha256'],
            files_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
        for arm in ARMS:
            append_trial_registry(dict(source='launch_warning', status='completed_event_study', arm=arm,
                timestamp=datetime.now(timezone.utc).isoformat(), output=str(output.relative_to(ROOT)), preregistration_sha256=sha(SPEC),
                full_account_returns_computed=False))
    print(dict(events=report['events'], elapsed_seconds=report['elapsed_seconds']))
    return report


def publish(first, second):
    folders = [Path(p).resolve() for p in (first, second)]
    if folders[0] == folders[1] or any(not p.is_relative_to(ROOT/'.cache') for p in folders): raise ValueError('Two distinct cache runs required')
    reports = []; manifests = []
    for folder in folders:
        r = read(folder/'report.json'); m = read(folder/'manifest.json')
        if (r['schema'] != 'launch_warning_v1' or m['schema'] != 'launch_warning_manifest_v1'
                or m['source_sha256'] != r['source_sha256'] or m['files_sha256'].get('report.json') != sha(folder/'report.json')
                or r['strategy_net_return'] is not None or r['completed'] is not True
                or any(r[k] is not False for k in ('live_qualified','adopted','unseen_validation','full_account_returns_computed','fee_results_are_account_returns'))):
            raise ValueError('Unsupported or unbound report')
        verify_hashes({ROOT/p:h for p,h in m['source_sha256'].items()})
        for name, digest in m['files_sha256'].items():
            if Path(name).name != name or sha(folder/name) != digest: raise ValueError('Changed study output')
        normalized = copy.deepcopy(r); normalized.pop('elapsed_seconds')
        for p in normalized['entry_preflight'].values(): p['path'] = Path(p['path']).name
        reports.append(normalized); manifests.append(m)
    files = lambda m: {k:v for k,v in m['files_sha256'].items() if k != 'report.json'}
    if reports[0] != reports[1] or files(manifests[0]) != files(manifests[1]): raise ValueError('Independent studies differ')
    write(PUBLICATION, dict(schema='launch_warning_publication_v1', live_qualified=False, adopted=False,
        report=dict(path=str((folders[0]/'report.json').relative_to(ROOT)), sha256=sha(folders[0]/'report.json')),
        reproducibility=dict(passed=True, files_sha256=files(manifests[0]),
            runs=[dict(path=str((p/'manifest.json').relative_to(ROOT)), sha256=sha(p/'manifest.json')) for p in folders])))
    PUBLICATION.with_suffix('.sha256').write_text(sha(PUBLICATION)+'\n')
    print('Published identical post-launch warning studies')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path); group.add_argument('--publish', nargs=2, type=Path)
    args = p.parse_args()
    if args.publish: publish(*args.publish)
    else: run(args.output)
