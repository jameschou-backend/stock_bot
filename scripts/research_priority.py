#!/usr/bin/env python3
"""Frozen-sample ranking contrasts and explicitly post-hoc concentration stress."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from app.file_lock import file_lock
from scripts import research_five_axis as five
from scripts.research_exit_scenarios import read, write, sha, encoded, summarize, TrackedCorporateActions
from scripts.research_cash_risk import attribution
from skills.priority_replay import PriorityReplay, group_strength_scores
from skills.scenario_exit_replay import ExitSignals
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable
from skills.execution_stress import audit_stress
from skills.million_replay import UnresolvedAction

SOURCE = ROOT / '.cache/five-axis-20260913'
OUTPUT = ROOT / '.cache/priority-20260924'
SPEC = ROOT / 'docs/prereg_priority_20260924.md'


def inventory():
    files = five.identity(SOURCE)
    for name, digest in read(SOURCE/'execution-manifest.json')['files_sha256'].items():
        path = SOURCE/name
        if sha(path) != digest:
            raise ValueError('Parent execution evidence changed: '+name)
        files[str(path.relative_to(ROOT))] = digest
    for path in (SPEC, Path(__file__), ROOT/'skills/priority_replay.py',
                 ROOT/'docs/data_completion_summary_20260914.json'):
        files[str(path.relative_to(ROOT))] = sha(path)
    completion = read(ROOT/'docs/data_completion_summary_20260914.json')
    for ref in completion['sources'].values():
        if sha(ROOT/ref['path']) != ref['sha256']:
            raise ValueError('Data audit changed: '+ref['path'])
        files[ref['path']] = ref['sha256']
    return files


def inputs():
    original = five.parent.cash.load_inputs()
    entries = read(SOURCE/'rebuild/signals.json')['entries']
    close = pd.read_parquet(SOURCE/'rebuild/close-official.parquet').set_index('date')
    close.index = pd.to_datetime(close.index)
    # Explicit market-calendar alignment: a missing whole day stays missing.
    close = close.reindex(pd.DatetimeIndex(original.days))
    pool = sorted({'0050'} | {e['members'][0] for e in entries})
    refs = read(five.parent.cash.INPUT/'manifest.json')['references']
    quotes = pd.read_parquet(ROOT/refs['quotes']['path'], filters=[('stock_id','in',pool)])
    quotes['date'] = pd.to_datetime(quotes.date)
    for row in read(five.AUDIT)['quarantine']:
        quotes = quotes[~(quotes.stock_id.eq(row['stock_id']) & quotes.date.eq(pd.Timestamp(row['date'])))]
    data = replace(original, quotes=quotes, entries=entries, features=ExitSignals(close[pool], original.days))
    return data, group_strength_scores(close, entries)


def run_case(data, scores, config):
    cache = SOURCE/'execution-inputs'
    feeds = ReplayMarketFeeds(cache/'execution-feeds', offline=True)
    from scripts.research_chip import ADDITIONS
    overrides = read(five.parent.cash.OVERRIDES)['overrides'] | read(ADDITIONS)['overrides']
    corp = TrackedCorporateActions(data.events, cache/'dividends', None, offline=True, overrides=overrides)
    entries = [e for e in data.entries if e['members'][0] not in config.get('exclude', [])]
    engine = PriorityReplay(data.quotes, data.companies, data.days, entries, feeds, corp,
        start=data.start, end=data.end, stress_mode=config['stress'], priority=config['priority'],
        scores=scores, exit_signals=data.features,
        action_dates=list(zip(data.events.stock_id, data.events.event_date)))
    try:
        account = engine.run()
        audit = audit_stress(account)
    except (ReplayDataUnavailable, UnresolvedAction) as exc:
        return dict(completed=False, config=config, reason=str(exc), live_qualified=False)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        return dict(completed=False, config=config, reason=str(exc), live_qualified=False)
    for trade in account['trades']:
        if trade['stock_id'] != '0050' and pd.Timestamp(trade['signal_date']) >= pd.Timestamp(trade['date']):
            raise ValueError('Non-causal trade')
    return dict(completed=True, config=config, account=account, audit=audit, summary=summarize(account),
                priority_decisions=engine.priority_decisions, attribution=attribution(account),
                live_qualified=False, unseen_validation=False)


def rolling_comparison(account, benchmark):
    def nav(a):
        return pd.Series({r['date']: r['nav'] for r in a['daily']}, dtype=float).sort_index()
    a, b = nav(account), nav(benchmark)
    if not a.index.equals(b.index):
        raise ValueError('Benchmark calendar differs')
    excess = (a/a.shift(252) - b/b.shift(252)).dropna()
    return dict(overlapping_windows=len(excess), independent_samples=False,
                outperform_fraction=float((excess > 0).mean()) if len(excess) else None,
                median_excess=float(excess.median()) if len(excess) else None,
                worst_excess=float(excess.min()) if len(excess) else None)


def run(output, replay=False):
    tick = time.monotonic()
    source = inventory()
    identity = output/'identity.json'
    if identity.exists() and read(identity) != source:
        raise ValueError('Source/code changed; choose a new output')
    if replay:
        for name, digest in read(output/'manifest.json')['files_sha256'].items():
            if sha(output/name) != digest:
                raise ValueError('Research artifact changed: '+name)
    else:
        if output.exists():
            raise ValueError('Choose a new immutable output')
        write(identity, source)
    data, scores = inputs()
    old = read(SOURCE/'cases/capacity_combined.json')['account']
    leaders = [r['stock_id'] for r in attribution(old) if r['stock_id'] != '0050' and r['net_pnl'] > 0][:3]
    configs = []
    for stress in ('control', 'combined'):
        for priority in ('control', 'capacity', 'group_strength'):
            configs.append((priority+'_'+stress, dict(priority=priority, stress=stress)))
        for n in (1, 3):
            for priority in ('control', 'capacity'):
                configs.append((f'{priority}_{stress}_omit{n}', dict(priority=priority, stress=stress, exclude=leaders[:n])))
    rows = {}
    for name, config in configs:
        result = run_case(data, scores, config)
        if result['completed']:
            if config['priority'] in ('control', 'capacity') and 'exclude' not in config:
                parent = read(SOURCE/'cases'/f"{config['priority']}_{config['stress']}.json")
                if encoded(parent['account']) != encoded(result['account']):
                    raise ValueError('Neutral control failed reproduction: '+name)
            benchmark = read(SOURCE/'cases'/f"benchmark_{config['stress']}.json")
            result['benchmark'] = benchmark['summary']
            result['rolling252'] = rolling_comparison(result['account'], benchmark['account'])
        target = output/'cases'/f'{name}.json'
        if replay:
            if read(target) != result:
                raise ValueError('Offline replay differs: '+name)
        else:
            write(target, result)
        rows[name] = {k:v for k,v in result.items() if k not in ('account','audit','priority_decisions')}
        print(name, 'COMPLETE' if result['completed'] else 'BLOCKED '+result['reason'], flush=True)
    if inventory() != source:
        raise ValueError('Inputs changed during research')
    report = dict(cases=rows, omitted_stocks_posthoc=leaders, all_completed=all(r['completed'] for r in rows.values()),
                  candidate_count=len(data.entries), missing_group_scores=sum(r['score'] is None for r in scores.values()),
                  live_qualified=False, unseen_validation=False, network_requests=0, database_writes=0,
                  limitations=['Fixed historical universe remains incomplete', 'Historical odd-lot execution is not verified',
                               'Missing adjustment/announcement evidence remains unresolved', 'Omissions use future winners only for diagnosis'],
                  elapsed_seconds=round(time.monotonic()-tick, 3))
    if replay:
        write(output/'verification.json', dict(all_cases_identical=True, all_completed=report['all_completed'],
              elapsed_seconds=report['elapsed_seconds'], network_requests=0, manifest_sha256=sha(output/'manifest.json')))
    else:
        write(output/'summary.json', report)
        write(output/'manifest.json', dict(files_sha256={str(p.relative_to(output)):sha(p)
            for p in output.rglob('*.json')}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--offline-replay', action='store_true')
    args = parser.parse_args()
    with file_lock(ROOT/'.cache/priority-research.lock', timeout=0):
        report = run(args.output, args.offline_replay)
    print('completed', sum(c['completed'] for c in report['cases'].values()), '/', len(report['cases']))
