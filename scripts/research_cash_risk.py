#!/usr/bin/env python3
"""Bounded cash-account research; immutable parents and independent case files."""
import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

from app.config import load_config
from app.file_lock import file_lock
from app.finmind import fetch_dataset
from scripts import research_cash_allocation as cash
from scripts.research_exit_scenarios import read, write, sha, encoded, summarize, TrackedCorporateActions
from skills.cash_risk_replay import CashRiskReplay
from skills.execution_stress import MODES, StressBenchmark, audit_stress
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable
from skills.million_replay import UnresolvedAction

OUTPUT = ROOT / '.cache/cash-risk-20260913'
SPEC = ROOT / 'docs/prereg_cash_risk_20260913.md'
CODE = ('scripts/research_cash_risk.py', 'skills/cash_risk_replay.py', 'skills/execution_stress.py')


def source_inventory():
    """A new research lineage explicitly records known downloader code drift."""
    inventory = dict(read(cash.OUTPUT / 'manifest.json')['verification_files_sha256'])
    inventory.update(read(ROOT / '.cache/execution-stress/manifest.json')['files_sha256'])
    changes = []
    for name, expected in inventory.items():
        actual = sha(ROOT / name)
        if actual != expected:
            if name != 'app/finmind.py':
                raise ValueError('Unexpected parent source drift: ' + name)
            old = subprocess.run(['git', 'show', 'ead1b10ee3432d66f36e4515aef956a10dd0fc64:app/finmind.py'],
                                 cwd=ROOT, check=True, capture_output=True).stdout
            import hashlib
            if hashlib.sha256(old).hexdigest() != expected:
                raise ValueError('Historical downloader code cannot be authenticated')
            changes.append(dict(path=name, sealed_sha256=expected, current_sha256=actual,
                treatment='Explicit new lineage; original verifier remains blocked; require offline control equality'))
    current = {name: sha(ROOT / name) for name in inventory}
    current.update({name: sha(ROOT / name) for name in CODE})
    current[str(SPEC.relative_to(ROOT))] = sha(SPEC)
    return dict(files_sha256=current, known_parent_code_drift=changes)


def prepare_cache(output):
    cache = output / 'inputs'
    if cache.exists():
        return cache
    cache.mkdir(parents=True)
    for name in ('execution-feeds', 'dividends'):
        shutil.copytree(ROOT / '.cache/execution-stress-inputs' / name, cache / name)
    # Both seed feed sets are immutable; the stress cache is a strict superset.
    small = read(cash.INPUT / 'execution-feeds/index.json')
    big = read(cache / 'execution-feeds/index.json')
    for section in ('entries', 'files_sha256'):
        if any(big[section].get(k) != v for k, v in small[section].items()):
            raise ValueError('Stress feed seed does not contain the cash evidence')
    for path in (cash.INPUT / 'dividends').glob('*.parquet'):
        target = cache / 'dividends' / path.name
        if target.exists() and sha(target) != sha(path):
            raise ValueError('Dividend seed conflict: ' + path.name)
        if not target.exists():
            shutil.copyfile(path, target)
    return cache


class Budget:
    def __init__(self, output, enabled):
        self.path = output / 'request-budget.json'
        self.enabled = enabled
        self.counts = read(self.path) if self.path.exists() else dict(official=0, finmind=0)

    def take(self, kind):
        if not self.enabled:
            raise ReplayDataUnavailable('Network forbidden in offline replay')
        limit = 300 if kind == 'official' else 50
        if self.counts[kind] >= limit:
            raise ReplayDataUnavailable('Research lifetime request budget exhausted: ' + kind)
        self.counts[kind] += 1
        write(self.path, self.counts)

    def http(self, *args, **kwargs):
        self.take('official')
        return requests.get(*args, **kwargs)

    def finmind(self, *args, **kwargs):
        self.take('finmind')
        kwargs['max_retries'] = 0
        return fetch_dataset(*args, **kwargs)


def attribution(account):
    values = defaultdict(float)
    final = account['daily'][-1]['date']
    prices = {}
    for row in account['cash_ledger']:
        if row.get('stock_id'):
            values[row['stock_id']] += row['cash_change']
    for row in account['holdings']:
        if row['date'] == final:
            values[row['stock_id']] += row['market_value']
            prices[row['stock_id']] = row['price']
    for row in account['receivables']:
        sid = row['stock_id']
        if row['kind'] == 'cash':
            values[sid] += row['amount']
        else:
            if row.get('qty', 0) and sid not in prices:
                raise ValueError('Missing final stock entitlement valuation')
            values[sid] += row.get('qty', 0)*prices.get(sid, 0)
            values[sid] += row.get('fraction', 0)*(row.get('fractional_cash_per_share') or 0)
    if abs(sum(values.values()) - (account['daily'][-1]['nav']-account['settings']['initial_cash'])) > .1:
        raise ValueError('Stock profit attribution does not reconcile')
    return sorted((dict(stock_id=sid, net_pnl=pnl) for sid, pnl in values.items()),
                  key=lambda x: -x['net_pnl'])


def case(data, config, cache, budget):
    token = load_config().finmind_token if budget.enabled else None
    feeds = ReplayMarketFeeds(cache / 'execution-feeds', offline=not budget.enabled,
        token=token, http_get=budget.http, finmind_fetch=budget.finmind)
    overrides = read(cash.OVERRIDES)['overrides']
    # Additional terms used by the already sealed execution-stress study.
    from scripts.research_chip import ADDITIONS
    overrides = overrides | read(ADDITIONS)['overrides']
    corp = TrackedCorporateActions(data.events, cache / 'dividends', token,
                                    offline=True, overrides=overrides)
    # Dividend cache is deliberately offline: unknown corporate terms block a case.
    start = config.get('start', data.start)
    entries = [e for e in data.entries if e['members'][0] not in config.get('exclude', [])
               and e['signal_date'] >= start]
    kwargs = dict(start=start, end=data.end, stress_mode=config.get('stress', 'control'))
    args = (data.quotes, data.companies, data.days, entries, feeds, corp)
    if config.get('benchmark'):
        engine = StressBenchmark(*args, **kwargs)
    else:
        engine = CashRiskReplay(*args, exit_signals=data.features,
                                risk_mode=config.get('risk', 'none'), **kwargs)
        if 'seed' in config:
            rng = random.Random(config['seed'])
            for day in sorted(engine.events):
                rng.shuffle(engine.events[day])
    try:
        account = engine.run()
        checked = audit_stress(account)
    except (ReplayDataUnavailable, UnresolvedAction) as exc:
        return dict(completed=False, config=config, reason=str(exc), live_qualified=False)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        return dict(completed=False, config=config, reason=str(exc), live_qualified=False)
    if config == {'stress': 'control'}:
        if encoded(account) != encoded(read(cash.OUTPUT / 'cases/cash.json')['account']):
            raise ValueError('New neutral cash account differs from sealed original')
    for trade in account['trades']:
        if trade['stock_id'] != '0050':
            lag = engine.positions[pd.Timestamp(trade['date'])] - engine.positions[pd.Timestamp(trade['signal_date'])]
            if lag < 1:
                raise ValueError('Trade uses same-day/future signal')
    stats = summarize(account)
    stats['average_stock_weight'] = sum(d['market_value']/d['nav'] for d in account['daily']) / len(account['daily'])
    return dict(completed=True, config=config, account=account, summary=stats, audit=checked,
                risk_decisions=getattr(engine, 'risk_decisions', []),
                attribution=attribution(account), live_qualified=False)


def run(output, prepare=False, replay=False):
    tick = time.monotonic()
    context = source_inventory()
    if replay:
        meta = read(output / 'manifest.json')
        for name, value in meta['output_sha256'].items():
            if sha(output / name) != value:
                raise ValueError('Research output changed: ' + name)
    identity = output / 'identity.json'
    if identity.exists() and read(identity) != context:
        raise ValueError('Research code/source changed; choose a new output path')
    write(identity, context)
    cache = prepare_cache(output)
    budget = Budget(output, prepare)
    data = cash.load_inputs()
    configs = [(m, dict(stress=m)) for m in MODES]
    configs += [(m+'_benchmark', dict(stress=m, benchmark=True)) for m in MODES]
    configs += [(f'{risk}_{stress}', dict(risk=risk, stress=stress))
                for risk in ('trend60', 'shock') for stress in ('control', 'combined')]
    configs += [(f'random_{seed}', dict(seed=seed)) for seed in (11, 29, 47)]
    for year in (2023, 2024, 2025):
        start = str(data.days[data.days.year == year][0].date())
        configs += [(f'start_{year}', dict(start=start)),
                    (f'start_{year}_benchmark', dict(start=start, benchmark=True))]
    baseline = read(cash.OUTPUT / 'cases/cash.json')['account']
    leaders = [x['stock_id'] for x in attribution(baseline)[:2]]
    configs += [('exclude_top1', dict(exclude=leaders[:1])), ('exclude_top2', dict(exclude=leaders))]
    results = {}
    for name, config in configs:
        path = output / 'cases' / (name+'.json')
        previous = read(path) if path.exists() else None
        if previous and previous['completed'] and not replay:
            result = previous
        else:
            result = case(data, config, cache, budget)
            if replay and previous is not None and encoded(result) != encoded(previous):
                raise ValueError('Offline case differs: '+name)
            if not replay:
                write(path, result)
        results[name] = {k: v for k, v in result.items() if k not in ('account', 'risk_decisions', 'audit')}
        print(name, 'COMPLETE' if result['completed'] else 'BLOCKED '+result['reason'], flush=True)
    if source_inventory() != context:
        raise ValueError('Parent sources changed during research')
    summary = dict(cases=results, elapsed_seconds=round(time.monotonic()-tick, 3),
        request_budget_counts=budget.counts, all_completed=all(x['completed'] for x in results.values()),
        neutral_matches_sealed=results['control']['completed'], live_qualified=False,
        unseen_validation=False, known_parent_code_drift=context['known_parent_code_drift'])
    if not replay:
        write(output / 'summary.json', summary)
        inventory = {str(p.relative_to(output)): sha(p) for p in output.rglob('*')
                     if p.is_file() and p.suffix in ('.json', '.parquet') and p.name != 'manifest.json'}
        write(output / 'manifest.json', dict(output_sha256=inventory, live_qualified=False))
    else:
        print('All recorded cases reproduced offline', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--offline-replay', action='store_true')
    args = parser.parse_args()
    if args.prepare and args.offline_replay:
        parser.error('Preparation and offline replay are separate modes')
    with file_lock(args.output / 'run.lock', timeout=0):
        run(args.output, args.prepare, args.offline_replay)
