#!/usr/bin/env python3
"""Isolated chronology quarantine and observation-window risk experiment."""
from dataclasses import replace
from pathlib import Path
import argparse
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.config import load_config
from app.file_lock import file_lock
from scripts import research_cash_risk as parent
from scripts.research_exit_scenarios import read, write, sha, encoded, summarize, TrackedCorporateActions
from skills.cash_risk_replay import CashRiskReplay
from skills.observed_risk import ObservedRiskReplay
from skills.scenario_exit_replay import ExitSignals
from skills.execution_stress import StressBenchmark, audit_stress
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable
from skills.million_replay import UnresolvedAction

OUTPUT = ROOT / '.cache/observed-risk-20260913'
AUDIT = ROOT / 'artifacts/forward_simulation/price_chronology_20260913.json'
SPEC = ROOT / 'docs/prereg_observed_risk_20260913.md'
CODE = ['scripts/research_observed_risk.py', 'skills/observed_risk.py', 'scripts/audit_price_chronology.py']


class Budget(parent.Budget):
    def take(self, kind):
        if not self.enabled:
            raise ReplayDataUnavailable('Network forbidden in offline replay')
        limit = 100 if kind == 'official' else 20
        if self.counts[kind] >= limit:
            raise ReplayDataUnavailable('Research lifetime request budget exhausted: '+kind)
        self.counts[kind] += 1
        write(self.path, self.counts)


def identity():
    current = parent.source_inventory()
    if current != read(parent.OUTPUT/'identity.json'):
        raise ValueError('Sealed parent identity changed')
    audit = read(AUDIT)
    for name, value in audit['input_sha256'].items():
        if sha(ROOT/name) != value:
            raise ValueError('Chronology audit input changed: '+name)
    result = dict(current['files_sha256'])
    for path in [AUDIT, SPEC, *(ROOT/name for name in CODE), parent.OUTPUT/'manifest.json']:
        result[str(path.relative_to(ROOT))] = sha(path)
    result.update(audit['input_sha256'])
    return result


def quarantine(data):
    quotes, close = data.quotes.copy(), data.features.adjusted_close.copy()
    for row in read(AUDIT)['quarantine']:
        sid, day = row['stock_id'], pd.Timestamp(row['date'])
        quotes = quotes[~(quotes.stock_id.eq(sid) & quotes.date.eq(day))]
        if sid in close and day in close.index:
            close.at[day, sid] = float('nan')
    return replace(data, quotes=quotes, features=ExitSignals(close, data.days))


def case(data, config, cache, budget):
    token = load_config().finmind_token if budget.enabled else None
    feeds = ReplayMarketFeeds(cache/'execution-feeds', offline=not budget.enabled, token=token,
                              http_get=budget.http, finmind_fetch=budget.finmind)
    from scripts.research_chip import ADDITIONS
    overrides = read(parent.cash.OVERRIDES)['overrides'] | read(ADDITIONS)['overrides']
    corp = TrackedCorporateActions(data.events, cache/'dividends', token,
                                    offline=True, overrides=overrides)
    args = (data.quotes, data.companies, data.days, data.entries, feeds, corp)
    kwargs = dict(start=data.start, end=data.end, stress_mode=config['stress'])
    if config.get('benchmark'):
        engine = StressBenchmark(*args, **kwargs)
    else:
        engine_type = ObservedRiskReplay if config.get('observed') else CashRiskReplay
        engine = engine_type(*args, exit_signals=data.features, risk_mode=config.get('risk', 'none'), **kwargs)
    try:
        account = engine.run()
        checked = audit_stress(account)
    except (ReplayDataUnavailable, UnresolvedAction) as exc:
        return dict(completed=False, config=config, reason=str(exc))
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        return dict(completed=False, config=config, reason=str(exc))
    for trade in account['trades']:
        if trade['stock_id'] != '0050':
            if pd.Timestamp(trade['signal_date']) >= pd.Timestamp(trade['date']):
                raise ValueError('Trade signal is not strictly earlier than execution')
    risk = getattr(engine, 'risk', pd.DataFrame())
    risk = risk.loc[data.start:data.end] if not risk.empty else risk
    return dict(completed=True, config=config, summary=summarize(account), account=account,
        audit=checked, attribution=parent.attribution(account),
        unknown_risk_days=int(risk.cap.isna().sum()) if not risk.empty else 0,
        risk_decisions=getattr(engine, 'risk_decisions', []), live_qualified=False)


def run(output, prepare=False, replay=False):
    tick = time.monotonic()
    context = identity()
    path = output/'identity.json'
    if path.exists() and read(path) != context:
        raise ValueError('Research sources changed; choose a new output')
    if replay:
        for name, value in read(output/'manifest.json')['output_sha256'].items():
            if sha(output/name) != value:
                raise ValueError('Frozen output changed: '+name)
    else:
        write(path, context)
    cache = output/'inputs'
    if not cache.exists():
        if replay:
            raise ValueError('Offline replay requires sealed inputs')
        shutil.copytree(parent.OUTPUT/'inputs', cache)
    budget = Budget(output, prepare)
    original = parent.cash.load_inputs()
    clean = quarantine(original)
    configs = []
    for stress in ('control', 'combined'):
        configs += [(f'original_{stress}', dict(stress=stress)),
                    (f'quarantine_{stress}', dict(stress=stress, quarantine=True)),
                    (f'benchmark_{stress}', dict(stress=stress, benchmark=True))]
        configs += [(f'{risk}_{window}_{stress}', dict(stress=stress, quarantine=True,
                        risk=risk, observed=window == 'observed'))
                    for risk in ('trend60', 'shock') for window in ('market', 'observed')]
    results = {}
    for name, config in configs:
        target = output/'cases'/f'{name}.json'
        previous = read(target) if target.exists() else None
        if previous and previous['completed'] and not replay:
            result = previous
        else:
            result = case(clean if config.get('quarantine') else original, config, cache, budget)
            if replay and (previous is None or encoded(previous) != encoded(result)):
                raise ValueError('Offline replay differs: '+name)
            if not replay:
                write(target, result)
        if name.startswith('original_') and result['completed']:
            if encoded(result['account']) != encoded(read(parent.OUTPUT/'cases'/f'{config["stress"]}.json')['account']):
                raise ValueError('Original full account no longer reproduces')
        results[name] = {k:v for k,v in result.items() if k not in ('account','audit','risk_decisions','attribution')}
        print(name, 'COMPLETE' if result['completed'] else 'BLOCKED '+result['reason'], flush=True)
    if identity() != context:
        raise ValueError('Sources changed while running')
    result = dict(cases=results, elapsed_seconds=round(time.monotonic()-tick,3),
        all_completed=all(r['completed'] for r in results.values()), request_counts=budget.counts,
        live_qualified=False, unseen_validation=False)
    if replay:
        if not result['all_completed']:
            raise ValueError('Cannot certify blocked research')
        write(output/'offline-verification.json', dict(all_identical=True, elapsed_seconds=result['elapsed_seconds'],
            source_identity_sha256=sha(output/'identity.json'), manifest_sha256=sha(output/'manifest.json'), network_calls=0))
    else:
        write(output/'summary.json', result)
        inventory = {str(p.relative_to(output)):sha(p) for p in output.rglob('*')
                     if p.is_file() and p.suffix in ('.json','.parquet') and p.name not in
                     ('manifest.json','offline-verification.json')}
        write(output/'manifest.json', dict(output_sha256=inventory, live_qualified=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--offline-replay', action='store_true')
    args = parser.parse_args()
    if args.prepare and args.offline_replay:
        parser.error('Preparation and offline replay are separate modes')
    with file_lock(args.output/'run.lock', timeout=0):
        result = run(args.output, args.prepare, args.offline_replay)
    print('all_completed',result['all_completed'], 'seconds',result['elapsed_seconds'])
