#!/usr/bin/env python3
"""Reproduce the winning daily model, then isolate precommitted cash/slots."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import shutil
import sys
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import research_five_axis as parent
from scripts.research_exit_scenarios import read, write, sha, encoded, summarize, TrackedCorporateActions
from scripts.research_intraday_limit import OUTPUT as TICK_OUTPUT
from skills.reservation_replay import ReservedCapacityReplay, ReservedBenchmark, audit_reservations
from skills.execution_stress import audit_stress
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable
from skills.million_replay import UnresolvedAction
from app.file_lock import file_lock
from app.finmind import fetch_dataset


class PreparationBudget:
    def __init__(self, output):
        self.path = output/'preparation-budget.json'
        self.calls = 0

    def take(self, kind):
        with file_lock(self.path.with_suffix('.lock')):
            counts = read(self.path) if self.path.exists() else dict(finmind=0, official=0)
            if counts[kind] >= 100:
                raise ReplayDataUnavailable('Bridge preparation lifetime request ceiling')
            counts[kind] += 1
            write(self.path, counts)
        self.calls += 1

    def http(self, *args, **kwargs):
        self.take('official')
        return requests.get(*args, **kwargs)

    def finmind(self, *args, **kwargs):
        self.take('finmind')
        kwargs['max_retries'] = 0
        return fetch_dataset(*args, **kwargs)

OUTPUT = ROOT / '.cache/reservation-bridge-20260914-verified'
CODE = ['skills/reservation_replay.py', 'scripts/research_reservation_bridge.py',
        'docs/prereg_reservation_bridge_20260914.md']


def inspect_ledger(account, markets):
    trades = defaultdict(list)
    for row in account['trades']:
        trades[row['date']].append(row)
    cash = account['settings']['initial_cash']
    dependencies = []
    for day in account['daily']:
        rows = trades[day['date']]
        buys = [r for r in rows if r['side'] == 'buy']
        spent = sum(-r['cash_change'] for r in buys)
        members = {c['stock_id'] for c in account['cohorts']
                   if c['entry_date'] < day['date'] and (c['exit_date'] is None or c['exit_date'] >= day['date'])}
        new_members = {r['stock_id'] for r in buys if r['stock_id'] != '0050'}-members
        needs_slot = len(members)+len(new_members) > account['settings']['slots']
        if spent > cash+.01 or needs_slot:
            dependencies.append(dict(date=day['date'], previous_cash=cash, buy_outflow=round(spent, 2),
                later_cash_required=round(max(0, spent-cash), 2), needs_same_day_slot_release=needs_slot,
                buy_stocks=sorted({r['stock_id'] for r in buys}),
                sell_stocks=sorted({r['stock_id'] for r in rows if r['side']=='sell'})))
        cash = day['cash']
    requests = sorted({(r['date'], r['stock_id'], markets[r['stock_id']], r['channel'])
                       for r in account['orders'] if r['requested_qty'] and r['channel'] in ('board','odd')})
    odd = [r for r in account['trades'] if r['channel']=='odd']
    return dict(cash_dependent_days=sum(d['later_cash_required'] > .01 for d in dependencies),
        slot_dependent_days=sum(d['needs_same_day_slot_release'] for d in dependencies),
        dependencies=dependencies, trades_by_channel=dict(Counter(r['channel'] for r in account['trades'])),
        first_odd_trade=odd[0] if odd else None,
        source_requests=[dict(date=d, stock_id=s, market=m, channel=c) for d,s,m,c in requests],
        strict_intraday=dict(completed=False, total_return=None, live_qualified=False,
            missing=['historical_odd_auction_evidence', 'original_precommitted_limit_and_order_time',
                     'verified_intraday_cash_and_slot_sequence']))


def run(output=OUTPUT, offline_replay=False, prepare=False):
    output = Path(output).resolve()
    if offline_replay and prepare:
        raise ValueError('Offline replay cannot prepare data')
    start = time.monotonic()
    context = parent.identity(parent.OUTPUT)
    for name in CODE:
        context[name] = sha(ROOT/name)
    manifest = read(parent.OUTPUT/'execution-manifest.json')['files_sha256']
    for name, digest in manifest.items():
        if sha(parent.OUTPUT/name) != digest:
            raise ValueError('Original sealed execution evidence changed: '+name)
    extra = read(ROOT/'docs/intraday_corporate_additions_20260914.json')
    for name, digest in extra['evidence_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Corporate evidence changed')
        context[name] = digest
    context['docs/intraday_corporate_additions_20260914.json'] = sha(ROOT/'docs/intraday_corporate_additions_20260914.json')
    context[str((parent.OUTPUT/'execution-manifest.json').relative_to(ROOT))] = sha(parent.OUTPUT/'execution-manifest.json')
    output.mkdir(parents=True, exist_ok=True)
    identity = output/'identity.json'
    if identity.exists() and read(identity) != context:
        raise ValueError('Source/code changed; use a new output directory')
    write(identity, context)
    if offline_replay:
        for name, digest in read(output/'manifest.json').items():
            if sha(output/name) != digest:
                raise ValueError('Frozen bridge source changed: '+name)
    cache = output/'inputs'
    if not cache.exists():
        if offline_replay:
            raise ValueError('Offline inputs missing')
        # Verify the additional isolated source copy before using it.
        for name, digest in read(TICK_OUTPUT/'manifest.json')['files_sha256'].items():
            if name.startswith('inputs/') and sha(TICK_OUTPUT/name) != digest:
                raise ValueError('Additional corporate source changed')
        shutil.copytree(TICK_OUTPUT/'inputs', cache)
    _, data, _ = parent.load_data(parent.OUTPUT)
    if any(item.get('accounting_only_after_end') and item['pay_date'] <= str(data.end)[:10]
           for item in extra['overrides'].values()):
        raise ValueError('Post-period delivery cannot enter study')
    from scripts.research_chip import ADDITIONS
    overrides = read(parent.parent.cash.OVERRIDES)['overrides'] | read(ADDITIONS)['overrides'] | extra['overrides']
    cases, ledgers = {}, {}
    budget = PreparationBudget(output)
    for stress in ('control', 'combined'):
        for ranking in ('capacity', 'benchmark'):
            for reserve in (False, True):
                name = f'{ranking}_{stress}_'+('reserved' if reserve else 'original')
                feeds = ReplayMarketFeeds(cache/'execution-feeds', offline=not prepare,
                    http_get=budget.http, finmind_fetch=budget.finmind)
                corporate = TrackedCorporateActions(data.events, cache/'dividends', None,
                                                    offline=True, overrides=overrides)
                args = (data.quotes, data.companies, data.days, data.entries, feeds, corporate)
                kwargs = dict(start=data.start, end=data.end, stress_mode=stress, reserve_before_open=reserve)
                engine = (ReservedBenchmark(*args, **kwargs) if ranking=='benchmark' else
                          ReservedCapacityReplay(*args, exit_signals=data.features,
                            action_dates=list(zip(data.events.stock_id, data.events.event_date)), **kwargs))
                print('running', name, flush=True)
                try:
                    account = engine.run()
                    audit = audit_reservations(account, engine.plans) if reserve else audit_stress(account)
                    if not reserve:
                        old = read(parent.OUTPUT/'cases'/f'{ranking}_{stress}.json')['account']
                        if encoded(account) != encoded(old):
                            raise ValueError('Neutral replay differs from original account')
                        ledgers[name] = inspect_ledger(account, engine.markets)
                    result = dict(completed=True, account=account, plans=engine.plans,
                                  audit=audit, summary=summarize(account), live_qualified=False)
                except (ReplayDataUnavailable, UnresolvedAction) as exc:
                    result = dict(completed=False, reason=str(exc), live_qualified=False)
                path = output/'cases'/f'{name}.json'
                if offline_replay:
                    if encoded(result) != encoded(read(path)):
                        raise ValueError('Offline bridge mismatch: '+name)
                else:
                    write(path, result)
                cases[name] = {k:v for k,v in result.items() if k not in ('account','plans')}
                cases[name].update(path=str(path.relative_to(ROOT)), sha256=sha(path))
                print(name, round(result['summary']['total_return']*100, 2) if result['completed'] else result['reason'], flush=True)
    report = dict(cases=cases, legacy_ledger_checks=ledgers,
                  all_completed=all(r['completed'] for r in cases.values()), live_qualified=False,
                  scope='daily_execution_reservation_contrast', network_calls=budget.calls,
                  elapsed_seconds=round(time.monotonic()-start, 3))
    if offline_replay:
        if encoded(ledgers) != encoded(read(output/'summary.json')['legacy_ledger_checks']):
            raise ValueError('Ledger diagnosis differs')
        write(output/'offline.json', dict(identical=True, network_calls=0,
            manifest_sha256=sha(output/'manifest.json'), elapsed_seconds=report['elapsed_seconds']))
    else:
        write(output/'summary.json', report)
        files = [p for folder in ('inputs','cases') for p in (output/folder).rglob('*')
                 if p.is_file() and p.suffix in ('.json','.parquet')]
        files += [output/'summary.json', identity]
        write(output/'manifest.json', {str(p.relative_to(output)):sha(p) for p in files})
    print('done', report['all_completed'], report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--offline-replay', action='store_true')
    p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    with file_lock(args.output/'run.lock', timeout=1):
        run(args.output, args.offline_replay, args.prepare)
