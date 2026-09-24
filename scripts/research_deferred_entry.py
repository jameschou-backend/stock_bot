#!/usr/bin/env python3
"""Bounded two-session candidate experiment with immutable/offline controls."""
from pathlib import Path
import argparse
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import research_slot_reuse as parent
from skills.deferred_entry_replay import DeferredEntryReplay
from skills.slot_reuse_replay import audit_slots
from app.file_lock import file_lock

OUTPUT = ROOT / '.cache/deferred-entry-20260924'
SPEC = ROOT / 'docs/prereg_deferred_entry_20260924.md'


def inventory():
    refs = parent.inventory()
    for name, digest in parent.read(parent.OUTPUT/'manifest.json')['files_sha256'].items():
        if parent.sha(parent.OUTPUT/name) != digest:
            raise ValueError('Sealed slot evidence changed: '+name)
    for path in (SPEC, Path(__file__), ROOT/'skills/deferred_entry_replay.py', parent.OUTPUT/'manifest.json'):
        refs[str(path.relative_to(ROOT))] = parent.sha(path)
    return refs


def run_case(data, config, cache, budget, prepare):
    token = parent.load_config().finmind_token if prepare else None
    feeds = parent.ReplayMarketFeeds(cache/'execution-feeds', offline=not prepare, token=token,
        http_get=budget.http, finmind_fetch=budget.finmind)
    overrides = (parent.read(parent.parent.parent.five.parent.cash.OVERRIDES)['overrides'] |
        parent.read(parent.ADDITIONS)['overrides'] |
        parent.read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'])
    corp = parent.TrackedCorporateActions(data.events, cache/'dividends', token, offline=True, overrides=overrides)
    args = (data.quotes, data.companies, data.days, data.entries, feeds, corp)
    kw = dict(start=data.start, end=data.end, stress_mode=config['stress'])
    if config['benchmark']:
        engine = parent.ResourceBenchmark(*args, **kw, opening_cash_only=True, lock_unused=True)
    else:
        engine = DeferredEntryReplay(*args, **kw, validity_sessions=config['validity_sessions'],
            exit_signals=data.features, action_dates=list(zip(data.events.stock_id,data.events.event_date)))
    try:
        account = engine.run()
        if config['benchmark']:
            audit = parent.audit_resources(account, engine.resource_plans, opening_cash_only=True,
                lock_unused=True, lock_slots=False)
        else:
            audit = audit_slots(account, engine.resource_plans, engine.slot_decisions,
                opening_cash_only=True, lock_unused=True, lock_opening_slots=True, lock_failed_slots=True)
    except (parent.ReplayDataUnavailable, parent.UnresolvedAction) as exc:
        return dict(completed=False, reason=str(exc), config=config, live_qualified=False)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        return dict(completed=False, reason=str(exc), config=config, live_qualified=False)
    control = None
    if config['benchmark']:
        control = f"benchmark_{config['stress']}_cash1"
    elif config['validity_sessions'] == 1:
        control = f"capacity_{config['stress']}_cash1_rf11"
    if control and parent.encoded(account) != parent.encoded(parent.read(parent.OUTPUT/'cases'/f'{control}.json')['account']):
        raise ValueError('Parent account mismatch: '+control)
    return dict(completed=True, config=config, account=account, summary=parent.summarize(account),
        resource_plans=engine.resource_plans, slot_decisions=getattr(engine,'slot_decisions',[]),
        retry_decisions=getattr(engine,'retry_decisions',[]), audit=audit, parent_control=control,
        live_qualified=False, unseen_validation=False)


def run(output=OUTPUT, prepare=False, offline=False):
    if prepare and offline:
        raise ValueError('Offline cannot prepare inputs')
    start = time.monotonic(); refs = inventory()
    if (output/'identity.json').exists() and parent.read(output/'identity.json') != refs:
        raise ValueError('Source/code changed; choose a new immutable output')
    if offline:
        for name, digest in parent.read(output/'manifest.json')['files_sha256'].items():
            if parent.sha(output/name) != digest:
                raise ValueError('Sealed result changed: '+name)
    else:
        parent.write(output/'identity.json',refs)
    cache = output/'inputs'
    if not cache.exists():
        if offline:
            raise ValueError('Offline inputs missing')
        shutil.copytree(parent.OUTPUT/'inputs',cache)
    data, _ = parent.parent.parent.inputs(); budget = parent.parent.Budget(output)
    rows = {}
    for stress in ('control','combined'):
        for valid in (1,2,0):
            name = f"{'benchmark' if valid==0 else 'capacity'}_{stress}_{valid}"
            config = dict(stress=stress, validity_sessions=valid, benchmark=valid==0)
            path = output/'cases'/f'{name}.json'
            print('running',name,flush=True)
            result = run_case(data,config,cache,budget,prepare)
            if offline:
                if parent.encoded(result) != parent.encoded(parent.read(path)):
                    raise ValueError('Offline case differs: '+name)
            else:
                parent.write(path,result)
            rows[name] = {k:v for k,v in result.items() if k not in (
                'account','resource_plans','slot_decisions','retry_decisions')}
            rows[name].update(path=str(path.relative_to(ROOT)),sha256=parent.sha(path))
            print(name,result.get('summary',{}).get('total_return',result.get('reason')),flush=True)
    for name,row in rows.items():
        if row['config']['benchmark'] or not row['completed']:
            continue
        base = rows[f"benchmark_{row['config']['stress']}_0"]
        if base['completed']:
            row['benchmark'] = base['summary']
            row['rolling252'] = parent.parent.parent.rolling_comparison(
                parent.read(ROOT/row['path'])['account'],parent.read(ROOT/base['path'])['account'])
    if inventory() != refs:
        raise ValueError('Sources changed during research')
    report = dict(cases=rows,all_completed=all(x['completed'] for x in rows.values()),
        elapsed_seconds=round(time.monotonic()-start,3),network_calls=budget.calls,
        live_qualified=False,unseen_validation=False)
    if offline:
        if rows != parent.read(output/'summary.json')['cases']:
            raise ValueError('Offline summaries differ')
        parent.write(output/'offline.json',dict(all_cases_identical=True,**{k:v for k,v in report.items() if k!='cases'}))
    else:
        parent.write(output/'summary.json',report)
        files = [p for p in output.rglob('*') if p.is_file() and p.suffix in ('.json','.parquet')
                 and p.name not in ('manifest.json','offline.json')]
        parent.write(output/'manifest.json',dict(files_sha256={str(p.relative_to(output)):parent.sha(p) for p in files}))
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--offline-replay',action='store_true')
    a=p.parse_args()
    with file_lock(ROOT/'.cache/deferred-entry.lock',timeout=0):
        result=run(a.output,a.prepare,a.offline_replay)
    print('completed',sum(x['completed'] for x in result['cases'].values()),'/ 6')
