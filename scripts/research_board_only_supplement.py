#!/usr/bin/env python3
"""Immutable offline source supplementation; board-only strategy parameters stay sealed."""
from copy import deepcopy
from pathlib import Path
import argparse
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.file_lock import file_lock
from scripts import research_board_only as parent
from scripts.research_exit_scenarios import read, write, sha, encoded, summarize, TrackedCorporateActions
from skills.board_only_replay import execution_summary
from skills.board_only_verified_replay import (BoardOnlyVerifiedReplay, BoardOnlyVerifiedBenchmark,
                                              audit_verified_board_only)
from skills.execution_resources import audit_resources
from skills.slot_reuse_replay import audit_slots
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable
from skills.million_replay import UnresolvedAction

SPEC = ROOT / 'docs/prereg_board_only_supplement_20260925.md'
SOURCES = ROOT / '.cache/board-only-source-supplement-20260925'
OUTPUT = ROOT / '.cache/board-only-supplement-20260925'


def inventory(sources):
    refs = parent.inventory()
    if refs != read(parent.OUTPUT / 'identity.json'):
        raise ValueError('Original board-only source/code changed')
    for name, digest in read(parent.OUTPUT / 'manifest.json')['files_sha256'].items():
        if sha(parent.OUTPUT / name) != digest:
            raise ValueError('Original board-only evidence changed: ' + name)
    budget = read(sources / 'budget.json')
    for key, maximum in (('finmind', 10), ('official', 10)):
        if type(budget.get(key)) is not int or not 0 <= budget[key] <= maximum:
            raise ValueError('Supplement source budget invalid/exceeded: ' + key)
    ledger = read(sources / 'source-ledger.json')
    preparation = ROOT / 'scripts/prepare_board_only_supplement.py'
    if (ledger['parent_manifest_sha256'] != sha(parent.OUTPUT / 'manifest.json')
            or ledger['preparation_code_sha256'] != sha(preparation)
            or ledger['finmind_fetch_attempts'] != budget['finmind']):
        raise ValueError('Supplement preparation provenance/budget differs')
    refs[str(preparation.relative_to(ROOT))] = sha(preparation)
    old = read(parent.OUTPUT / 'inputs/execution-feeds/index.json')
    current = ReplayMarketFeeds(sources / 'inputs/execution-feeds', offline=True).manifest()
    accepted = {'limits:' + row['stock_id'] for row in ledger['requests'] if row['status'] == 'success'}
    if not (set(current['entries']) - set(old['entries'])).issubset(accepted):
        raise ValueError('Supplement feed has no successful preparation record')
    for key, value in old['entries'].items():
        if current['entries'].get(key) != value:
            raise ValueError('Original feed entry was changed: ' + key)
    for name, digest in old['files_sha256'].items():
        if current['files_sha256'].get(name) != digest or sha(sources / 'inputs/execution-feeds' / name) != digest:
            raise ValueError('Original feed file was changed: ' + name)
    for path in (parent.OUTPUT / 'inputs/dividends').glob('*.parquet'):
        if sha(sources / 'inputs/dividends' / path.name) != sha(path):
            raise ValueError('Original dividend source was changed: ' + path.name)
    additions = read(sources / 'overrides.json')
    base = (read(parent.OVERRIDES)['overrides'] | read(parent.ADDITIONS)['overrides'] |
            read(ROOT / 'docs/intraday_corporate_additions_20260914.json')['overrides'])
    if set(additions['overrides']) & set(base):
        raise ValueError('Supplement must not overwrite an existing corporate override')
    if additions['overrides'] and not additions.get('evidence_sha256'):
        raise ValueError('Supplement corporate override lacks source hashes')
    for name, digest in additions.get('evidence_sha256', {}).items():
        if sha(ROOT / name) != digest:
            raise ValueError('Supplement corporate evidence changed: ' + name)
        refs[name] = digest
    for path in (SPEC, Path(__file__), parent.OUTPUT / 'manifest.json',
                 ROOT / 'skills/board_only_verified_replay.py', ROOT / 'tests/test_board_only_verified_replay.py'):
        refs[str(path.relative_to(ROOT))] = sha(path)
    for path in sources.rglob('*'):
        if path.is_file() and path.suffix != '.lock':
            refs[str(path.relative_to(ROOT))] = sha(path)
    return refs


def audit_account(account, plans, slots, decisions, benchmark):
    audit = (audit_resources(account, plans, opening_cash_only=True, lock_unused=True, lock_slots=False)
             if benchmark else audit_slots(account, plans, slots, opening_cash_only=True, lock_unused=True,
                 lock_opening_slots=True, lock_failed_slots=True))
    audit.update(audit_verified_board_only(account, decisions, plans))
    return audit


def run_case(data, config, inputs, additions):
    if not config['board_only']:
        return parent.run_case(data, config, inputs, parent.NoNetwork())
    feeds = ReplayMarketFeeds(inputs / 'execution-feeds', offline=True,
        http_get=parent.NoNetwork().http, finmind_fetch=parent.NoNetwork().finmind)
    overrides = (read(parent.OVERRIDES)['overrides'] | read(parent.ADDITIONS)['overrides'] |
        read(ROOT / 'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    corp = TrackedCorporateActions(data.events, inputs / 'dividends', None, offline=True, overrides=overrides)
    args = (data.quotes, data.companies, data.days, data.entries, feeds, corp)
    kwargs = dict(start=data.start, end=data.end, stress_mode=config['stress'])
    engine = (BoardOnlyVerifiedBenchmark(*args, **kwargs) if config['benchmark'] else BoardOnlyVerifiedReplay(
        *args, **kwargs, exit_signals=data.features, action_dates=list(zip(data.events.stock_id, data.events.event_date))))
    try:
        account = engine.run()
        audit = audit_account(account, engine.resource_plans, getattr(engine, 'slot_decisions', []),
                              engine.board_decisions, config['benchmark'])
    except (ReplayDataUnavailable, UnresolvedAction) as exc:
        return parent.blocked(config, str(exc), engine)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        return parent.blocked(config, str(exc), engine)
    result = dict(completed=True, config=config, account=account, summary=summarize(account), audit=audit,
        resource_plans=engine.resource_plans, slot_decisions=getattr(engine, 'slot_decisions', []),
        board_decisions=engine.board_decisions, execution=execution_summary(account, engine.board_decisions),
        live_qualified=False, unseen_validation=False)
    if config['benchmark']:
        prior = parent.OUTPUT / 'cases' / f"benchmark_{config['stress']}_board_only.json"
        if encoded(account) != encoded(read(prior)['account']):
            raise ValueError('Source supplement changed the complete board-only benchmark account')
        result['original_board_benchmark_identical'] = True
    return result


def partial_audit(case, mixed):
    account = deepcopy(case['partial_account'])
    if not account['daily']:
        return dict(completed_days=0, audit=None)
    last = account['daily'][-1]['date']
    for key in ('daily', 'trades', 'orders', 'corporate_actions', 'cash_ledger', 'holdings'):
        account[key] = [row for row in account[key] if row['date'] <= last]
    account['cohorts'] = [row for row in account['cohorts'] if row['entry_date'] <= last]
    account['settings'] = dict(mixed['account']['settings'], execution_policy='board_only')
    journals = [[r for r in case[key] if r['date'] <= last]
                for key in ('resource_plans', 'slot_decisions', 'board_decisions')]
    return dict(last_complete_date=last, completed_days=len(account['daily']),
        audit=audit_account(account, *journals, case['config']['benchmark']),
        scope='Only completed market days; no full-period return is inferred')


def run(output=OUTPUT, sources=SOURCES, offline_replay=False):
    tick = time.monotonic()
    refs = inventory(sources)
    if offline_replay:
        if refs != read(output / 'identity.json'):
            raise ValueError('Supplement source/code identity changed')
        for name, digest in read(output / 'manifest.json')['files_sha256'].items():
            if sha(output / name) != digest:
                raise ValueError('Sealed supplement result changed: ' + name)
    else:
        if output.exists():
            raise ValueError('Choose a new immutable output; previous results must be retained')
        write(output / 'identity.json', refs)
        shutil.copytree(sources / 'inputs', output / 'inputs')
        write(output / 'overrides.json', read(sources / 'overrides.json'))
    additions = read(output / 'overrides.json')['overrides']
    data, _ = parent.source.inputs()
    if len(data.entries) != 458 or str(data.start)[:10] != '2022-01-03' or str(data.end)[:10] != '2026-09-09':
        raise ValueError('Preregistered sample changed')
    rows, controls_ok = {}, True
    for name, config in parent.configurations():
        print('running', name, flush=True)
        result = (parent.blocked(config, 'Latest mixed controls did not all reproduce')
                  if config['board_only'] and not controls_ok else run_case(data, config, output / 'inputs', additions))
        if not config['board_only']:
            controls_ok = controls_ok and result['completed'] and result.get('parent_account_identical', False)
        if not result['completed'] and 'partial_account' in result:
            mixed = read(output / 'cases' / f"{'benchmark' if config['benchmark'] else 'capacity'}_{config['stress']}_mixed.json")
            result['partial_audit'] = partial_audit(result, mixed)
        path = output / 'cases' / f'{name}.json'
        if offline_replay:
            if encoded(result) != encoded(read(path)):
                raise ValueError('Supplement offline case differs: ' + name)
        else:
            write(path, result)
        rows[name] = {k: v for k, v in result.items() if k not in parent.LARGE_FIELDS}
        rows[name].update(path=str(path.relative_to(ROOT)), sha256=sha(path))
        print(name, result.get('summary', {}).get('total_return', result.get('reason')), flush=True)
    for row in rows.values():
        if row['config']['benchmark'] or not row['completed']:
            continue
        policy = 'board_only' if row['config']['board_only'] else 'mixed'
        for key, variant in (('same_policy', policy), ('mixed_reference', 'mixed')):
            base = f"benchmark_{row['config']['stress']}_{variant}"
            if rows[base]['completed']:
                row[key] = dict(benchmark_case=base, summary=rows[base]['summary'],
                    rolling252=parent.source.rolling_comparison(read(ROOT / row['path'])['account'],
                                                                read(ROOT / rows[base]['path'])['account']))
    if inventory(sources) != refs:
        raise ValueError('Supplement sources changed during execution')
    report = dict(cases=rows, parent_controls_identical=controls_ok,
        all_completed=all(r['completed'] for r in rows.values()), candidate_count=len(data.entries),
        elapsed_seconds=round(time.monotonic()-tick, 3), network_calls=0,
        source_preparation_budget=read(sources / 'budget.json'), database_writes=0,
        live_qualified=False, unseen_validation=False, limitations=read(parent.OUTPUT / 'summary.json')['limitations'])
    if offline_replay:
        if rows != read(output / 'summary.json')['cases']:
            raise ValueError('Supplement offline summaries differ')
        write(output / 'offline.json', dict(all_cases_identical=True, all_completed=report['all_completed'],
            parent_controls_identical=controls_ok, elapsed_seconds=report['elapsed_seconds'], network_calls=0,
            manifest_sha256=sha(output / 'manifest.json')))
    else:
        write(output / 'summary.json', report)
        files = [p for p in output.rglob('*') if p.is_file() and p.suffix in ('.json', '.parquet')
                 and p.name not in ('manifest.json', 'offline.json')]
        write(output / 'manifest.json', dict(files_sha256={str(p.relative_to(output)): sha(p) for p in files},
                                            live_qualified=False, unseen_validation=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--sources', type=Path, default=SOURCES)
    parser.add_argument('--offline-replay', action='store_true')
    args = parser.parse_args()
    with file_lock(ROOT / '.cache/board-only-supplement.lock', timeout=0):
        report = run(args.output, args.sources, args.offline_replay)
    print('completed', sum(r['completed'] for r in report['cases'].values()), '/ 8', flush=True)
