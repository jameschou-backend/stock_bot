#!/usr/bin/env python3
"""Offline-only board-lot execution contrast on the sealed conservative five-slot account."""
from pathlib import Path
import argparse
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.file_lock import file_lock
from scripts import research_conservative_diversification as parent
from scripts import research_priority as source
from scripts.research_cash_allocation import OVERRIDES
from scripts.research_chip import ADDITIONS
from scripts.research_exit_scenarios import read, write, sha, encoded, summarize, TrackedCorporateActions
from skills.board_only_replay import BoardOnlyReplay, BoardOnlyBenchmark, audit_board_only, execution_summary
from skills.execution_resources import audit_resources
from skills.slot_reuse_replay import audit_slots
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable
from skills.million_replay import UnresolvedAction

OUTPUT = ROOT / '.cache/board-only-20260925'
SPEC = ROOT / 'docs/prereg_board_only_20260925.md'
LARGE_FIELDS = ('account', 'partial_account', 'resource_plans', 'slot_decisions', 'retry_decisions', 'board_decisions')


class NoNetwork:
    calls = 0

    def http(self, *args, **kwargs):
        raise RuntimeError('Board-only preregistration prohibits network requests')

    finmind = http


def inventory():
    refs = parent.inventory()
    if refs != read(parent.OUTPUT / 'identity.json'):
        raise ValueError('Sealed diversification source/code identity changed')
    for name, digest in read(parent.OUTPUT / 'manifest.json')['files_sha256'].items():
        if sha(parent.OUTPUT / name) != digest:
            raise ValueError('Sealed diversification evidence changed: ' + name)
    for path in (SPEC, Path(__file__), ROOT / 'skills/board_only_replay.py',
                 ROOT / 'tests/test_board_only_replay.py', parent.OUTPUT / 'manifest.json'):
        refs[str(path.relative_to(ROOT))] = sha(path)
    return refs


def configurations():
    # All four exact controls must finish before ANY new-account calculation.
    for board in (False, True):
        for stress in ('control', 'combined'):
            for benchmark in (False, True):
                label = 'benchmark' if benchmark else 'capacity'
                yield f"{label}_{stress}_{'board_only' if board else 'mixed'}", dict(
                    stress=stress, benchmark=benchmark, board_only=board, position_count=0 if benchmark else 5)


def blocked(config, reason, engine=None):
    result = dict(completed=False, config=config, reason=reason, live_qualified=False, unseen_validation=False)
    if engine is not None:
        result.update(partial_account={k: getattr(engine, attr) for k, attr in (
            ('daily', 'daily'), ('trades', 'trades'), ('orders', 'orders'),
            ('corporate_actions', 'actions'), ('cash_ledger', 'cash_ledger'),
            ('holdings', 'holding_rows'), ('cohorts', 'cohorts'), ('receivables', 'receivables'))},
            resource_plans=engine.resource_plans, slot_decisions=getattr(engine, 'slot_decisions', []),
            board_decisions=engine.board_decisions)
    return result


def run_case(data, config, cache, budget):
    if not config['board_only']:
        original_config = {k: v for k, v in config.items() if k != 'board_only'}
        result = parent.run_case(data, original_config, cache, budget, False)
        if not result['completed']:
            return blocked(config, result['reason'])
        control = f"{'benchmark' if config['benchmark'] else 'capacity'}_{config['stress']}_{config['position_count']}"
        if encoded(result['account']) != encoded(read(parent.OUTPUT / 'cases' / f'{control}.json')['account']):
            raise ValueError('Latest complete parent account mismatch: ' + control)
        result.update(config=config, parent_control=control, parent_account_identical=True,
                      execution=execution_summary(result['account']))
        return result
    feeds = ReplayMarketFeeds(cache / 'execution-feeds', offline=True,
                              http_get=budget.http, finmind_fetch=budget.finmind)
    overrides = (read(OVERRIDES)['overrides'] | read(ADDITIONS)['overrides'] |
                 read(ROOT / 'docs/intraday_corporate_additions_20260914.json')['overrides'])
    corp = TrackedCorporateActions(data.events, cache / 'dividends', None, offline=True, overrides=overrides)
    args = (data.quotes, data.companies, data.days, data.entries, feeds, corp)
    kwargs = dict(start=data.start, end=data.end, stress_mode=config['stress'])
    engine = BoardOnlyBenchmark(*args, **kwargs) if config['benchmark'] else BoardOnlyReplay(
        *args, **kwargs, exit_signals=data.features, action_dates=list(zip(data.events.stock_id, data.events.event_date)))
    try:
        account = engine.run()
        if config['benchmark']:
            audit = audit_resources(account, engine.resource_plans, opening_cash_only=True,
                                    lock_unused=True, lock_slots=False)
        else:
            audit = audit_slots(account, engine.resource_plans, engine.slot_decisions,
                opening_cash_only=True, lock_unused=True, lock_opening_slots=True, lock_failed_slots=True)
        audit.update(audit_board_only(account, engine.board_decisions, engine.resource_plans))
    except (ReplayDataUnavailable, UnresolvedAction) as exc:
        return blocked(config, str(exc), engine)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        return blocked(config, str(exc), engine)
    return dict(completed=True, config=config, account=account, summary=summarize(account),
        resource_plans=engine.resource_plans, slot_decisions=getattr(engine, 'slot_decisions', []),
        board_decisions=engine.board_decisions, audit=audit,
        execution=execution_summary(account, engine.board_decisions), live_qualified=False, unseen_validation=False)


def run(output=OUTPUT, offline_replay=False):
    tick = time.monotonic()
    refs = inventory()
    if offline_replay:
        if read(output / 'identity.json') != refs:
            raise ValueError('Board-only source/code changed; sealed run cannot be replayed')
        for name, digest in read(output / 'manifest.json')['files_sha256'].items():
            if sha(output / name) != digest:
                raise ValueError('Sealed board-only result changed: ' + name)
    else:
        if output.exists():
            raise ValueError('Choose a new immutable output; existing evidence must be retained')
        write(output / 'identity.json', refs)
        shutil.copytree(parent.OUTPUT / 'inputs', output / 'inputs')
    data, _ = source.inputs()
    if len(data.entries) != 458 or str(data.start)[:10] != '2022-01-03' or str(data.end)[:10] != '2026-09-09':
        raise ValueError('Preregistered sample changed')
    budget, rows, controls_ok = NoNetwork(), {}, True
    for name, config in configurations():
        print('running', name, flush=True)
        result = (blocked(config, 'Latest mixed controls did not all reproduce') if config['board_only'] and not controls_ok
                  else run_case(data, config, output / 'inputs', budget))
        if not config['board_only']:
            controls_ok = controls_ok and result['completed'] and result.get('parent_account_identical', False)
        path = output / 'cases' / f'{name}.json'
        if offline_replay:
            if encoded(result) != encoded(read(path)):
                raise ValueError('Offline case differs: ' + name)
        else:
            write(path, result)
        rows[name] = {k: v for k, v in result.items() if k not in LARGE_FIELDS}
        rows[name].update(path=str(path.relative_to(ROOT)), sha256=sha(path))
        print(name, result.get('summary', {}).get('total_return', result.get('reason')), flush=True)
    for name, row in rows.items():
        if row['config']['benchmark'] or not row['completed']:
            continue
        policy = 'board_only' if row['config']['board_only'] else 'mixed'
        benchmark = f"benchmark_{row['config']['stress']}_{policy}"
        mixed = f"benchmark_{row['config']['stress']}_mixed"
        row['benchmark_case'] = benchmark
        for key, base in (('same_policy', benchmark), ('mixed_reference', mixed)):
            if rows[base]['completed']:
                row[key] = dict(benchmark_case=base, summary=rows[base]['summary'],
                    rolling252=source.rolling_comparison(read(ROOT / row['path'])['account'],
                                                        read(ROOT / rows[base]['path'])['account']))
    if inventory() != refs:
        raise ValueError('Sources changed during board-only research')
    report = dict(cases=rows, parent_controls_identical=controls_ok,
        all_completed=all(row['completed'] for row in rows.values()), candidate_count=len(data.entries),
        elapsed_seconds=round(time.monotonic()-tick, 3), network_calls=0, database_writes=0,
        live_qualified=False, unseen_validation=False,
        limitations=['Daily board-lot execution is an assumption, not intraday/order evidence',
                     'Residual shares remain marked and occupy slots; final NAV is not spendable cash',
                     'Historical universe and publication revisions remain incomplete',
                     'Historical period is already seen; overlapping windows are not independent samples'])
    if offline_replay:
        if rows != read(output / 'summary.json')['cases']:
            raise ValueError('Offline summaries differ')
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
    parser.add_argument('--offline-replay', action='store_true')
    options = parser.parse_args()
    with file_lock(ROOT / '.cache/board-only.lock', timeout=0):
        report = run(options.output, options.offline_replay)
    print('completed', sum(r['completed'] for r in report['cases'].values()), '/ 8', flush=True)
