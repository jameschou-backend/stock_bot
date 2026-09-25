#!/usr/bin/env python3
"""Fill only an audited plan's missing ordinary-board stock-days via shared quota."""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.file_lock import file_lock
from scripts.research_intraday_limit import TickCache
from scripts.audit_market_identity import resolve_on
from skills.backtest_data_evidence import digest, verify_report, inspect_tape

MAXIMUM_ALLOWED = 600
OUTPUT = ROOT / '.cache/backtest-board-ticks-20260925'


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def requests_from_report(report, episodes):
    indexed = defaultdict(list)
    for item in episodes:
        indexed[item['stock_id']].append(item)
    requests = sorted({(row['date'], row['stock_id']) for case in report['cases'].values()
                       for row in case['ordinary']['missing']})
    result = []
    for day, sid in requests:
        identity = resolve_on(indexed[sid], sid, day)
        if identity['status'] != 'identified':
            raise ValueError('Missing dated market identity for planned tick: ' + sid + ' ' + day)
        result.append(dict(date=day, stock_id=sid, market=identity['market'].upper()))
    return result


def prepare(plan, output, maximum, fetch=False):
    plan, output = Path(plan).resolve(), Path(output).resolve()
    if not output.is_relative_to(ROOT / '.cache') or output in (ROOT, ROOT / '.cache'):
        raise ValueError('Use a separate cache directory for the prepared sources')
    if type(maximum) is not int or not 1 <= maximum <= MAXIMUM_ALLOWED:
        raise ValueError('Explicit hard budget must be 1..600 shared-adapter calls')
    report = verify_report(plan, ROOT)
    listing_path = ROOT / '.cache/listing-continuation-20260924/report.json'
    if report['input_sha256'][str(listing_path.relative_to(ROOT))] != digest(listing_path):
        raise ValueError('Dated identity source differs from plan')
    requests = requests_from_report(report, json.loads(listing_path.read_text())['episodes'])
    if len(requests) > maximum:
        raise ValueError(f'{len(requests)} missing stock-days exceed explicit budget {maximum}; no fetch started')
    identity = dict(schema='bounded_backtest_board_tick_plan_v1',
        plan_path=str(plan.relative_to(ROOT)), plan_sha256=digest(plan), requests=requests,
        maximum=maximum, code_sha256={name:digest(ROOT/name) for name in (
            'scripts/prepare_backtest_board_ticks.py', 'scripts/research_intraday_limit.py',
            'skills/intraday_limit_replay.py', 'app/finmind.py')})
    if not fetch:
        print(json.dumps(dict(planned_stock_days=len(requests), hard_maximum=maximum, network_calls=0)))
        return identity
    if output.exists() and any(output.iterdir()) and not (output / 'identity.json').is_file():
        raise ValueError('Refusing to write even a lock into an existing unbound cache')
    with file_lock(output / 'run.lock', timeout=1):
        identity_path = output / 'identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError('Existing preparation identity differs; use a new cache version')
        elif any(p.name != 'run.lock' for p in output.iterdir()):
            raise ValueError('Refusing to adopt pre-existing unbound source files')
        else:
            write(identity_path, identity)
        attempt_path = output / 'attempts.json'
        attempts = json.loads(attempt_path.read_text()) if attempt_path.exists() else {}
        # TickCache itself persists a reservation before transport, uses the
        # shared fetch_dataset limiter/cache and disables automatic retries.
        cache = TickCache(output / 'ticks', online=True, maximum=maximum)
        rows = []
        def summary():
            budget = output / 'ticks/budget.json'
            return dict(schema='bounded_backtest_board_ticks_v1', rows=rows,
                all_completed=len(rows)==len(requests) and all(r['completed'] for r in rows),
                planned_stock_days=len(requests), requests_this_run=cache.calls,
                adapter_attempts_lifetime=json.loads(budget.read_text())['reserved'] if budget.exists() else 0,
                hard_maximum=maximum, live_qualified=False, historical_odd_verified=False,
                independently_authenticated_complete_sessions=False,
                plan_path=str(plan.relative_to(ROOT)), plan_sha256=digest(plan),
                identity_path=str(identity_path.relative_to(ROOT)), identity_sha256=digest(identity_path),
                attempts_path=str(attempt_path.relative_to(ROOT)), attempts_sha256=digest(attempt_path))
        write(attempt_path, attempts)
        for request in requests:
            key = request['stock_id'] + '-' + request['date']
            path = cache.root / (key + '.parquet')
            previous = attempts.get(key)
            if previous and previous['status'] != 'success':
                raise RuntimeError('Prior interrupted/failed request requires review before retry: ' + key)
            if previous is None:
                if path.exists() or path.with_suffix('.json').exists():
                    raise ValueError('Unbound file exists for an unattempted stock-day')
                attempts[key] = dict(status='started', **request)
                write(attempt_path, attempts)
            elif not path.exists() or not path.with_suffix('.json').exists():
                raise ValueError('Previously completed tape is missing; refusing implicit refetch')
            try:
                frame, value = cache.get(request['stock_id'], request['date'], request['market'])
                item = dict(request, path=str(path.relative_to(ROOT)), sha256=value,
                    metadata_sha256=digest(path.with_suffix('.json')), format='finmind_board', channel='board')
                inspect_tape(item, ROOT, {})
            except Exception as exc:
                attempts[key]['status'] = 'failed'
                attempts[key]['error_type'] = type(exc).__name__
                write(attempt_path, attempts)
                rows.append(dict(request, completed=False, error_type=type(exc).__name__))
                write(output / 'summary.json', summary())
                # Quota/service failures stop here. No secret-bearing exception
                # payload is copied into repository evidence or stdout.
                raise RuntimeError('Tick preparation stopped: ' + type(exc).__name__ + ' for ' + key) from None
            attempts[key]['status'] = 'success'
            write(attempt_path, attempts)
            rows.append(dict(item, completed=True, rows=len(frame)))
            write(output / 'summary.json', summary())
            if len(rows) % 10 == 0 or len(rows) == len(requests):
                print(json.dumps(dict(completed_stock_days=len(rows), planned_stock_days=len(requests),
                    adapter_calls=cache.calls)), flush=True)
        return summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--maximum', type=int, required=True)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    prepare(args.plan, args.output, args.maximum, args.fetch)
