#!/usr/bin/env python3
"""Audit original order evidence; frozen-order estimates are NOT a new account."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.publish_entry_context_terminal import digest, write_sealed
from skills.all_entry_account_data import AllEntryData
from skills.poc_executable_replay import match_board_prints
from skills.repaired_execution_context import dated_market_resolver
from skills.replay_market_feeds import ReplayDataUnavailable
from skills.strategy_comparison_data import ComparisonDataRequired

SOURCE = 'artifacts/forward_simulation/poc_optimization_20261013.json'
SOURCE_SHA = 'ff6a949d487ad880afbb4191295221eef46d80fa4df362abd1a33b3c5cc9a264'


def audit(output, *, fetch=False):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve the previous audit; choose a new output')
    if digest(ROOT/SOURCE) != SOURCE_SHA:
        raise ValueError('Original account publication changed')
    publication = json.loads((ROOT/SOURCE).read_text())
    provider = AllEntryData(online=fetch)
    identity_path = ROOT/'.cache/poc-latest-20261003/inputs-v1/identity.json'
    market = dated_market_resolver(json.loads(identity_path.read_text()))
    refs = {SOURCE: SOURCE_SHA, str(identity_path.relative_to(ROOT)): digest(identity_path),
            str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    cases, cached = {}, {}
    for key, descriptor in publication['cases'].items():
        if descriptor['variant'] != 'original':
            continue
        path = ROOT/descriptor['path']
        if digest(path) != descriptor['sha256']:
            raise ValueError('Original case changed')
        refs[descriptor['path']] = descriptor['sha256']
        case = json.loads(path.read_text())
        if not case['completed']:
            raise ValueError('Original case is incomplete')
        account = case['account']
        plans = {(p['date'], p['event_id'], p['side']): p for p in account['tick_plans']}
        rows = []
        for order in account['orders']:
            if order['channel'] not in ('board', 'odd') or order['requested_qty'] <= 0:
                continue
            day, sid = order['date'], order['stock_id']
            plan = plans[(day, order['event_id'], order['side'])]
            if not plan['signal_date'] <= plan['reference_date'] < day:
                raise ValueError('Order was not planned from an earlier session')
            row = {k: order[k] for k in ('date', 'stock_id', 'event_id', 'side', 'channel',
                                         'requested_qty', 'filled_qty', 'limit_price')}
            row.update(original_reference_price=order.get('reference_price'),
                       original_hindsight_price=True, causal_fill_verified=False)
            if order['channel'] == 'odd':
                row.update(status='intraday_odd_chronology_missing', estimate=None,
                           reason='The archived source is daily OHLC/volume; no auction sequence or order allocation exists in it.')
            else:
                coordinate = sid, day, market(day, sid)
                if coordinate not in cached:
                    try:
                        tape, sha = provider.board_ticks.get(*coordinate)
                        cached[coordinate] = (tape, sha, None)
                    except (ReplayDataUnavailable, ComparisonDataRequired) as exc:
                        cached[coordinate] = (None, None, str(exc))
                tape, sha, error = cached[coordinate]
                if error:
                    row.update(status='ordinary_tape_unavailable_or_conflicting', estimate=None, reason=error)
                else:
                    estimate = match_board_prints(tape, order['side'], plan['limit_price'],
                        order['requested_qty'], order['prior_avg_volume20'])
                    row.update(status='ordinary_print_estimate_only', raw_sha256=sha,
                        estimate={k: v for k, v in estimate.items() if k != 'allocations'},
                        first_print=estimate['allocations'][0] if estimate['allocations'] else None,
                        estimate_equals_proxy_quantity=estimate['filled_qty'] == order['filled_qty'],
                        ordinary_daily_totals_reconciled=bool(
                            tape.attrs.get('execution_audit', {}).get('ordinary_volume_matched')
                            and tape.attrs.get('execution_audit', {}).get('ordinary_amount_matched')),
                        reason='Print availability permits a conservative estimate, not proof of complete tape or our own queue.')
            rows.append(row)
        counts = Counter(r['status'] for r in rows)
        cases[key] = dict(arm=descriptor['id'], mode=descriptor['mode'], rows=rows,
            checked_child_orders=len(rows), status_counts=dict(counts),
            ordinary_estimates=counts['ordinary_print_estimate_only'],
            original_net_return=descriptor['metrics']['total_return'],
            executable_account_net_return=None, executable_account_replayed=False,
            causal_fill_verified=False, actual_fill_verified=False,
            paper_enabled=False, live_qualified=False)
        print(key, dict(counts), flush=True)
    provider._merge_live_refs()
    refs.update(provider.refs)
    for path, expected in refs.items():
        if digest(ROOT/path) != expected:
            raise ValueError('Evidence changed during audit: '+path)
    report = dict(schema='poc_order_timing_audit_v1', created_at=datetime.now(timezone.utc).isoformat(),
        start=publication['start'], end=publication['end'], cases=cases, source_sha256=refs,
        unique_ordinary_coordinates=len(cached), finmind_requests=provider.finmind_calls,
        completed=True, passed=False, whole_account_replayed=False, executable_net_return=None,
        blockers=['Original daily HL2/sell30 prices are known only after the session.',
                  'Intraday odd-lot chronological observations and allocation are absent.',
                  'Ordinary prints do not certify complete exchange sequence or own queue.',
                  'Different fills require a new full cash/holdings account; frozen-order estimates cannot be compounded.'],
        live_qualified=False, actual_fill_verified=False, paper_enabled=False)
    write_sealed(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    result = audit(args.output, fetch=args.fetch)
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'source_sha256')}))
