#!/usr/bin/env python3
"""Reconstruct independent daily opportunities and optionally fill missing tapes.

One bounded invocation, not a scheduler. Every published snapshot is immutable.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.file_lock import file_lock
from scripts.prepare_volume_profile import write
from skills.poc_daily_opportunities import DailyProfiles, BASE, MAXIMUM_CALLS


def publish(provider, output, html):
    report = provider.snapshot(output)
    if html:
        from scripts.export_poc_daily_explorer import run
        run(SimpleNamespace(report=output/'report.json', output=html,
            template=ROOT/'ui/poc_daily_signal_explorer.html',
            payload=output/'explorer/payload.json', receipt=output/'explorer/receipt.json'))
    summary = dict(snapshot=str(output.relative_to(ROOT)), status_counts=report['status_counts'],
                   attempted_calls=provider.attempt_count, quota_delay_seconds=provider.quota_delay,
                   updated_at=datetime.now(timezone.utc).isoformat())
    write(BASE/'latest.json', summary)
    print(json.dumps(summary), flush=True)
    return summary


def run(args):
    if not 0 <= args.maximum_new_calls <= MAXIMUM_CALLS:
        raise ValueError('New-call budget must be between 0 and 40000')
    if args.maximum_new_calls and not args.fetch:
        raise ValueError('A positive request budget needs --fetch')
    if args.wait_for_quota and not args.fetch:
        raise ValueError('Quota waiting is only valid with --fetch')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = args.output.resolve()/run_id
    with file_lock(BASE/'data-v1/.run.lock', timeout=0):
        provider = DailyProfiles()
        print('sources_verified; checking all cached days', flush=True)
        provider.offline()
        serial = 0
        publish(provider, destination/f'{serial:04d}', args.html)
        def progress(p):
            print(json.dumps(dict(stage='fetching', attempted_calls=p.attempt_count,
                calls_this_run=p.calls_this_run, audited_days=len(p.days_cache),
                recent_completed=dict(Counter(r.get('status', 'pending_data') for r in p.rows.values())))), flush=True)
        while args.fetch and provider.calls_this_run < args.maximum_new_calls:
            remaining = args.maximum_new_calls-provider.calls_this_run
            before = provider.calls_this_run
            provider.fill(min(args.checkpoint_calls, remaining), progress=progress)
            serial += 1
            publish(provider, destination/f'{serial:04d}', args.html)
            if not any(r['status'] == 'pending_data' for r in provider.rows.values()): break
            if provider.quota_delay:
                if not args.wait_for_quota: break
                # A single active job may wait; no persistent/recurring schedule is created.
                left = provider.quota_delay
                while left > 0:
                    interval = min(left, 30.)
                    print(json.dumps(dict(stage='quota_wait', seconds_left=round(left),
                        attempted_calls=provider.attempt_count)), flush=True)
                    time.sleep(interval); left -= interval
            elif provider.calls_this_run == before:
                # Only provider failures or orphaned/empty attempts remain; never retry blindly.
                break
        return dict(Counter(r['status'] for r in provider.rows.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--maximum-new-calls', type=int, default=0)
    parser.add_argument('--checkpoint-calls', type=int, default=500)
    parser.add_argument('--wait-for-quota', action='store_true')
    parser.add_argument('--output', type=Path, default=BASE/'snapshots')
    parser.add_argument('--html', type=Path)
    args = parser.parse_args()
    if args.checkpoint_calls <= 0: parser.error('--checkpoint-calls must be positive')
    print(json.dumps(run(args)), flush=True)
