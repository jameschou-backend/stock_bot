#!/usr/bin/env python3
"""Acquire only a frozen audit's missing full-market days, with resumable receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skills.market_input_validation import require
from skills.official_daily_acquisition import (
    OfficialDailyAcquisition, create_plan, request_item,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize(report, cache):
    report = report.resolve()
    require(report.is_relative_to(ROOT), 'Audit report must remain inside the repository')
    sidecar = report.with_suffix('.sha256')
    require(digest(report) == sidecar.read_text().strip(), 'Audit report hash differs')
    value = json.loads(report.read_text())
    require(value['schema'] in ('market_input_validation_v1', 'market_input_validation_v2')
            and value['live_qualified'] is False, 'Unsupported audit report')
    items = [request_item(row['market'], row['date']) for row in value['request_plan']
             if row['status'] == 'source_missing']
    require(len(items) == value['requests_lower_bound'], 'Missing-day count differs')
    create_plan(ROOT, cache, items, source_sha256={
        str(p.relative_to(ROOT)): digest(p) for p in (report, sidecar)})
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--mode', choices=('plan', 'probe', 'fetch', 'export'), default='plan')
    parser.add_argument('--market', choices=('TWSE', 'TPEX'))
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--user-request', help='Exact current user request; never an access-control bypass grant')
    parser.add_argument('--proof', type=Path, help='Fresh successful exact-endpoint probe receipt')
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    require(1 <= args.limit <= 4000, 'Request limit must be 1..4000')
    cache = args.cache.resolve()
    require(cache.is_relative_to(ROOT), 'Cache must remain inside the repository')
    items = initialize(args.report, cache)
    if args.mode == 'plan':
        print(json.dumps(dict(planned_missing_days=len(items), network_requests=0)))
        return
    require(args.mode == 'export' or args.market is not None, 'Choose one market per acquisition process')
    authorization = cache/'authorization.json'
    if args.mode == 'probe':
        require(bool(args.user_request and args.user_request.strip()), 'Current scoped user request is required')
        if not authorization.exists():
            value = dict(schema='official_daily_authorization_v1',
                         user_request=args.user_request, scope='missing_official_daily_tables',
                         security_bypass_authorized=False,
                         created_at=datetime.now(timezone.utc).isoformat(),
                         plan_sha256=digest(cache/'plan.json'))
            with authorization.open('x') as stream:
                stream.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        else:
            require(json.loads(authorization.read_text())['user_request'] == args.user_request,
                    'Preserve the original authorization record')
    proofs = {args.market: args.proof} if args.proof else None
    client = OfficialDailyAcquisition(ROOT, cache,
        authorization_path=authorization if authorization.exists() else None,
        recovery_proofs=proofs)
    selected = [item for item in items if item['market'] == args.market]
    try:
        if args.mode == 'probe':
            require(bool(selected), 'No missing dates for this market')
            result = client.probe(selected[0], allow_probe=True)
            print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
            if not result['accepted']:
                raise SystemExit(2)
        elif args.mode == 'fetch':
            pending = [item for item in selected if not (
                (cache/'attempts'/(item['identity']+'.json')).exists()
                or (cache/'receipts'/(item['identity']+'.json')).exists())]
            for item in pending[:args.limit]:
                result = client.fetch(item)
                print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
                if not result['accepted']:
                    raise SystemExit(2)
    finally:
        if args.manifest:
            client.export_manifest(args.manifest)


if __name__ == '__main__':
    main()
