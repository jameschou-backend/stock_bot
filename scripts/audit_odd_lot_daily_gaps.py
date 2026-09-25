#!/usr/bin/env python3
"""Acquire and verify the 13 explicitly unresolved odd-lot daily records.

This closes daily evidence gaps only. It never creates auction ticks, enables
execution, or edits the historical strategy's sealed input directories.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_odd_lot_evidence_request import verify_request
from skills.replay_market_feeds import ReplayMarketFeeds, parse_odd

REQUEST = ROOT / '.cache/odd-lot-provider-request-20260925-v3'
OUTPUT = ROOT / '.cache/odd-lot-daily-gaps-20260925'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def demand(directory):
    request = verify_request(directory)
    rows = json.loads((directory / 'stock_days.json').read_bytes())
    selected = [r for r in rows if r['daily_row_missing']]
    if len(selected) != 13 or len({(r['market'], r['date']) for r in selected}) != 8:
        raise ValueError('Only the reviewed 13 stock-days / 8 market dates are authorized by this recipe')
    if any(r['market'] not in ('TWSE', 'TPEX') for r in selected):
        raise ValueError('Every requested daily table needs verified market routing')
    return request, selected


def build(directory, request_directory):
    request, selected = demand(request_directory)
    feeds = ReplayMarketFeeds(directory / 'feeds', offline=True)
    manifest = feeds.manifest()
    if manifest['request_counters']['official_http_requests'] != 8 or manifest['finmind_requests_upper_bound']:
        raise ValueError('Unexpected market-data request count')
    refs = dict(request['input_sha256'])
    for name in ('request.json', 'stock_days.json', 'stock_days.csv', 'manifest.json'):
        p = request_directory / name
        refs[str(p.relative_to(ROOT))] = sha(p)
    refs[str((directory/'feeds/index.json').relative_to(ROOT))] = sha(directory/'feeds/index.json')
    for name, value in manifest['files_sha256'].items():
        refs[str((directory/'feeds'/name).relative_to(ROOT))] = value
    results = []
    for item in selected:
        day, sid, market = item['date'], item['stock_id'], item['market'].lower()
        name = f'odd-{market}-{day}.raw.json'
        raw = json.loads((directory/'feeds'/name).read_bytes())
        parsed = parse_odd(raw, market, day)
        row = feeds.get_odd(day, sid, market)
        if parsed.get(sid) != row:
            raise ValueError('Raw daily parsing differs from stored rows')
        status = 'stock_row_absent_unproven' if row is None else (
            'official_daily_zero_trades' if row['odd_shares'] == 0 else 'official_daily_positive_trade')
        results.append(dict(date=day, stock_id=sid, market=market.upper(), status=status,
            official_daily_row=row, daily_row_missing=row is None,
            no_trades_confirmed_by_daily_record=bool(row is not None and row['odd_shares'] == 0),
            trading_suspension_proven=False, historical_auction_sequence_acquired=False,
            accepted_for_strict_replay=False,
            source_path=str((directory/'feeds'/name).relative_to(ROOT)),
            source_sha256=sha(directory/'feeds'/name)))
    code = ['scripts/audit_odd_lot_daily_gaps.py', 'scripts/prepare_odd_lot_evidence_request.py',
            'skills/replay_market_feeds.py']
    return dict(schema='odd_lot_daily_gap_evidence_v1',
        request_directory=str(request_directory.relative_to(ROOT)),
        required_stock_days=13, required_market_dates=8,
        statuses=dict(Counter(r['status'] for r in results)), rows=results,
        preparation_official_requests=8, finmind_requests=0, verification_network_requests=0,
        historical_auction_rows_acquired=0, accepted_sequence_sessions=0,
        strict_data_ready=False, live_qualified=False,
        input_sha256=refs, code_sha256={p:sha(ROOT/p) for p in code})


def verify_report(path):
    path = Path(path).resolve()
    if sha(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Daily-gap report SHA mismatch')
    report = json.loads(path.read_bytes())
    if report['schema'] != 'odd_lot_daily_gap_evidence_v1':
        raise ValueError('Unsupported daily-gap report')
    for mapping in ('input_sha256', 'code_sha256'):
        for name, value in report[mapping].items():
            p = (ROOT/name).resolve()
            if not p.is_relative_to(ROOT) or sha(p) != value:
                raise ValueError('Daily-gap source changed: ' + name)
    rebuilt = build(path.parent, ROOT/report['request_directory'])
    if report != rebuilt:
        raise ValueError('Daily-gap offline replay differs')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--request-directory', type=Path, default=REQUEST)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.fetch == args.verify:
        parser.error('Choose exactly one of --fetch or --verify')
    output, request_directory = args.output.resolve(), args.request_directory.resolve()
    if not output.is_relative_to(ROOT/'.cache'):
        raise ValueError('Use a new directory within the project cache')
    if args.verify:
        result = verify_report(output/'report.json')
    else:
        _, selected = demand(request_directory)
        output.mkdir(parents=True, exist_ok=False)
        (output/'budget.json').write_bytes(encoded(dict(maximum_official_requests=8,
            planned_market_dates=sorted({(r['market'],r['date']) for r in selected}))))
        feeds = ReplayMarketFeeds(output/'feeds', official_min_interval=5)
        # get_odd reuses the market-date response for stocks sharing a date.
        # Any failure stops the run; an existing directory cannot be fetched again.
        for row in selected:
            feeds.get_odd(row['date'], row['stock_id'], row['market'])
        result = build(output, request_directory)
        path = output/'report.json'
        path.write_bytes(encoded(result))
        path.with_suffix('.sha256').write_text(sha(path)+'\n')
        verify_report(path)
    print(json.dumps(dict(statuses=result['statuses'], verified=args.verify,
        official_requests=0 if args.verify else result['preparation_official_requests'],
        historical_auction_rows_acquired=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
