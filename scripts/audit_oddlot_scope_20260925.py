#!/usr/bin/env python3
"""Offline inventory of odd-lot evidence required by three sealed accounts.

This extracts the realized daily-model path, not the unknown path that would
result from sequential auctions. It never converts a daily quote into a tape.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.cache/conservative-diversification-20260924'
OUTPUT = ROOT / '.cache/oddlot_scope_20260925'
CASES = ('capacity_combined_3', 'capacity_combined_5', 'benchmark_combined_0')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def demand(account, case):
    """Keep requests, executions, and rejected quantities distinct."""
    groups = {}
    for kind in ('orders', 'trades'):
        for index, row in enumerate(account[kind]):
            if row['channel'] != 'odd':
                continue
            key = row['date'], row['stock_id'], row['side']
            target = groups.setdefault(key, dict(case=case, date=key[0], stock_id=key[1], side=key[2],
                order_count=0, trade_count=0, requested_shares=0, filled_shares=0,
                gross_twd=0., order_indices=[], trade_sequences=[], failures=[]))
            if kind == 'orders':
                target['order_count'] += 1
                target['requested_shares'] += row['requested_qty']
                target['order_indices'].append(index)
                if row.get('failure'):
                    target['failures'].append(row['failure'])
            else:
                if not isinstance(row['qty'], int) or row['qty'] <= 0:
                    raise ValueError('Executed odd quantity must be a positive integer')
                target['trade_count'] += 1
                target['filled_shares'] += row['qty']
                target['gross_twd'] += row['gross']
                target['trade_sequences'].append(row['sequence'])
    for row in groups.values():
        if not row['order_count'] or row['filled_shares'] > row['requested_shares']:
            raise ValueError('Odd trade/request ledger mismatch')
        row['gross_twd'] = round(row['gross_twd'], 2)
    return [groups[key] for key in sorted(groups)]


class DailyEvidence:
    """Read only files authenticated by the sealed account manifest and index."""
    def __init__(self, source, manifest):
        self.root = source / 'inputs/execution-feeds'
        self.source, self.manifest, self.used, self.memo = source, manifest, {}, {}
        self.index = self.checked(self.root / 'index.json', index=False)

    def checked(self, path, index=True):
        name = str(path.relative_to(self.source))
        digest = sha(path)
        if self.manifest.get(name) != digest:
            raise ValueError('Sealed daily evidence changed: ' + name)
        if index and self.index['files_sha256'].get(path.name) != digest:
            raise ValueError('Daily index hash mismatch: ' + name)
        self.used[str(path.relative_to(ROOT))] = digest
        return read(path)

    def get(self, day, sid):
        matches = []
        for market in ('twse', 'tpex'):
            key = f'odd:{market}:{day}'
            entry = self.index['entries'].get(key)
            if not entry:
                continue
            if key not in self.memo:
                raw_path = self.root / entry['raw_file']
                self.checked(raw_path)
                normalized = self.checked(self.root / entry['rows_file'])
                if normalized['raw_sha256'] != sha(raw_path):
                    raise ValueError('Raw/normalized evidence mismatch: ' + key)
                self.memo[key] = normalized['rows']
            row = self.memo[key].get(sid)
            if row:
                if row['source_date'] != day or row['market'] != market:
                    raise ValueError('Daily row identity mismatch')
                matches.append(dict(market=market.upper(), daily_record_present=True,
                    daily_volume_shares=row['odd_shares'], last_bid=row['odd_bid'], last_ask=row['odd_ask'],
                    last_bid_shares=row['bid_qty'], last_ask_shares=row['ask_qty'],
                    daily_raw_path=str((self.root/entry['raw_file']).relative_to(ROOT)),
                    daily_rows_path=str((self.root/entry['rows_file']).relative_to(ROOT))))
        if len(matches) > 1:
            raise ValueError('Ambiguous historical market: ' + sid + ' ' + day)
        return matches[0] if matches else dict(market='UNKNOWN', daily_record_present=False)


def fingerprint_audit(identity):
    def check(item):
        name, expected = item
        path = ROOT / name
        actual = sha(path) if path.is_file() else None
        return None if actual == expected else dict(path=name, expected=expected, actual=actual)
    with ThreadPoolExecutor(max_workers=4) as pool:
        changed = [row for row in pool.map(check, sorted(identity.items())) if row]
    return dict(files_checked=len(identity), all_match=not changed, changed=changed,
        note='Fingerprint comparison only; the historical execution engine was not replayed.')


def csv_bytes(rows):
    out = io.StringIO(newline='')
    columns = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(out, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict))
                         else value for key, value in row.items()})
    return out.getvalue().encode('utf-8')


def build(source=SOURCE):
    manifest = read(source/'manifest.json')['files_sha256']
    inputs = {str((source/'manifest.json').relative_to(ROOT)): sha(source/'manifest.json')}
    for name in ('identity.json', *(f'cases/{case}.json' for case in CASES)):
        path = source/name
        if manifest[name] != sha(path):
            raise ValueError('Sealed source changed: ' + name)
        inputs[str(path.relative_to(ROOT))] = sha(path)
    evidence = DailyEvidence(source, manifest)
    all_rows, summaries = [], {}
    for case in CASES:
        saved = read(source/'cases'/f'{case}.json')
        if not saved['completed']:
            raise ValueError('Demand extraction requires completed accounts')
        rows = demand(saved['account'], case)
        for row in rows:
            row.update(evidence.get(row['date'], row['stock_id']))
            row.update(accepted_sequence_tape_present=False, sequence_status='not_declared_in_sealed_sources')
        all_rows.extend(rows)
        filled = [row for row in rows if row['filled_shares']]
        summaries[case] = dict(total_return=saved['summary']['total_return'],
            all_trade_count=len(saved['account']['trades']), odd_order_count=sum(r['order_count'] for r in rows),
            odd_trade_count=sum(r['trade_count'] for r in rows), odd_filled_shares=sum(r['filled_shares'] for r in rows),
            odd_filled_stock_dates=len({(r['stock_id'], r['date']) for r in filled}),
            odd_filled_dates=len({r['date'] for r in filled}), odd_filled_stocks=len({r['stock_id'] for r in filled}),
            odd_filled_by_market=dict(Counter(r['market'] for r in filled)),
            odd_filled_by_side=dict(Counter(r['side'] for r in filled)),
            requested_stock_date_sides=len(rows), daily_missing=sum(not r['daily_record_present'] for r in rows),
            sequence_tapes_declared=0)
    union = defaultdict(list)
    for row in all_rows:
        union[(row['market'], row['date'], row['stock_id'], row['side'])].append(row)
    needed = [dict(market=k[0], date=k[1], stock_id=k[2], side=k[3],
        cases=[r['case'] for r in rows], filled_cases=[r['case'] for r in rows if r['filled_shares']],
        accepted_sequence_tape_present=False, daily_record_present=all(r['daily_record_present'] for r in rows))
        for k, rows in sorted(union.items())]
    strict = {}
    strict_root = ROOT/'.cache/crossday-contingent-20260914-verified'
    for case in ('strategy', 'benchmark'):
        path = strict_root/f'{case}.json'
        saved = read(path); inputs[str(path.relative_to(ROOT))] = sha(path)
        strict[case] = dict(completed=saved['completed'], completed_sessions=len(saved['daily']),
            blocked=saved['blocked'], total_return=saved['total_return'])
    identity = read(strict_root/'identity.json')
    inputs[str((strict_root/'identity.json').relative_to(ROOT))] = sha(strict_root/'identity.json')
    if identity.get('external_manifest') is not None:
        raise ValueError('Strict external manifest changed; review declared sequence tapes')
    sample_path = ROOT/'.cache/execution-sources-20260924-v3/mth-sample-inspection.json'
    sample = read(sample_path); inputs[str(sample_path.relative_to(ROOT))] = sha(sample_path)
    if sample['execution_tape_accepted'] or sample['historical_session_complete'] or sample['source_authenticated']:
        raise ValueError('MTH sample evidence changed; review separately')
    inputs.update(evidence.used)
    filled_union = [row for row in needed if row['filled_cases']]
    report = dict(scope='sealed_daily_model_oddlot_dependency', live_qualified=False, network_calls=0,
        start='2022-01-03', end='2026-09-09', cases=summaries,
        union=dict(requested_stock_date_sides=len(needed), filled_stock_date_sides=len(filled_union),
            filled_stock_dates=len({(r['market'],r['date'],r['stock_id']) for r in filled_union}),
            requested_stock_dates=len({(r['market'],r['date'],r['stock_id']) for r in needed}),
            requested_dates=len({r['date'] for r in needed}), filled_dates=len({r['date'] for r in filled_union}),
            filled_by_market=dict(Counter(r['market'] for r in filled_union)),
            sequence_tapes_declared=0), strict_first_blockers=strict,
        mth_sample={key:sample[key] for key in ('dates','stock_ids','historical_session_complete',
            'execution_tape_accepted','source_authenticated')},
        source_fingerprint_audit=fingerprint_audit(read(source/'identity.json')),
        limitations=[
            'Demand is the realized path of sealed daily-model accounts, not a complete request list for an unobserved sequential replay.',
            'Accounts are independent alternatives; requested/filled shares across accounts are not one portfolio or a purchase order.',
            'Daily volume, final quote and final depth do not establish auction timestamps, sequence, executable depth, or order priority.',
            'The strict historical run stops at the first missing session; its three earliest missing securities do not measure the full gap.',
            'No accepted odd sequence tape is declared in the inspected sealed inputs. This is not an exhaustive search of arbitrary user files.',
            'The 2023-09-15 3105 MTH sample is a format diagnostic and is not accepted as a complete authenticated tape.',
            'Market comes from a dated official daily record; this does not independently resolve every historical listing or transfer issue.'])
    return report, all_rows, needed, inputs


def run(output=OUTPUT, verify=False):
    report, rows, needed, inputs = build()
    products = {'summary.json': encoded(report), 'exact_demands.json': encoded(rows),
                'exact_demands.csv': csv_bytes(rows), 'union_stock_date_side.json': encoded(needed),
                'union_stock_date_side.csv': csv_bytes(needed)}
    inputs[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    seal = dict(input_files_sha256=inputs,
        output_files_sha256={name: hashlib.sha256(data).hexdigest() for name, data in products.items()})
    products['manifest.json'] = encoded(seal)
    if verify:
        if any(not (output/name).is_file() or (output/name).read_bytes() != data for name,data in products.items()):
            raise ValueError('Offline odd-lot demand extraction differs')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Output is immutable; verify or choose a new directory')
        output.mkdir(parents=True, exist_ok=True)
        for name, data in products.items():
            (output/name).write_bytes(data)
    print(json.dumps(dict(cases=report['cases'], union=report['union'],
        source_fingerprints=report['source_fingerprint_audit'], verified=verify), ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    run(args.output, args.verify)
