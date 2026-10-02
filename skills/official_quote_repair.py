"""Offline, source-bound OHLC supplements; volume and adjustment scopes stay explicit.

This module produces evidence, not a replacement strategy frame. In particular,
ordinary-session volume never fills daily-total volume, and an official raw
close never fills either adjusted-price column.
"""
from collections import defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from skills.market_input_validation import numeric, require
from skills.official_market_supplement import bound_path, sha, validate_entry


SCHEMA = 'official_quote_repair_evidence_v1'
SCOPES = ('all_daily_sessions', 'ordinary_session', 'unclassified_daily')
PRICE_FIELDS = ('open', 'high', 'low', 'close')


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def verify_document(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    require(path.is_relative_to(root), 'Evidence document escapes repository')
    require(sha(path) == path.with_suffix('.sha256').read_text().strip(), 'Evidence document hash changed')
    value = json.loads(path.read_text())
    refs = {str(path.relative_to(root)): sha(path)}
    for name, digest in value['source_sha256'].items():
        require(sha(bound_path(root, name)) == digest, 'Evidence source changed: ' + name)
        refs[name] = digest
    return value, refs


def row_key(row):
    require(isinstance(row, dict) and row.get('market') in ('TWSE', 'TPEX')
            and isinstance(row.get('date'), str)
            and date.fromisoformat(row['date']).isoformat() == row['date']
            and isinstance(row.get('stock_id'), str)
            and re.fullmatch(r'[0-9]{4}', row['stock_id']), 'Invalid stock/date/market identity')
    return row['market'], row['date'], row['stock_id']


def normalize_repair(target, observations):
    """Preserve every source scope; disagreements and partial OHLC fail closed."""
    key = row_key(target)
    require(target.get('positive_official_price') is True, 'Target is not an observed positive-price gap')
    require(isinstance(observations, list) and bool(observations), 'Missing official quote evidence')
    result, volumes, sources = None, {}, []
    for row in observations:
        require(row_key(row) == key, 'Repair observation identity mismatch')
        prices = {name: numeric(row.get(name)) for name in PRICE_FIELDS}
        require(all(v > 0 for v in prices.values()), 'Repair requires all four positive raw prices')
        require(prices['low'] <= min(prices['open'], prices['close'])
                <= max(prices['open'], prices['close']) <= prices['high'], 'Impossible repair OHLC')
        require(result is None or prices == result, 'Conflicting official repair prices')
        result = prices
        scope = row.get('volume_scope')
        require(scope in SCOPES, 'Unknown official repair volume scope')
        volume = numeric(row.get('volume'), integral=True)
        require(scope not in volumes or volumes[scope] == volume, 'Conflicting same-scope repair volume')
        volumes[scope] = volume
        require(isinstance(row.get('source_id'), str) and bool(row['source_id']), 'Missing repair source identity')
        sources.append(row['source_id'])
    if 'source_volume' in target:
        require(numeric(target['source_volume'], integral=True) in volumes.values(),
                'Observed gap volume differs from bound official evidence')
    total, ordinary = volumes.get('all_daily_sessions'), volumes.get('ordinary_session')
    require(total is None or ordinary is None or total >= ordinary, 'Daily total smaller than ordinary volume')
    return dict(market=key[0], date=key[1], stock_id=key[2], **result,
        # `volume` has the same daily-total meaning as quotes-unmasked.parquet.
        # Null is intentional and must be resolved before copying to that frame.
        volume=total, total_daily_volume=total, ordinary_session_volume=ordinary,
        unclassified_daily_volume=volumes.get('unclassified_daily'),
        source_volume_scopes=sorted(volumes), source_ids=sorted(set(sources)),
        raw_prices_verified=True, total_daily_volume_verified=total is not None,
        adjusted_close=None, adjusted_price_verified=False,
        ready_for_raw_quote_insert=total is not None)


def daily_request_plan(rows):
    """One provider stock range per missing-total group; exact required days retained."""
    by_stock = defaultdict(list)
    for row in rows:
        if row['total_daily_volume'] is None:
            by_stock[row['stock_id']].append(row['date'])
    return [dict(dataset='TaiwanStockPrice', data_id=sid, start_date=min(days), end_date=max(days),
                 required_dates=sorted(set(days)), fields=['Trading_Volume'],
                 reason='official_source_does_not_establish_daily_total_volume')
            for sid, days in sorted(by_stock.items())]


def prepare(root, audit_path, repair_path, output):
    """Normalize the complete published source set and isolate its exact quote gaps.

    Every original source and receipt is verified against the published audit,
    then independently parsed. Source IDs join to one descriptor manifest instead
    of repeating thousands of hashes on every stock row. Output is create-only.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), 'Choose a new repository evidence directory')
    audit, refs = verify_document(root, audit_path)
    require(audit['schema'] == 'market_input_validation_v2' and audit['live_qualified'] is False,
            'Unsupported audited source set')
    repair, repair_refs = verify_document(root, repair_path)
    require(repair['schema'] == 'market_input_repairs_v1' and repair['live_qualified'] is False,
            'Unsupported existing repair bundle')
    refs.update(repair_refs)
    for name, digest in repair['output_sha256'].items():
        require(sha(bound_path(root, name)) == digest, 'Existing repair output changed')
        refs[name] = digest
    quote_name = str((Path(repair_path).resolve().parent / 'quotes-unmasked.parquet').relative_to(root))
    require(quote_name in repair['output_sha256'], 'Existing quote frame is not bound by repair output')
    existing = pd.read_parquet(root / quote_name, columns=['date', 'stock_id'])
    existing_keys = set(zip(pd.to_datetime(existing.date).dt.strftime('%Y-%m-%d'), existing.stock_id))
    targets = audit['missing_positive_quotes_in_scope']
    require(isinstance(targets, list) and bool(targets), 'No explicit quote-gap targets')
    keys = [row_key(r) for r in targets]
    require(len(keys) == len(set(keys)), 'Duplicate quote-gap target')
    require(all((stamp, sid) not in existing_keys for market, stamp, sid in keys), 'Repair would overwrite an existing quote')
    del existing_keys, existing
    wanted = set(keys)
    target_observations = defaultdict(list)
    descriptors, seen_sources = {}, set()
    schema = pa.schema([(k, pa.string()) for k in ('source_id', 'market', 'date', 'stock_id', 'name')]
        + [(k, pa.float64()) for k in PRICE_FIELDS] + [('volume', pa.int64()),
           ('volume_scope', pa.string()), ('table_category', pa.string())])
    output.mkdir(parents=True)
    normalized = output / 'official-normalized.parquet'
    row_count = 0
    with pq.ParquetWriter(normalized, schema, compression='zstd') as writer:
        for descriptor in audit['sources']:
            raw_name, receipt_name = descriptor['path'], descriptor['receipt']
            require(raw_name in refs and receipt_name in refs and refs[raw_name] == descriptor['sha256'],
                    'Audited source or receipt not bound')
            if 'receipt_sha256' in descriptor:
                require(descriptor['receipt_sha256'] == refs[receipt_name], 'Audited receipt hash conflict')
            entry = dict(market=descriptor['market'], date=descriptor['date'], raw_path=raw_name,
                raw_sha256=refs[raw_name], receipt_path=receipt_name, receipt_sha256=refs[receipt_name])
            source_id = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
            require(source_id not in seen_sources, 'Duplicate official source descriptor')
            seen_sources.add(source_id)
            rows, verified = validate_entry(entry, root)
            require(verified['rows'] == descriptor['rows'] and verified['volume_scope'] == descriptor['volume_scope'],
                    'Normalized official source differs from audit')
            for name, digest in verified.get('recovery_source_sha256', {}).items():
                require(name in refs and refs[name] == digest, 'Unbound recovery source')
            descriptors[source_id] = verified
            records = []
            for row in rows.values():
                record = dict(row, source_id=source_id, table_category=row.get('table_category'))
                records.append(record)
                if row_key(row) in wanted:
                    target_observations[row_key(row)].append(record)
            writer.write_table(pa.Table.from_pylist(records, schema=schema))
            row_count += len(records)
    supplements = [normalize_repair(target, target_observations[row_key(target)])
                   for target in sorted(targets, key=row_key)]
    # JSON keeps explicit nulls and integer volume types; Parquet is for fast joins.
    (output / 'quote-supplement.json').write_text(encode(supplements))
    pd.DataFrame(supplements).to_parquet(output / 'quote-supplement.parquet', index=False)
    plan = dict(schema='official_quote_missing_fields_plan_v1', network_requests=0, finmind_requests=0,
        daily_total_requests=daily_request_plan(supplements),
        adjusted_price_status='pending_independent_snapshot_alignment',
        live_qualified=False)
    (output / 'missing-fields-plan.json').write_text(encode(plan))
    for name in ('skills/official_quote_repair.py', 'skills/official_market_supplement.py',
                 'skills/market_input_validation.py'):
        path = root / name
        if path.exists():
            require(name not in refs or refs[name] == sha(path), 'Code differs from audited dependency')
            refs[name] = sha(path)
    report = dict(schema=SCHEMA, created_at=datetime.now(timezone.utc).isoformat(), sources=descriptors,
        source_count=len(descriptors), market_day_count=len({(d['market'], d['date']) for d in descriptors.values()}),
        normalized_rows=row_count, required_quote_gaps=len(targets), raw_ohlc_repaired=len(supplements),
        total_daily_volume_verified=sum(r['total_daily_volume_verified'] for r in supplements),
        total_daily_volume_missing=sum(not r['total_daily_volume_verified'] for r in supplements),
        adjusted_price_verified=0, frozen_inputs_changed=False, database_mutations=False,
        full_account_replay_required=True, live_qualified=False, network_requests=0, finmind_requests=0,
        source_sha256=refs, output_sha256={str(p.relative_to(root)):sha(p) for p in output.iterdir() if p.is_file()})
    path = output / 'official-sources.json'
    path.write_text(encode(report))
    path.with_suffix('.sha256').write_text(sha(path) + '\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--repairs', type=Path, required=True, help='Existing repair report.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.root, args.audit, args.repairs, args.output)
    print(encode({k:v for k,v in result.items() if k not in ('sources','source_sha256','output_sha256')}))
