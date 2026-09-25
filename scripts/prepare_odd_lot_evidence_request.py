#!/usr/bin/env python3
"""Create a provider-ready odd-lot request from the sealed 20-account audit.

Offline only. Daily rows identify the requested exchange; they never become
auction sequences or certify execution. This script neither buys nor downloads.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.backtest_data_evidence import verify_report  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + '\n').encode()


def build(report_path, identity_overlay_path=None):
    report = verify_report(report_path, ROOT)
    fingerprints = {str(report_path.relative_to(ROOT)): sha(report_path)}
    overlay = None
    overlay_reference = None
    if identity_overlay_path is not None:
        from scripts.audit_historical_universe_followup import verify_report as verify_identity
        identity_overlay_path = Path(identity_overlay_path).resolve()
        if not identity_overlay_path.is_relative_to(ROOT):
            raise ValueError('Identity overlay must stay inside the repository')
        overlay = verify_identity(identity_overlay_path)
        overlay_reference = dict(path=str(identity_overlay_path.relative_to(ROOT)),
                                 sha256=sha(identity_overlay_path))
        fingerprints[overlay_reference['path']] = overlay_reference['sha256']
        fingerprints[str(identity_overlay_path.with_suffix('.sha256').relative_to(ROOT))] = sha(identity_overlay_path.with_suffix('.sha256'))
        fingerprints.update(overlay['source_sha256'])
    required = defaultdict(set)
    for name, case in report['cases'].items():
        for row in case['odd_lot']['missing']:
            required[(row['date'], row['stock_id'])].add(name)
    if len(required) != report['coverage_totals']['odd_lot']['missing_unique_sessions']:
        raise ValueError('Missing-session union disagrees with the sealed audit')
    source_hashes = {}
    for base in sorted({(ROOT / item['path']).parent.parent for item in report['case_sources'].values()}):
        manifest_path = base / 'manifest.json'
        key = str(manifest_path.relative_to(ROOT))
        if sha(manifest_path) != report['input_sha256'][key]:
            raise ValueError('Case manifest changed: ' + key)
        manifest = json.loads(manifest_path.read_text())['files_sha256']
        identity_path = base / 'identity.json'
        if sha(identity_path) != manifest['identity.json']:
            raise ValueError('Case identity changed: ' + str(identity_path))
        fingerprints[key] = sha(manifest_path)
        fingerprints[str(identity_path.relative_to(ROOT))] = sha(identity_path)
        identity = json.loads(identity_path.read_text())
        for name, digest in identity.get('source_sha256', identity).items():
            if name in source_hashes and source_hashes[name] != digest:
                raise ValueError('Conflicting sealed source fingerprint: ' + name)
            source_hashes[name] = digest
    candidates = defaultdict(list)
    days = {day for day, _ in required}
    for name in source_hashes:
        path = Path(name)
        for market in ('twse', 'tpex'):
            prefix = f'odd-{market}-'
            if path.name.startswith(prefix) and path.name.endswith('.rows.json'):
                day = path.name[len(prefix):-len('.rows.json')]
                if day in days:
                    candidates[(day, market)].append(name)
    dated = {}
    for (day, market), paths in sorted(candidates.items()):
        # Prefer the latest sector preparation; all choices remain hash-bound.
        paths.sort(key=lambda p: ('sector-account-sources-r2-' not in p, p))
        for name in paths:
            path = ROOT / name
            raw_name = name[:-len('.rows.json')] + '.raw.json'
            if raw_name not in source_hashes:
                continue
            if sha(path) != source_hashes[name] or sha(ROOT / raw_name) != source_hashes[raw_name]:
                raise ValueError('Dated daily evidence changed: ' + name)
            payload = json.loads(path.read_text())
            if payload['raw_sha256'] != source_hashes[raw_name]:
                raise ValueError('Daily row/raw mismatch: ' + name)
            fingerprints[name] = sha(path)
            fingerprints[raw_name] = sha(ROOT / raw_name)
            dated[(day, market)] = (payload['rows'], name)
            break
    requests = []
    for (day, sid), cases in sorted(required.items()):
        matches = []
        for market in ('twse', 'tpex'):
            rows, path = dated.get((day, market), ({}, None))
            if sid in rows:
                row = rows[sid]
                if row['source_date'] != day or row['market'] != market:
                    raise ValueError('Dated daily row identity mismatch')
                matches.append((market.upper(), path))
        if len(matches) > 1:
            raise ValueError(f'Ambiguous dated exchange for {sid}/{day}')
        market, path = matches[0] if matches else ('UNKNOWN', None)
        identity_resolution = None
        if not matches and overlay is not None:
            from skills.historical_universe_followup import resolve_followup
            identity_resolution = resolve_followup(overlay, sid, day)
            resolved_market = str(identity_resolution.get('market', '')).upper()
            if identity_resolution['status'] == 'identified' and resolved_market in ('TWSE', 'TPEX'):
                market = resolved_market
        requests.append(dict(date=day, stock_id=sid, market=market,
            channel='intraday_odd_lot', cases=sorted(cases), daily_market_evidence=path,
            daily_row_missing=not bool(matches), no_trades_unproven=not bool(matches),
            market_evidence_kind='dated_official_daily_row' if matches else (
                'dated_identity_overlay' if market != 'UNKNOWN' else 'unresolved'),
            identity_resolution=identity_resolution,
            complete_sequence_received=False, execution_certified=False))
    by_market = {}
    for market in sorted({r['market'] for r in requests}):
        selected = [r for r in requests if r['market'] == market]
        by_market[market] = dict(stock_days=len(selected),
            stocks=len({r['stock_id'] for r in selected}),
            dates=len({r['date'] for r in selected}),
            first_date=min(r['date'] for r in selected),
            last_date=max(r['date'] for r in selected),
            by_month=dict(sorted(Counter(r['date'][:7] for r in selected).items())))
    summary = dict(schema='odd_lot_provider_request_v1',
        purpose='supply_scope_inquiry_only_not_purchase_or_execution_proof',
        historical_auction_rows_acquired=0, accepted_sessions=0,
        finmind_requests=0, network_requests=0, strict_data_ready=False,
        total_stock_days=len(requests), stocks=len({r['stock_id'] for r in requests}),
        by_market=by_market, required_fields=report['required_tape_fields'],
        known_daily_paths_only=True, all_possible_replay_paths_covered=False,
        source_report=str(report_path.relative_to(ROOT)),
        identity_overlay=overlay_reference,
        daily_row_missing_stock_days=sum(r['daily_row_missing'] for r in requests),
        dated_identity_resolved_missing_daily_stock_days=sum(r['market_evidence_kind'] == 'dated_identity_overlay' for r in requests),
        market_evidence_limitation='Dated daily row or verified identity overlay is market routing only; neither proves no trades or complete historical eligibility.',
        input_sha256=fingerprints)
    twse_months = by_market.get('TWSE', {}).get('by_month', {})
    purchasable = [m for m in twse_months if m <= '2026-07']
    deferred = [m for m in twse_months if m > '2026-07']
    tpex = [r for r in requests if r['market'] == 'TPEX']
    summary['procurement_plan'] = dict(as_of='2026-09-25', purchase_authorized=False,
        supplier_messages_sent=False, quote_confirmed=False,
        twse_h4=dict(product_url='https://eshop.twse.com.tw/zh/product/detail/0000000080da7fa70182334eb932009d',
            pricing_basis='Previously sealed official H4 page retrieved 2026-09-25; not a confirmed supplier quotation.',
            delivery_unit='all_stocks_per_month', internal_use_list_price_twd_per_month=1500,
            required_months=list(twse_months), months_within_published_age_window=purchasable,
            months_outside_published_age_window=deferred,
            candidate_list_price_twd=len(purchasable) * 1500,
            candidate_stock_days=sum(twse_months[m] for m in purchasable),
            outside_age_window_stock_days=sum(twse_months[m] for m in deferred),
            source_product_format_change_date='2026-04-01',
            taxes_discounts_delivery_and_current_availability_confirmed=False),
        tpex_mth=dict(product_url='https://eshop.tpex.org.tw/zh/product/detail/2c92e0139984eab70199892c78bf0004',
            published_start='2022-11-01', published_minimum_age='one_year',
            older_than_window_and_after_published_start=sum('2022-11-01' <= r['date'] < '2025-09-25' for r in tpex),
            before_published_start=sum(r['date'] < '2022-11-01' for r in tpex),
            within_last_year=sum(r['date'] >= '2025-09-25' for r in tpex),
            exact_boundary_and_price_require_supplier_confirmation=True),
        unknown_market_stock_days=by_market.get('UNKNOWN', {}).get('stock_days', 0))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=['market', 'date', 'stock_id', 'channel'], lineterminator='\n')
    writer.writeheader()
    writer.writerows({k: r[k] for k in writer.fieldnames} for r in requests)
    return {'request.json': encode(summary), 'stock_days.json': encode(requests),
            'stock_days.csv': out.getvalue().encode()}


def artifacts(report_path, identity_overlay_path=None):
    products = build(report_path, identity_overlay_path)
    products['manifest.json'] = encode(dict(files_sha256={name: hashlib.sha256(data).hexdigest()
        for name, data in products.items()}, code_sha256=sha(Path(__file__))))
    return products


def verify_request(directory):
    """Recompute the hash-bound local demand and reject stale/modified exports."""
    directory = Path(directory)
    request = json.loads((directory / 'request.json').read_text())
    report_path = (ROOT / request['source_report']).resolve()
    if not report_path.is_relative_to(ROOT):
        raise ValueError('Request source must stay inside the repository')
    overlay = request.get('identity_overlay')
    overlay_path = (ROOT / overlay['path']).resolve() if overlay else None
    if overlay_path is not None and (not overlay_path.is_relative_to(ROOT) or sha(overlay_path) != overlay['sha256']):
        raise ValueError('Recorded identity overlay changed')
    for name, data in artifacts(report_path, overlay_path).items():
        if (directory / name).read_bytes() != data:
            raise ValueError('Reproduction differs: ' + name)
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'artifacts/forward_simulation/backtest_data_completion_20260925.json')
    parser.add_argument('--output', type=Path, default=ROOT / '.cache/odd-lot-provider-request-20260925-v3')
    parser.add_argument('--identity-overlay', type=Path, help='Explicit verified dated-market overlay; never implies trades')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        if args.identity_overlay:
            parser.error('--verify replays the archived overlay; do not supply another overlay')
        summary = verify_request(args.output)
    else:
        products = artifacts(args.report.resolve(), args.identity_overlay)
        args.output.mkdir(parents=True, exist_ok=False)
        for name, data in products.items():
            (args.output / name).write_bytes(data)
        summary = json.loads(products['request.json'])
    print(json.dumps(dict(total_stock_days=summary['total_stock_days'],
        stocks=summary['stocks'], stock_days_by_market={market: item['stock_days']
            for market, item in summary['by_market'].items()},
        historical_auction_rows_acquired=0, verified=args.verify), ensure_ascii=False))


if __name__ == '__main__':
    main()
