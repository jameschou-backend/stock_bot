#!/usr/bin/env python3
"""Join independently reconstructed daily POC onto a sealed account explorer.

No data acquisition, strategy execution, or account-result changes occur here.
"""
import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.export_poc_signal_explorer import merge_refs, verify_refs
from scripts.export_signal_explorer import digest, json_for_script, render_html

BASE = '.cache/poc-latest-20261003/explorer-v2'
BASE_PAYLOAD_SHA = '0faa0d005c492d3a58293be0d10cee3d84bdd6cacba759953ba3bc5f653f87c8'
BASE_RECEIPT_SHA = '2014462759d7adddd3410bf2305b1518456cd9b9e01467b871ec7f1ac28c0a5e'
STATUSES = ('up', 'down', 'unknown', 'pending_data')
ARCHIVE = 'artifacts/reports/signal_explorer_account_20261002.html'


def read(path):
    def invalid(value):
        raise ValueError('Non-finite JSON value: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def bound_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Source or output escapes repository: ' + str(name))
    return path


def atomic_write(path, content):
    """Readers observe either the previous complete file or the next complete file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix='.' + path.name + '.',
                                         suffix='.tmp', dir=path.parent, delete=False) as stream:
            pending = Path(stream.name)
            stream.write(content if isinstance(content, bytes) else content.encode('utf-8'))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        if pending is not None and pending.exists():
            pending.unlink()


def load_daily(report_path, root=ROOT):
    root = Path(root).resolve()
    path = bound_path(root, Path(report_path))
    report = read(path)
    if (report.get('schema') != 'poc_daily_profiles_v1'
            or report.get('all_signals_materialized') is not True):
        raise ValueError('Require a complete daily-profile opportunity ledger')
    if report.get('year') != 2026 or report.get('end') != '2026-10-02':
        raise ValueError('Daily-profile report must cover 2026 through 2026-10-02')
    sidecar = path.with_suffix('.sha256')
    if sidecar.read_text().strip() != digest(path):
        raise ValueError('Daily-profile report SHA differs')
    refs = dict(report['source_sha256'])
    merge_refs(refs, {str(path.relative_to(root)): digest(path),
                      str(sidecar.relative_to(root)): digest(sidecar)})
    for key, name, expected in (
            ('base_payload', BASE + '/payload.json', BASE_PAYLOAD_SHA),
            ('base_receipt', BASE + '/receipt.json', BASE_RECEIPT_SHA)):
        item = report[key]
        if item.get('path') != name or item.get('sha256') != expected:
            raise ValueError('Daily profiles refer to another sealed account explorer')
        merge_refs(refs, {name: expected})
    base_receipt = read(bound_path(root, report['base_receipt']['path']))
    if (base_receipt.get('schema') != 'poc_latest_explorer_receipt_v1'
            or base_receipt['output_sha256'].get(BASE + '/payload.json') != BASE_PAYLOAD_SHA):
        raise ValueError('Account explorer payload is not bound by its receipt')
    merge_refs(refs, base_receipt['source_sha256'])
    item = report['profiles']
    merge_refs(refs, {item['path']: item['sha256']})
    verify_refs(refs, root)
    payload = read(bound_path(root, report['base_payload']['path']))
    profiles = read(bound_path(root, item['path']))
    if not isinstance(profiles, list):
        raise ValueError('Daily profiles must contain one list row per signal')
    return payload, profiles, report, base_receipt, refs


def _utc(value):
    if not isinstance(value, str):
        raise ValueError('Every daily record needs a UTC computed_at timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError('computed_at must be timezone-aware UTC')


def build_payload(base, profiles):
    """Strict identity/calendar join. Account dictionaries remain untouched."""
    signals = base['signals']
    if not signals or len({s['signal_id'] for s in signals}) != len(signals):
        raise ValueError('Base signals must have unique identities')
    if 'opportunities' in base:
        raise ValueError('Do not silently replace an existing opportunity study')
    end = base['metadata']['date_end']
    index = {}
    for row in profiles:
        key = row['signal_id']
        if key in index:
            raise ValueError('Duplicate daily-profile signal identity')
        index[key] = row
    if set(index) != {s['signal_id'] for s in signals}:
        raise ValueError('Daily profiles must match every original signal exactly')
    dates_by_stock = {}
    date_column = base['price_columns'].index('date')
    opportunities = {}
    for s in signals:
        sid, day, key = s['stock_id'], s['signal_date'], s['signal_id']
        if date.fromisoformat(day).year != 2026 or day > end:
            raise ValueError('Signal is outside the sealed 2026 scope')
        row = index[key]
        if row.get('stock_id') != sid or row.get('signal_date') != day:
            raise ValueError('Daily profile stock/date differs from signal')
        if row.get('account_independent') is not True or row.get('reconstructed') is not True:
            raise ValueError('Daily POC must explicitly be independent historical reconstruction')
        if sid not in dates_by_stock:
            prices = base['stocks'][sid]['prices']
            dates = [p[date_column] if isinstance(p, list) else p['date'] for p in prices]
            if dates != sorted(set(dates)) or any(d > end for d in dates):
                raise ValueError('Stock context has a noncanonical market calendar')
            dates_by_stock[sid] = dates
        dates = dates_by_stock[sid]
        if day not in dates:
            raise ValueError('Signal lacks its observed market session')
        position = dates.index(day)
        expected = dates[max(0, position - 20):position]
        if len(expected) != 20 or row.get('prior_dates') != expected:
            raise ValueError('POC window must be the exact 20 strictly prior market sessions')
        if any(row.get(k) != wanted for k, wanted in (
                ('window_start', expected[0]), ('window_end', expected[-1]),
                ('source_date_end', expected[-1]))):
            raise ValueError('POC window metadata disagrees with its source dates')
        status = row.get('status')
        if status not in STATUSES or type(row.get('available')) is not bool:
            raise ValueError('Unknown daily-profile status or availability')
        known = status in ('up', 'down')
        if row['available'] != known:
            raise ValueError('Daily-profile status/availability disagree')
        _utc(row.get('computed_at'))
        result = {k: row[k] for k in ('signal_id', 'stock_id', 'signal_date', 'status',
            'available', 'reason', 'prior_dates', 'window_start', 'window_end', 'source_date_end',
            'account_independent', 'reconstructed', 'computed_at')}
        if known:
            values = [row.get(k) for k in ('poc_before', 'poc_after')]
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError('Known POC needs two finite positive raw prices')
            if row['reason'] is not None or (values[1] > values[0]) != (status == 'up'):
                raise ValueError('Known POC direction or reason disagrees')
            result.update(poc_before=values[0], poc_after=values[1], poc_price_basis='raw')
        else:
            if not isinstance(row['reason'], str) or not row['reason'].strip():
                raise ValueError('Unavailable POC requires an explicit reason')
            if any(row.get(k) is not None for k in ('poc_before', 'poc_after')):
                raise ValueError('Unavailable POC must not invent price levels')
        opportunities[key] = result
    counts = Counter(r['status'] for r in opportunities.values())
    red_counts = Counter(opportunities[s['signal_id']]['status'] for s in signals if s['candle'] == 'red')
    # A shallow top-level copy avoids duplicating the large immutable candle/account arrays.
    result = dict(base)
    result.update(schema='poc_daily_signal_explorer_v1', opportunities=opportunities,
        opportunity_metadata=dict(schema='poc_daily_opportunities_v1', year=2026,
            date_start=base['metadata']['date_start'], date_end=end, signal_count=len(signals),
            status_counts={s: counts[s] for s in STATUSES},
            red_status_counts={s: red_counts[s] for s in STATUSES},
            all_signals_materialized=True, all_profiles_available=not(counts['unknown'] or counts['pending_data']),
            account_independent=True, reconstructed=True, historical_availability_certified=False,
            account_results_changed=False, account_archive_url=Path(ARCHIVE).name,
            limitations=['每日 POC 使用訊號前 20 個交易日，涵蓋所有原始候選，與三檔帳戶名額無關。',
                '這是事後重建的歷史指標；computed_at 是重建時間，不代表資料當時已公開或可取得。',
                '資料品質未知與原始資料待補分開標示，均不當作 POC 未上移。',
                '末日 POC 可計算；下一交易日、買入與成交仍未知。',
                '原有三檔帳戶的決策、交易、資產與績效完整保留，未以新指標重跑。']))
    return result


def archive_account_page(output, base_receipt, root=ROOT):
    """Preserve the old published account HTML before replacing that exact URL."""
    output = Path(output).resolve()
    active = root / 'artifacts/reports/signal_explorer_2026.html'
    if output != active or not output.exists():
        return None
    expected = base_receipt['output_sha256'].get(str(active.relative_to(root)))
    archive = root / ARCHIVE
    if archive.exists():
        if not expected or digest(archive) != expected:
            raise ValueError('Existing account archive differs from sealed HTML')
        return archive
    if not expected or digest(active) != expected:
        raise ValueError('Active HTML differs; preserve and review it before replacement')
    atomic_write(archive, active.read_bytes())
    return archive


def run(args, root=ROOT):
    root = Path(root).resolve()
    base, profiles, report, old_receipt, refs = load_daily(args.report, root)
    payload = build_payload(base, profiles)
    meta = payload['opportunity_metadata']
    meta['report_path'] = str(bound_path(root, args.report).relative_to(root))
    meta['report_sha256'] = digest(bound_path(root, args.report))
    for name in ('scripts/export_poc_daily_explorer.py', 'tests/test_poc_daily_explorer.py'):
        merge_refs(refs, {name: digest(root / name)})
    outputs = {}
    path = bound_path(root, args.payload)
    atomic_write(path, json_for_script(payload))
    outputs[str(path.relative_to(root))] = digest(path)
    if args.output:
        template = bound_path(root, args.template)
        merge_refs(refs, {str(template.relative_to(root)): digest(template)})
        dest = bound_path(root, args.output)
        archive = archive_account_page(dest, old_receipt, root)
        if archive:
            outputs[str(archive.relative_to(root))] = digest(archive)
        atomic_write(dest, render_html(template.read_text(), payload))
        outputs[str(dest.relative_to(root))] = digest(dest)
    receipt = dict(schema='poc_daily_explorer_receipt_v1', created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=refs, output_sha256=outputs, opportunity_metadata=meta,
        no_network=True, no_strategy_execution=True, account_results_changed=False)
    dest = bound_path(root, args.receipt)
    atomic_write(dest, json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    atomic_write(dest.with_suffix('.sha256'), digest(dest)+'\n')
    print(json.dumps(dict(signal_count=meta['signal_count'], status_counts=meta['status_counts'],
                         output_sha256=outputs), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output', help='Omit for payload and receipt only')
    parser.add_argument('--template', default='ui/poc_daily_signal_explorer.html')
    parser.add_argument('--payload', default='.cache/poc-daily-20261004/explorer/payload.json')
    parser.add_argument('--receipt', default='.cache/poc-daily-20261004/explorer/receipt.json')
    run(parser.parse_args())
