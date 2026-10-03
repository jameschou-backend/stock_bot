#!/usr/bin/env python3
"""Adapt the sealed October daily extension without revising the POC history.

This is an offline input adapter, not execution/corporate-action certification.
Pending last-close signals are kept separately from executable T+1 candidates.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from skills.repaired_execution_context import dated_market_resolver

BASE = ROOT / '.cache/market-input-repair-20261002/inputs-v2'
EXTENSION = ROOT / '.cache/all-signals-2019-20261002/inputs'
OUTPUT = ROOT / '.cache/poc-latest-20261003/inputs-v1'
ANCHOR = '2026-09-09'
OLD_SIGNAL_END = '2026-09-08'
END = '2026-10-02'
MATRICES = ('raw-close', 'raw-volume', 'close-official', 'close-quality', 'eligibility')
COPY_FILES = tuple(n + '.parquet' for n in MATRICES) + (
    'quotes-unmasked.parquet', 'companies.parquet', 'events.parquet',
    'identity.json', 'extension-coverage.json', 'signal-features.parquet')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bind(path, root, refs, expected=None):
    path = Path(path).resolve()
    require(path.is_relative_to(root), 'Input path escapes repository')
    actual = digest(path)
    require(expected is None or actual == expected, 'Sealed source changed: ' + str(path))
    name = str(path.relative_to(root))
    require(name not in refs or refs[name] == actual, 'Conflicting source hash: ' + name)
    refs[name] = actual
    return path


def verify_bundle(bundle, root, refs):
    manifest_path = bind(bundle / 'manifest.json', root, refs,
                         (bundle / 'manifest.sha256').read_text().strip())
    bind(bundle / 'manifest.sha256', root, refs)
    manifest = read(manifest_path)
    for name, expected in manifest['files_sha256'].items():
        bind(bundle / name, root, refs, expected)
    for name, expected in manifest.get('source_sha256', {}).items():
        bind(root / name, root, refs, expected)
    return manifest


def matrix_prefix(base, extension, name, anchor=ANCHOR, end=END):
    before, after = [pd.read_parquet(path / (name + '.parquet')).set_index('date')
                     for path in (base, extension)]
    for frame in (before, after):
        frame.index = pd.DatetimeIndex(frame.index)
        require(frame.index.is_unique and frame.index.is_monotonic_increasing
                and frame.columns.is_unique, 'Invalid matrix axes: ' + name)
    require(str(before.index[-1].date()) == anchor and str(after.index[-1].date()) == end,
            'Unexpected matrix endpoint: ' + name)
    require(before.columns.equals(after.columns), 'Stock universe changed: ' + name)
    require(after.loc[:anchor].equals(before), 'Historical matrix prefix changed: ' + name)
    return after.index


def partition_signals(original, extension, calendar, *, anchor=ANCHOR, end=END):
    """Reject changed history, duplicate identities and invented next sessions."""
    require(extension.get('prefix_exact') is True, 'Extension lacks prefix evidence')
    entries = extension['entries']
    require(len({e['event_id'] for e in entries}) == len(entries), 'Duplicate signal identity')
    require([e for e in entries if e['signal_date'] < anchor] == original,
            'Historical candidate prefix changed')
    days = [str(pd.Timestamp(d).date()) for d in calendar]
    positions = {day: index for index, day in enumerate(days)}
    executable, pending = [], []
    for event in entries:
        sid = event.get('members', [])
        require(len(sid) == 1 and len(sid[0]) == 4 and sid[0].isdigit()
                and not sid[0].startswith('0'), 'Only four-digit individual stocks are candidates')
        signal = event['signal_date']
        require(signal in positions, 'Signal is not a known market session')
        index = positions[signal]
        expected = days[index + 1] if index + 1 < len(days) else None
        require(event.get('entry_date') == expected, 'Signal entry differs from observed T+1')
        if expected is None:
            require(signal == end, 'Pending signal precedes final observed session')
            pending.append(event)
        else:
            executable.append(event)
    return executable, pending


def routing_audit(identity, entries, pending, calendar, *, anchor=ANCHOR, end=END):
    require(identity['coverage_end'] == end, 'Extended dated identity has wrong endpoint')
    observation = identity.get('extension_observation', {})
    require(observation.get('provisional') is True
            and observation.get('official_identity_verified_through') == anchor,
            'Extension must retain provisional identity disclosure')
    resolve = dated_market_resolver(identity)
    checks = {(e['entry_date'], e['members'][0]) for e in entries if e['entry_date'] > anchor}
    checks.update((e['signal_date'], e['members'][0]) for e in pending)
    checks.update((str(day.date()), '0050') for day in calendar if str(day.date()) > anchor)
    rows = []
    for day, sid in sorted(checks):
        market = resolve(day, sid)
        market = market.upper() if isinstance(market, str) else market
        require(market in ('TWSE', 'TPEX'), 'Missing extended execution route: ' + sid + '/' + day)
        rows.append(dict(date=day, stock_id=sid, market=market,
                         identity_basis='provisional_current_snapshot_not_daily_certification'))
    return rows


def merge_events(before, additions, *, anchor=ANCHOR, end=END):
    """Event observations are appended; historical records are never revised."""
    require(set(before.columns) == set(additions.columns), 'Corporate event columns differ')
    additions = additions.loc[:, before.columns].copy()
    stamps = pd.to_datetime(additions.event_date)
    require(stamps.gt(anchor).all() and stamps.le(end).all(), 'Event extension contains outside-period records')
    require(not before.duplicated(['stock_id', 'event_date']).any()
            and not additions.duplicated(['stock_id', 'event_date']).any(), 'Duplicate corporate event identity')
    require(additions.stock_id.map(lambda s: isinstance(s, str) and len(s) == 4 and s.isdigit()).all(),
            'Corporate event has invalid stock identity')
    # Preserve the parent's date representation and all other values exactly.
    if pd.api.types.is_datetime64_any_dtype(before.event_date):
        additions['event_date'] = stamps
    else:
        additions['event_date'] = stamps.dt.date
    result = pd.concat([before, additions], ignore_index=True)
    require(result.iloc[:len(before)].reset_index(drop=True).equals(before.reset_index(drop=True)),
            'Historical corporate event prefix changed')
    require(not result.duplicated(['stock_id', 'event_date']).any(), 'Duplicate corporate event identity')
    return result


def load_event_extension(path, root, refs):
    path = Path(path).resolve()
    sidecar = path.with_suffix('.sha256')
    report = read(bind(path, root, refs, sidecar.read_text().strip()))
    bind(sidecar, root, refs)
    require(report.get('corporate_events_extension_complete') is True
            and report.get('start') == '2026-09-10' and report.get('end') == END,
            'Corporate extension is not complete for the requested period')
    for name, expected in report.get('source_sha256', {}).items():
        bind(root / name, root, refs, expected)
    for name, expected in report['output_sha256'].items():
        bind(root / name, root, refs, expected)
    event_name = report['events_path']
    require(event_name in report['output_sha256'], 'Corporate events are not bound by the report')
    return pd.read_parquet(root / event_name)


def prepare(output=OUTPUT, *, base=BASE, extension=EXTENSION, event_extension=None, root=ROOT):
    root, base, extension, output = (Path(p).resolve() for p in (root, base, extension, output))
    require(output.is_relative_to(root) and not output.exists(), 'Choose a new repository output directory')
    refs = {}
    old = verify_bundle(base, root, refs)
    new = verify_bundle(extension, root, refs)
    require(old['schema'] == 'repaired_market_input_bundle_v1'
            and new['schema'] == 'all_signals_extension_inputs_v1', 'Unexpected sealed input schemas')
    require(new.get('historical_inputs_unchanged') is True
            and new.get('old_median50m_prefix_exact') is True
            and new.get('officially_crosschecked_extension') is False,
            'Extension provenance flags changed')
    calendar = None
    for name in MATRICES:
        dates = matrix_prefix(base, extension, name)
        require(calendar is None or dates.equals(calendar), 'Matrices use different calendars')
        calendar = dates
    before, after = [pd.read_parquet(p / 'quotes-unmasked.parquet') for p in (base, extension)]
    for frame in (before, after):
        frame['date'] = pd.to_datetime(frame.date)
        require(not frame.duplicated(['date', 'stock_id']).any(), 'Duplicate daily quote identity')
    require(after.loc[after.date.le(ANCHOR)].reset_index(drop=True).equals(before.reset_index(drop=True)),
            'Historical quote prefix changed')
    require(digest(base / 'events.parquet') == digest(extension / 'events.parquet'),
            'Corporate events unexpectedly changed; use a separately evidenced supplement')
    old_signals = read(base / 'signals.json')['entries']['median50m']
    executable, pending = partition_signals(old_signals, read(extension / 'signals.json'), calendar)
    routes = routing_audit(read(extension / 'identity.json'), executable, pending, calendar)
    merged_events = None
    if event_extension is not None:
        merged_events = merge_events(pd.read_parquet(base / 'events.parquet'),
                                     load_event_extension(event_extension, root, refs))
    bind(Path(__file__), root, refs)
    test_path = root / 'tests/test_prepare_poc_latest_inputs.py'
    if test_path.exists():
        bind(test_path, root, refs)
    output.mkdir(parents=True)
    for name in COPY_FILES:
        require(name in new['files_sha256'], 'Copied input was not sealed: ' + name)
        shutil.copyfile(extension / name, output / name)
        require(digest(output / name) == new['files_sha256'][name], 'Copy differs from source')
    if merged_events is not None:
        merged_events.to_parquet(output / 'events.parquet', index=False)
    write(output / 'signals.json', dict(entries={'median50m': executable}, source_sha256=refs,
          candidate_parameters_changed=False, account_policy='unchanged_median50m_T_plus_1',
          pending_signal_file='pending-signals.json', live_qualified=False, unseen_validation=False))
    write(output / 'pending-signals.json', dict(schema='pending_last_close_signals_v1',
          signal_date=END, entries=pending, next_session_observed=False,
          execution_inferred=False, live_qualified=False))
    write(output / 'identity-route-audit.json', dict(rows=routes, route_available=True,
          official_daily_identity_certified=False, historical_identity_upgraded=False))
    gaps = dict(schema='poc_latest_preparation_gaps_v1', start='2026-09-10', end=END,
        corporate_events_extended=merged_events is not None,
        events_extension_complete=merged_events is not None,
        corporate_events_verified_through=END if merged_events is not None else ANCHOR,
        events_source='verified_extension_report' if merged_events is not None else 'unchanged_parent_bundle',
        required_execution_preparation=['official_corporate_events_and_payment_terms',
            'dated_legal_price_limits_for_traded_stocks', 'ordinary_session_volume_for_order_capacity',
            'official_odd_lot_days_for_actual_odd_share_orders',
            'official_daily_tape_checks_and_causal_POC_ticks_for_queried_candidates'],
        prepared_execution_certified=False, all_data_complete=False,
        market_identity_basis='provisional_extension_snapshot', live_qualified=False)
    write(output / 'preparation-gaps.json', gaps)
    manifest = dict(schema='poc_latest_input_bundle_v1', created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=refs, parent_bundle=str(base.relative_to(root)),
        extension_bundle=str(extension.relative_to(root)), start=str(calendar[0].date()), end=END,
        historical_end=ANCHOR, historical_signal_end=OLD_SIGNAL_END,
        historical_matrices_and_quotes_exact=True, historical_candidate_prefix_exact=True,
        historical_candidate_count=len(old_signals), executable_candidate_count=len(executable),
        new_executable_candidates=len(executable)-len(old_signals), pending_candidate_count=len(pending),
        new_sessions=sum(str(d.date()) > ANCHOR for d in calendar),
        candidate_parameters_changed=False, corporate_events_extended=merged_events is not None,
        events_extension_complete=merged_events is not None,
        event_extension_report=str(Path(event_extension).resolve().relative_to(root)) if event_extension else None,
        officially_crosschecked_extension=False, complete_historical_universe=False,
        actual_fill_verified=False, live_qualified=False, unseen_validation=False,
        database_mutations=False, network_requests=0, finmind_requests=0,
        limitations=new['limitations'],
        files_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    write(output / 'manifest.json', manifest)
    (output / 'manifest.sha256').write_text(digest(output / 'manifest.json') + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--event-extension', type=Path,
                        help='Sealed report for complete official corporate events in the added period')
    args = parser.parse_args()
    result = prepare(args.output, event_extension=args.event_extension)
    print(json.dumps({k: result[k] for k in ('end', 'new_sessions', 'historical_candidate_count',
          'new_executable_candidates', 'pending_candidate_count', 'corporate_events_extended')}, indent=2))
