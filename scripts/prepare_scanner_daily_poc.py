#!/usr/bin/env python3
"""Extend sealed scanner POC evidence using cached authentic tapes only.

This October 5 continuation preserves the historical candidate/profile cohort.
No network, portfolio allocation, execution price estimate, or DB write occurs.
Missing or conflicting evidence stays unknown; pending coordinates are exported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.prepare_volume_profile import local, verify_saved, write
from scripts.scan_market_strategies import DEFAULT_POC, digest, load_poc
from scripts.research_volume_profile import constant_scale
from skills.independent_three_black import path_issue
from skills.poc_daily_opportunities import DailyProfiles, INVENTORY, assess, base_row
from skills.poc_latest_data import BUNDLE, OFFICIAL, LatestProfiles, sealed_json
from skills.strategy_scanner.data import _Inputs

END = '2026-10-05'
OLD_END = '2026-10-02'
OLD_DATA = ROOT / '.cache/poc-daily-opportunities-20261004/data-v1'

RUN_CODE_SOURCES = (
    'scripts/prepare_scanner_daily_poc.py',
    'tests/test_prepare_scanner_daily_poc.py',
    # The continuation deliberately bypasses DailyProfiles.__init__, so its
    # inherited runtime algorithms need an explicit source binding here.
    'skills/poc_daily_opportunities.py',
    'skills/strategy_scanner/data.py',
    'scripts/scan_market_strategies.py',
)


def run_code_refs():
    return {name: digest(ROOT / name) for name in RUN_CODE_SOURCES}



def candidate_rows(reader):
    source = reader.read_json('signals.json')
    pending = reader.read_json(source.get('pending_signal_file', 'pending-signals.json'))
    rows = source['entries']['median50m'] + pending['entries']
    keys = [row['event_id'] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate original candidate identities')
    return rows


def candidate_identity(row):
    # A terminal signal may acquire an observed T+1 without changing the signal.
    return {key: value for key, value in row.items() if key != 'entry_date'}


def verify_prefix(base, extension):
    """Do not rebind old POC hashes to changed history or changed candidates."""
    if base.manifest.get('end') != OLD_END or extension.manifest.get('end') != END:
        raise ValueError('This adapter requires the sealed October 2 to October 5 continuation')
    for name in ('close-official.parquet', 'close-quality.parquet', 'eligibility.parquet'):
        before = pd.read_parquet(base.verify(name))
        after = pd.read_parquet(extension.verify(name), filters=[('date', '<=', pd.Timestamp(OLD_END))])
        if not before.equals(after):
            raise ValueError('Historical POC matrix prefix changed: ' + name)
    columns = ['date', 'stock_id', 'open', 'high', 'low', 'close', 'volume']
    before = pd.read_parquet(base.verify('quotes-unmasked.parquet'), columns=columns)
    after = pd.read_parquet(extension.verify('quotes-unmasked.parquet'), columns=columns,
                            filters=[('date', '<=', pd.Timestamp(OLD_END))])
    for frame in (before, after):
        frame.sort_values(['stock_id', 'date'], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if not before.equals(after):
        raise ValueError('Historical POC quote prefix changed')
    old = {row['event_id']: candidate_identity(row) for row in candidate_rows(base)}
    new = {row['event_id']: candidate_identity(row) for row in candidate_rows(extension)
           if row['signal_date'] <= OLD_END}
    if old != new:
        raise ValueError('Historical original candidate prefix changed')
    # Events can have source columns; all old records must be preserved exactly.
    before = pd.read_parquet(base.verify('events.parquet'))
    after = pd.read_parquet(extension.verify('events.parquet'))
    after = after.loc[pd.to_datetime(after.event_date).le(OLD_END), before.columns]
    order = list(before.columns)
    if not before.sort_values(order).reset_index(drop=True).equals(after.sort_values(order).reset_index(drop=True)):
        raise ValueError('Historical corporate-event prefix changed')
    base.finish()
    extension.finish()


def corporate_coverage_reason(manifest, report_reader=None):
    """A unchanged price scale is not proof of a complete action calendar."""
    if manifest.get('events_extension_complete') is not True:
        return 'corporate_event_coverage_incomplete'
    report = manifest.get('event_extension_report')
    if not report:
        return 'corporate_event_evidence_unbound'
    if isinstance(report, dict):
        report = report.get('path')
    if not report or report not in manifest.get('source_sha256', {}):
        return 'corporate_event_evidence_unbound'
    if report_reader is None:
        report_reader = lambda path: json.loads((ROOT / path).read_text())
    evidence = report_reader(report)
    required = {market + '_' + kind for market in ('twse', 'tpex')
                for kind in ('ex_rights', 'capital_reduction', 'par_value_change')}
    covered = {row.get('kind') for row in evidence.get('corporate_action_coverage', [])
               if row.get('complete') is True and row.get('start', '9999') <= END <= row.get('end', '')}
    if (evidence.get('corporate_events_extension_complete') is not True
            or not evidence.get('start', '9999') <= END <= evidence.get('end', '')
            or not required.issubset(covered)):
        return 'corporate_event_interval_incomplete'
    return None


class CachedContinuation(DailyProfiles):
    """Reuse the verified old official sources, without invoking old acquisition."""

    def __init__(self, bundle, *, extra_ticks=None):
        LatestProfiles.__init__(self, BUNDLE, OFFICIAL, online=False)
        self.bundle = local(bundle)
        self.extra_ticks = local(extra_ticks) if extra_ticks else None
        reader = _Inputs(self.bundle)
        self.manifest = reader.manifest
        for name in ('close-official.parquet', 'close-quality.parquet', 'eligibility.parquet',
                     'quotes-unmasked.parquet', 'events.parquet', 'signals.json', 'pending-signals.json'):
            self._mark(reader.verify(name), reader.hashes[name])
        self._mark(self.bundle / 'manifest.json', reader.hashes['manifest.json'])
        for name, expected in self.manifest.get('source_sha256', {}).items():
            self._mark(ROOT / name, expected)
        self.days = pd.DatetimeIndex(pd.read_parquet(self.bundle / 'close-official.parquet', columns=['date']).date)
        self.calendar = self.days.strftime('%Y-%m-%d').tolist()
        if self.calendar[-2:] != [OLD_END, END]:
            raise ValueError('Unexpected continuation calendar')
        self.events = pd.read_parquet(self.bundle / 'events.parquet')
        self.events['event_date'] = pd.to_datetime(self.events.event_date)
        rows = candidate_rows(reader)
        self.signals = [dict(signal_id=e['event_id'], signal_date=e['signal_date'], stock_id=e['members'][0])
                        for e in rows if e['signal_date'] == END]
        inventory = sealed_json(INVENTORY / 'manifest.json', self.refs)
        expected = inventory['files_sha256']['tick-reuse.json']
        self._mark(INVENTORY / 'tick-reuse.json', expected)
        self.raw_index = json.loads((INVENTORY / 'tick-reuse.json').read_text())
        self.directory = OLD_DATA  # Read-only receipts; never call fill or _fetch.
        self.attempt_paths = None
        self.days_cache = {}
        self.rows = {}
        self.structural = {}
        self.unavailable = {}
        self.allowed = set()
        self.calls_this_run = 0
        self.quota_delay = 0.
        self._seed_continuation()
        reader.finish()

    def _seed_continuation(self):
        coverage = corporate_coverage_reason(self.manifest)
        identities = Counter(self.official.index)
        for event in self.signals:
            row = base_row(event, self.calendar)
            sid = row['stock_id']
            i = self.calendar.index(row['signal_date'])
            path = self._path(sid)
            actions = self.events.loc[self.events.stock_id.eq(sid), 'event_date']
            reason = coverage
            if reason is None and (actions.between(self.days[i - 20], self.days[i]).any()
                    or not constant_scale(path.close[i - 20:i + 1], path.raw_close[i - 20:i + 1])):
                reason = 'corporate_action_or_nonconstant_price_scale'
            elif reason is None and path_issue(path, i - 20, i):
                reason = 'pre_signal_daily_path_invalid'
            elif reason is None and any(identities[(sid, day)] != 1 for day in row['prior_dates']):
                reason = 'official_identity_missing_or_ambiguous'
            if reason:
                row.update(status='unknown', available=False, reason=reason)
                self.structural[row['signal_id']] = row
            else:
                self.allowed.update((sid, day) for day in row['prior_dates'])
            self.rows[row['signal_id']] = row

    def _stored(self, sid, day):
        if self.extra_ticks is not None:
            path = self.extra_ticks / 'receipts' / f'{sid}-{day}.json'
            if path.exists():
                self._mark(path)
                query = dict(dataset='TaiwanStockPriceTick', data_id=sid, start_date=day)
                item = verify_saved(json.loads(path.read_text()), query)
                if item.get('raw_path'):
                    self._mark(ROOT / item['raw_path'], item['raw_sha256'])
                return item
        return super()._stored(sid, day)

    def offline(self):
        for sid, day in sorted(self.allowed):
            self.day(sid, day, fetch=False)
        self.refresh()

    def _fetch(self, *args, **kwargs):
        raise RuntimeError('This continuation is cache-only; acquire the exported missing coordinates separately')


def run(args):
    output = local(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty immutable output directory')
    code_refs = run_code_refs()
    base, extension = _Inputs(BUNDLE), _Inputs(args.bundle)
    verify_prefix(base, extension)
    old_rows, old_info = load_poc(args.base_report, bundle=BUNDLE,
                                 manifest_hash=base.hashes['manifest.json'])
    expected_old = {(e['event_id'], e['members'][0], e['signal_date'])
                    for e in candidate_rows(base) if e['signal_date'].startswith('2026-')}
    actual_old = {(e['signal_id'], e['stock_id'], e['signal_date']) for e in old_rows}
    if actual_old != expected_old or len(old_rows) != len(expected_old):
        raise ValueError('Historical POC rows do not match the complete original 2026 cohort')
    provider = CachedContinuation(args.bundle, extra_ticks=args.tick_directory)
    provider.offline()
    rows = old_rows + list(provider.rows.values())
    coordinates = [(r['stock_id'], d) for r in provider.rows.values()
                   if r['status'] == 'pending_data' for d in r['missing_dates']]
    missing = [dict(stock_id=sid, date=day, dataset='TaiwanStockPriceTick',
                    previous_status=provider.unavailable.get(sid + '-' + day, 'not_requested'))
               for sid, day in sorted(set(coordinates))]
    write(output / 'profiles.json', rows)
    write(output / 'missing-ticks.json', missing)
    write(output / 'day-audits.json', {k: {n: v for n, v in r.items() if n != 'prices'}
                                     for k, r in provider.days_cache.items()})
    refs = dict(provider.refs)
    refs[str(local(args.base_report).relative_to(ROOT))] = digest(args.base_report)
    refs[str((ROOT / json.loads(Path(args.base_report).read_text())['profiles']['path']).relative_to(ROOT))] = old_info['profiles_sha256']
    if run_code_refs() != code_refs:
        raise ValueError('POC continuation code changed during execution')
    for name, expected in code_refs.items():
        if name in refs and refs[name] != expected:
            raise ValueError('Conflicting POC runtime source hash: ' + name)
        refs[name] = expected
    for name in ('day-audits.json', 'missing-ticks.json'):
        refs[str((output / name).relative_to(ROOT))] = digest(output / name)
    counts = dict(Counter(r['status'] for r in rows))
    report = dict(schema='poc_daily_profiles_v1', year=2026, end=END,
                  created_at=datetime.now(timezone.utc).isoformat(),
                  profiles=dict(path=str((output / 'profiles.json').relative_to(ROOT)), sha256=digest(output / 'profiles.json')),
                  source_sha256=refs, all_signals_materialized=True,
                  all_profiles_available=all(r['available'] for r in rows), status_counts=counts,
                  continuation_status_counts=dict(Counter(r['status'] for r in provider.rows.values())),
                  continuation_signal_count=len(provider.rows), prior_profile_count=len(old_rows),
                  missing_tick_coordinates=len(missing), external_data_requests=0,
                  historical_prefix_verified=True, historical_profile_rows_unchanged=True,
                  window='20 observed market sessions strictly before each signal date',
                  known_at_signal_realtime_receipts_verified=False,
                  corporate_event_coverage_reason=corporate_coverage_reason(provider.manifest),
                  account_independent=True, live_qualified=False)
    write(output / 'report.json', report)
    (output / 'report.sha256').write_text(digest(output / 'report.json') + '\n')
    print(json.dumps({key: report[key] for key in ('end', 'continuation_signal_count',
        'continuation_status_counts', 'missing_tick_coordinates', 'corporate_event_coverage_reason')}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / '.cache/scanner-20261006/inputs-v1')
    parser.add_argument('--base-report', type=Path, default=DEFAULT_POC)
    parser.add_argument('--tick-directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
