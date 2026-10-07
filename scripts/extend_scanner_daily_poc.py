#!/usr/bin/env python3
"""Continue sealed daily POC across observed dates using verified cached tapes.

Historical reports remain immutable. Missing tape coordinates are exported for
bounded acquisition through the shared FinMind client, never fetched here.
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
from scripts.prepare_scanner_daily_poc import candidate_rows, candidate_identity
from scripts.scan_market_strategies import digest, load_poc
from scripts.research_volume_profile import constant_scale
from skills.independent_three_black import path_issue
from skills.poc_daily_opportunities import DailyProfiles, INVENTORY, base_row
from skills.poc_latest_data import BUNDLE, OFFICIAL, LatestProfiles, sealed_json
from skills.strategy_scanner.data import _Inputs

CODE_SOURCES = (
    'scripts/extend_scanner_daily_poc.py', 'tests/test_extend_scanner_daily_poc.py',
    'scripts/prepare_scanner_daily_poc.py', 'skills/poc_daily_opportunities.py',
    'skills/strategy_scanner/data.py', 'scripts/scan_market_strategies.py',
)


def verify_prefix(base, extension):
    old_end, end = base.manifest['end'], extension.manifest['end']
    if not ('2026-01-01' <= old_end < end <= '2026-12-31'):
        raise ValueError('Require a strictly newer 2026 POC continuation')
    for name in ('close-official.parquet', 'close-quality.parquet', 'eligibility.parquet'):
        before = pd.read_parquet(base.verify(name))
        after = pd.read_parquet(extension.verify(name), filters=[('date', '<=', pd.Timestamp(old_end))])
        if not before.equals(after):
            raise ValueError('Historical POC matrix prefix changed: ' + name)
    cols = ['date', 'stock_id', 'open', 'high', 'low', 'close', 'volume']
    before = pd.read_parquet(base.verify('quotes-unmasked.parquet'), columns=cols)
    after = pd.read_parquet(extension.verify('quotes-unmasked.parquet'), columns=cols,
                            filters=[('date', '<=', pd.Timestamp(old_end))])
    def ordered(frame):
        return frame.sort_values(['stock_id', 'date']).reset_index(drop=True)
    if not ordered(before).equals(ordered(after)):
        raise ValueError('Historical POC quote prefix changed')
    old = {r['event_id']: candidate_identity(r) for r in candidate_rows(base)}
    new = {r['event_id']: candidate_identity(r) for r in candidate_rows(extension)
           if r['signal_date'] <= old_end}
    if old != new:
        raise ValueError('Historical original candidate prefix changed')
    before = pd.read_parquet(base.verify('events.parquet'))
    after = pd.read_parquet(extension.verify('events.parquet'))
    after = after.loc[pd.to_datetime(after.event_date).le(old_end), before.columns]
    order = list(before.columns)
    if not before.sort_values(order).reset_index(drop=True).equals(after.sort_values(order).reset_index(drop=True)):
        raise ValueError('Historical corporate-event prefix changed')
    base.finish(); extension.finish()


def action_coverage_reason(manifest, signal_date, reports):
    if manifest.get('events_extension_complete') is not True:
        return 'corporate_event_coverage_incomplete'
    bound = manifest.get('event_extension_report')
    bound = bound.get('path') if isinstance(bound, dict) else bound
    if not bound or bound not in manifest.get('source_sha256', {}):
        return 'corporate_event_evidence_unbound'
    required = {m + '_' + k for m in ('twse', 'tpex')
                for k in ('ex_rights', 'capital_reduction', 'par_value_change')}
    for report in reports:
        if not (report.get('corporate_events_extension_complete') is True
                and report.get('start', '9999') <= signal_date <= report.get('end', '')):
            continue
        covered = {r.get('kind') for r in report.get('corporate_action_coverage', [])
                   if r.get('complete') is True and r.get('start', '9999') <= signal_date <= r.get('end', '')}
        if required <= covered:
            return None
    return 'corporate_event_interval_incomplete'


class Continuation(DailyProfiles):
    def __init__(self, bundle, old_end, *, official_reports, tick_directories=()):
        LatestProfiles.__init__(self, BUNDLE, OFFICIAL, online=False)
        self.bundle = local(bundle)
        self.old_end = old_end
        self.tick_directories = [local(p) for p in tick_directories]
        reader = _Inputs(self.bundle)
        self.manifest = reader.manifest
        for name in ('close-official.parquet', 'close-quality.parquet', 'eligibility.parquet',
                     'quotes-unmasked.parquet', 'events.parquet', 'signals.json', 'pending-signals.json'):
            self._mark(reader.verify(name), reader.hashes[name])
        self._mark(self.bundle / 'manifest.json', reader.hashes['manifest.json'])
        # Direct frozen data are validated, ancestors retain their historical
        # source hashes. Only the bound action report is interpreted at runtime.
        event_report = self.manifest['event_extension_report']
        event_report = event_report['path'] if isinstance(event_report, dict) else event_report
        self._mark(ROOT / event_report, self.manifest['source_sha256'][event_report])
        self.days = pd.DatetimeIndex(pd.read_parquet(self.bundle / 'close-official.parquet', columns=['date']).date)
        self.calendar = self.days.strftime('%Y-%m-%d').tolist()
        if old_end not in self.calendar or self.calendar[-1] != self.manifest['end']:
            raise ValueError('Continuation calendar does not span both endpoints')
        self.events = pd.read_parquet(self.bundle / 'events.parquet')
        self.events['event_date'] = pd.to_datetime(self.events.event_date)
        self.action_reports = []
        for path in official_reports:
            self._extend_official(local(path))
        self.signals = [dict(signal_id=r['event_id'], signal_date=r['signal_date'], stock_id=r['members'][0])
                        for r in candidate_rows(reader) if old_end < r['signal_date'] <= self.manifest['end']]
        inventory = sealed_json(INVENTORY / 'manifest.json', self.refs)
        self._mark(INVENTORY / 'tick-reuse.json', inventory['files_sha256']['tick-reuse.json'])
        self.raw_index = json.loads((INVENTORY / 'tick-reuse.json').read_text())
        self.directory = ROOT / '.cache/poc-daily-opportunities-20261004/data-v1'
        self.attempt_paths = None
        self.days_cache = {}; self.rows = {}; self.structural = {}; self.unavailable = {}; self.allowed = set()
        self.calls_this_run = 0; self.quota_delay = 0.
        self._seed()
        reader.finish()

    def _extend_official(self, report_path):
        report = sealed_json(report_path, self.refs)
        if report.get('daily_tables_extension_complete') is not True:
            raise ValueError('Official extension is incomplete')
        normalized, sources = report.get('normalized_path'), report.get('sources_path')
        outputs = report.get('output_sha256', {})
        if not normalized or not sources or normalized not in outputs or sources not in outputs:
            raise ValueError('Official extension normalized sources are not hash-bound')
        self._mark(ROOT / normalized, outputs[normalized]); self._mark(ROOT / sources, outputs[sources])
        metadata = json.loads((ROOT / sources).read_text())
        metadata = metadata.get('sources', metadata)
        extension = pd.read_parquet(ROOT / normalized)
        extension['date'] = pd.to_datetime(extension.date).dt.strftime('%Y-%m-%d')
        extension['market'] = extension.market.str.upper()
        expected = {(m, d) for m in ('TWSE', 'TPEX') for d in self.calendar
                    if report.get('start', '9999') <= d <= report.get('end', '')}
        actual = set(zip(extension.market, extension.date))
        if (not expected or actual != expected or report.get('accepted_market_days') != len(expected)
                or report.get('required_market_days') != len(expected) or report.get('missing_market_days') != []):
            raise ValueError('Missing official market-day coverage')
        if not set(extension.source_id) <= metadata.keys():
            raise ValueError('Official extension has an unbound source id')
        combined = pd.concat([self.official.reset_index(), extension], ignore_index=True)
        if combined.duplicated(['stock_id', 'date']).any():
            raise ValueError('Official profile identity duplicate')
        if set(metadata) & self.official_meta['sources'].keys():
            raise ValueError('Official source-id collision')
        self.official = combined.set_index(['stock_id', 'date']).sort_index()
        self.official_meta['sources'].update(metadata)
        self.action_reports.append(report)

    def _seed(self):
        identities = Counter(self.official.index)
        for event in self.signals:
            row = base_row(event, self.calendar); sid = row['stock_id']; i = self.calendar.index(row['signal_date'])
            path = self._path(sid)
            actions = self.events.loc[self.events.stock_id.eq(sid), 'event_date']
            reason = action_coverage_reason(self.manifest, row['signal_date'], self.action_reports)
            if reason is None and (actions.between(self.days[i-20], self.days[i]).any()
                    or not constant_scale(path.close[i-20:i+1], path.raw_close[i-20:i+1])):
                reason = 'corporate_action_or_nonconstant_price_scale'
            elif reason is None and path_issue(path, i-20, i):
                reason = 'pre_signal_daily_path_invalid'
            elif reason is None and any(identities[(sid, d)] != 1 for d in row['prior_dates']):
                reason = 'official_identity_missing_or_ambiguous'
            if reason:
                row.update(status='unknown', available=False, reason=reason)
                self.structural[row['signal_id']] = row
            else:
                self.allowed.update((sid, d) for d in row['prior_dates'])
            self.rows[row['signal_id']] = row

    def _stored(self, sid, day):
        query = dict(dataset='TaiwanStockPriceTick', data_id=sid, start_date=day)
        found = []
        for directory in self.tick_directories:
            path = directory / 'receipts' / f'{sid}-{day}.json'
            if path.exists():
                self._mark(path)
                item = verify_saved(json.loads(path.read_text()), query)
                if item.get('raw_path'):
                    self._mark(ROOT / item['raw_path'], item['raw_sha256'])
                found.append(item)
        # New directories must not mask an earlier sealed receipt for the same
        # coordinate. Compare normalized observations, since parquet encodings
        # can differ while the underlying tape is identical.
        historical = super()._stored(sid, day)
        if historical is not None:
            found.append(historical)
        received = [r for r in found if r['status'] in ('received', 'cached')]
        if received:
            if len({r['raw_sha256'] for r in received}) > 1:
                from skills.intraday_limit_replay import normalize_ticks
                official, _ = self._source(sid, day)
                if official is None:
                    raise ValueError('Cached tape comparison lacks dated official identity')
                canonical = None
                for item in received:
                    normalized = normalize_ticks(pd.read_parquet(ROOT / item['raw_path']), sid, day, official['market'])
                    if canonical is not None and not canonical.equals(normalized):
                        raise ValueError('Conflicting normalized cached tapes across sources')
                    canonical = normalized
            return received[0]
        return historical or (found[0] if found else None)

    def offline(self):
        for sid, day in sorted(self.allowed):
            self.day(sid, day, fetch=False)
        self.refresh()

    def _fetch(self, *args, **kwargs):
        raise RuntimeError('POC continuation is cache-only; acquire exported coordinates separately')


def run(args):
    output = local(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty immutable output directory')
    code_refs = {name: digest(ROOT / name) for name in CODE_SOURCES}
    base, extension = _Inputs(args.base_bundle), _Inputs(args.bundle)
    verify_prefix(base, extension)
    old_rows, old_info = load_poc(args.base_report, bundle=args.base_bundle,
                                 manifest_hash=base.hashes['manifest.json'])
    if old_info['end'] != base.manifest['end']:
        raise ValueError('Historical POC endpoint differs from the base bundle')
    expected = {(r['event_id'], r['members'][0], r['signal_date']) for r in candidate_rows(base)
                if r['signal_date'].startswith('2026-')}
    actual = {(r['signal_id'], r['stock_id'], r['signal_date']) for r in old_rows}
    if actual != expected or len(old_rows) != len(expected):
        raise ValueError('Historical POC rows do not cover the complete original 2026 cohort')
    provider = Continuation(args.bundle, base.manifest['end'], official_reports=args.official_report,
                            tick_directories=args.tick_directory)
    provider.offline()
    rows = old_rows + list(provider.rows.values())
    coordinates = sorted({(r['stock_id'], d) for r in provider.rows.values()
                          if r['status'] == 'pending_data' for d in r['missing_dates']})
    missing = [dict(stock_id=sid, date=day, dataset='TaiwanStockPriceTick',
                    previous_status=provider.unavailable.get(sid+'-'+day, 'not_requested')) for sid, day in coordinates]
    write(output / 'profiles.json', rows); write(output / 'missing-ticks.json', missing)
    write(output / 'day-audits.json', {k: {n: v for n, v in r.items() if n != 'prices'} for k, r in provider.days_cache.items()})
    refs = dict(provider.refs)
    for name, expected_hash in code_refs.items():
        if digest(ROOT / name) != expected_hash:
            raise ValueError('POC continuation code changed during execution')
        if name in refs and refs[name] != expected_hash:
            raise ValueError('Conflicting runtime source hash: ' + name)
        refs[name] = expected_hash
    refs[str(local(args.base_report).relative_to(ROOT))] = digest(args.base_report)
    old_profile = json.loads(Path(args.base_report).read_text())['profiles']['path']
    refs[old_profile] = old_info['profiles_sha256']
    for name in ('day-audits.json', 'missing-ticks.json'):
        refs[str((output / name).relative_to(ROOT))] = digest(output / name)
    report = dict(schema='poc_daily_profiles_v1', year=2026, end=extension.manifest['end'],
                  created_at=datetime.now(timezone.utc).isoformat(),
                  profiles=dict(path=str((output/'profiles.json').relative_to(ROOT)), sha256=digest(output/'profiles.json')),
                  source_sha256=refs, all_signals_materialized=True,
                  all_profiles_available=all(r['available'] for r in rows),
                  status_counts=dict(Counter(r['status'] for r in rows)),
                  continuation_status_counts=dict(Counter(r['status'] for r in provider.rows.values())),
                  continuation_signal_count=len(provider.rows), prior_profile_count=len(old_rows),
                  missing_tick_coordinates=len(missing), external_data_requests=0,
                  historical_prefix_verified=True, historical_profile_rows_unchanged=True,
                  window='20 observed market sessions strictly before each signal date',
                  known_at_signal_realtime_receipts_verified=False,
                  account_independent=True, live_qualified=False)
    write(output/'report.json', report)
    (output/'report.sha256').write_text(digest(output/'report.json')+'\n')
    print(json.dumps({key: report[key] for key in ('end', 'continuation_signal_count',
        'continuation_status_counts', 'missing_tick_coordinates')}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-bundle', type=Path, required=True)
    parser.add_argument('--base-report', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--official-report', type=Path, action='append', required=True)
    parser.add_argument('--tick-directory', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
