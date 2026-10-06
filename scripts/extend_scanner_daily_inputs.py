#!/usr/bin/env python3
"""Append one observed market session to a sealed scanner input bundle, offline.

Raw quotes and provider adjustments are caller-supplied local snapshots. The old
price history is never rebased. A current security master only provides a
provisional membership check, not historical market certification.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from skills.candidate_quality import candidate_features
from skills.historical_identity_repair import account_entry_decision
from skills.historical_selector_replay import eligibility_matrix
from skills.liquidity_diagnostics import liquidity_features

MATRICES = ('raw-close', 'raw-volume', 'close-official', 'close-quality', 'eligibility')
CODE_SOURCES = ('scripts/extend_scanner_daily_inputs.py', 'skills/candidate_quality.py',
                'skills/liquidity_diagnostics.py', 'skills/historical_identity_repair.py',
                'skills/historical_selector_replay.py', 'skills/regime_state.py',
                'skills/diffusion_signals.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def bind(path, root, refs, expected=None):
    path = Path(path).resolve()
    require(path.is_relative_to(root), 'Input path escapes repository root')
    value = digest(path)
    require(expected is None or value == expected, 'Sealed source hash mismatch: ' + str(path))
    key = str(path.relative_to(root))
    require(key not in refs or refs[key] == value, 'Source changed while preparing: ' + key)
    refs[key] = value
    return path


def day(value):
    require(isinstance(value, str) and date.fromisoformat(value).isoformat() == value,
            'Dates must be ISO YYYY-MM-DD')
    return pd.Timestamp(value)


def load_day(path, expected, root, refs, *, quotes=False):
    value = pd.read_parquet(bind(path, root, refs))
    mapping = {'max': 'high', 'min': 'low', 'Trading_Volume': 'volume'}
    for source, target in mapping.items():
        if quotes and source in value and target in value:
            require(value[source].equals(value[target]), 'Conflicting quote field mapping: ' + target)
    if quotes:
        value = value.rename(columns={k: v for k, v in mapping.items() if v not in value})
    fields = ['date', 'stock_id'] + (['open', 'high', 'low', 'close', 'volume'] if quotes else ['close'])
    require(set(fields).issubset(value.columns), 'Missing daily source columns')
    value = value[fields].copy()
    value['date'] = pd.to_datetime(value.date, errors='raise')
    require(len(value) > 0 and value.date.eq(expected).all(), 'Daily source has wrong date or is empty')
    require(value.stock_id.map(lambda v: isinstance(v, str) and bool(v)).all(),
            'Source stock_id must be a nonempty string; numeric coercion loses leading zeros')
    require(not value.duplicated(['date', 'stock_id']).any(), 'Duplicate daily source stock/date')
    for key in fields[2:]:
        require(pd.api.types.is_numeric_dtype(value[key]) and not pd.api.types.is_bool_dtype(value[key]),
                'Daily source needs numeric, nonboolean prices/volume')
    return value


def extend_identity(identity, companies, ids, snapshot, *, anchor, end):
    require({'stock_id', 'type', 'stock_name', 'industry_category', 'date'}.issubset(snapshot.columns),
            'Missing market snapshot columns')
    require(snapshot.stock_id.map(lambda v: isinstance(v, str) and bool(v)).all(),
            'Snapshot stock_id must be a nonempty string')
    require(len(snapshot) > 0, 'Market snapshot is empty')
    duplicates = int(snapshot.stock_id.duplicated().sum())
    conflicts = snapshot.groupby('stock_id')[['type', 'stock_name', 'date']].nunique(dropna=False)
    require(not conflicts.gt(1).any().any(), 'Conflicting snapshot stock_id market/name/date')
    # FinMind repeats broad/narrow industry labels. Only identical identity fields
    # may be collapsed; industry is display metadata and never selects this pool.
    snapshot = snapshot.groupby(['stock_id', 'type', 'stock_name', 'date'], dropna=False, as_index=False).agg(
        industry_category=('industry_category', lambda values: '|'.join(sorted({str(v) for v in values}))))
    dates = pd.to_datetime(snapshot.date, errors='coerce')
    snapshot = snapshot.assign(record_date_known=dates.notna())
    latest = snapshot.set_index('stock_id')
    result = deepcopy(identity)
    result['coverage_end'] = end
    checks, allowed = [], set()
    for sid in ids:
        episodes = [e for e in result['episodes'] if e['stock_id'] == sid
                    and e.get('start') is not None and e['start'] <= end
                    and (e['end'] is None or end < e['end'])]
        require(len(episodes) <= 1, 'Overlapping identity episodes: ' + sid)
        reason = 'identity_unavailable'
        if len(episodes) == 1:
            ep = episodes[0]
            if sid not in latest.index:
                reason = 'snapshot_stock_missing'
            elif not bool(latest.at[sid, 'record_date_known']):
                reason = 'snapshot_record_date_unknown'
            elif not isinstance(latest.at[sid, 'type'], str):
                reason = 'snapshot_market_unknown'
            elif latest.at[sid, 'type'].upper() not in ('TWSE', 'TPEX'):
                reason = 'snapshot_market_outside_listed_otc'
            elif latest.at[sid, 'type'].upper() != ep['market'].upper():
                reason = 'snapshot_market_changed'
            elif ep['category'] != ('ETF' if sid == '0050' else '股票'):
                reason = 'outside_ordinary_share_cohort'
            else:
                ep['snapshot_date'] = end
                allowed.add(sid)
                reason = 'provisional_snapshot_market_match'
        checks.append(dict(stock_id=sid, reason=reason, snapshot_match=sid in allowed))
    observation = identity.get('extension_observation', {})
    result['extension_observation'] = dict(through=end,
        source='caller_supplied_current_security_snapshot_market_match',
        official_identity_verified_through=observation.get('official_identity_verified_through',
                                                          identity['coverage_end']),
        provisional=True, snapshot_is_point_in_time_certification=False,
        snapshot_record_date_min=str(dates.min().date()) if dates.notna().any() else None,
        snapshot_record_date_max=str(dates.max().date()) if dates.notna().any() else None,
        duplicate_industry_rows_collapsed=duplicates,
        unknown=[r for r in checks if not r['snapshot_match']])
    result.update(complete_historical_universe=False, continuous_eligibility_proven=False,
                  publication_time_archive_complete=False, live_qualified=False)
    mask = eligibility_matrix(result, companies, pd.DatetimeIndex([end])).reindex(columns=ids)
    for sid in ids:
        if sid not in allowed:
            mask.loc[:, sid] = False
        elif bool(mask.iloc[0][sid]):
            decision = account_entry_decision(result, sid, end, channel='regular',
                                               research_risk_notice_assumed=False)
            if not decision['allowed']:
                mask.loc[:, sid] = False
                checks[ids.index(sid)]['account_exclusion'] = decision['reason']
    return result, mask, checks


def original_candidates(frames, companies, identity, signal_day, *, features=None):
    """Exact price/volume stage of the sealed median50m liquid-universe rules."""
    f = features if features is not None else candidate_features(frames, companies)
    close = f['close']; stamp = day(signal_day); days = close.index
    volume = frames['raw-volume'].where(close.notna() & frames['raw-volume'].gt(0))
    high = close.shift(1).rolling(60, min_periods=60).max()
    mean = volume.shift(1).rolling(20, min_periods=20).mean()
    own = close / close.shift(20) - 1
    median = liquidity_features(frames['raw-close'], frames['raw-volume'])['median20']
    technical = (f['quality'].loc[stamp] & close.loc[stamp].gt(high.loc[stamp])
                 & own.loc[stamp].gt(0) & f['relative20'].loc[stamp].gt(0)
                 & volume.loc[stamp].ge(mean.loc[stamp] * 1.5) & median.loc[stamp].ge(50_000_000))
    technical['0050'] = False
    month = str(stamp.to_period('M'))
    prior_month = days[days < pd.Timestamp(month + '-01')]
    require(len(prior_month) > 0, 'Candidate warmup requires a prior-month market session')
    cutoff = str(prior_month[-1].date())
    candidates, rejected = [], []
    if f['trend'].at[stamp] != 'ON':
        return candidates, rejected
    for sid in technical.index[technical]:
        event = dict(event_id=f'liquid_universe-{signal_day}-{sid}', signal_date=signal_day,
            entry_date=None, members=[sid], priority=float(f['relative20'].at[stamp, sid]),
            group_id='liquid_universe-' + month, group_cutoff_date=cutoff, group_members=[sid],
            selection_reason='causal breakout and volume expansion; liquid_universe',
            leader_evidence=dict(leader_return20=float(own.at[stamp, sid]),
                benchmark_return20=float(own.at[stamp, '0050']),
                leader_volume_ratio=float(volume.at[stamp, sid] / mean.at[stamp, sid])))
        decision = account_entry_decision(identity, sid, signal_day, channel='regular',
                                           research_risk_notice_assumed=False)
        if decision['allowed']:
            candidates.append(event)
        else:
            rejected.append(dict(event=event, decision=decision))
    return sorted(candidates, key=lambda e: (-e['priority'], e['event_id'])), rejected


def load_event_extension(path, *, anchor, end, base, manifest, root, refs):
    require(manifest.get('events_extension_complete') is True,
            'Parent corporate events must already be complete through its end')
    path = Path(path).resolve()
    sidecar = bind(path.with_suffix('.sha256'), root, refs)
    report = read(bind(path, root, refs, sidecar.read_text().strip()))
    require(report.get('schema') == 'poc_latest_official_extension_v1'
            and report.get('start') == end and report.get('end') == end
            and report.get('corporate_events_extension_complete') is True,
            'Corporate event report does not completely cover appended session')
    rows = report.get('corporate_action_coverage', [])
    expected = {m + '_' + k for m in ('twse', 'tpex')
                for k in ('ex_rights', 'capital_reduction', 'par_value_change')}
    require(len(rows) == 6 and {r.get('kind') for r in rows} == expected
            and all(r.get('complete') is True and r.get('start') == end and r.get('end') == end for r in rows),
            'Corporate event report needs all six complete action intervals')
    for field in ('source_sha256', 'output_sha256'):
        require(isinstance(report.get(field), dict), 'Corporate event report lacks source/output hashes')
        for name, expected_hash in report[field].items():
            bind(root / name, root, refs, expected_hash)
    name = report['events_path']
    require(name in report['output_sha256'], 'Corporate events are not a bound report output')
    old = pd.read_parquet(base / 'events.parquet')
    added = pd.read_parquet(root / name)
    require(set(old.columns) == set(added.columns), 'Corporate event columns differ')
    added = added.loc[:, old.columns].copy()
    stamps = pd.to_datetime(added.event_date, errors='raise')
    require(stamps.eq(day(end)).all(), 'Corporate event lies outside appended session')
    require(added.stock_id.map(lambda s: isinstance(s, str) and bool(re.fullmatch(r'\d{4}', s))).all(),
            'Corporate events need four-digit security identifiers')
    added['event_date'] = stamps if pd.api.types.is_datetime64_any_dtype(old.event_date) else stamps.dt.date
    combined = pd.concat([old, added], ignore_index=True)
    require(combined.iloc[:len(old)].equals(old), 'Corporate event prefix changed')
    require(not combined.duplicated(['stock_id', 'event_date']).any(), 'Duplicate corporate event identity')
    require(pd.to_datetime(old.event_date).le(day(anchor)).all(), 'Parent corporate event exceeds parent end')
    return combined


def extend(*, base, quotes, adj_anchor, adj_end, market_snapshot, end, output,
           root=ROOT, calendar=None, event_extension_report=None):
    root = Path(root).resolve(); base = Path(base).resolve(); output = Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), 'Choose a new repository output directory')
    refs = {}
    sidecar = bind(base / 'manifest.sha256', root, refs)
    manifest = read(bind(base / 'manifest.json', root, refs, sidecar.read_text().strip()))
    require(manifest.get('schema') == 'poc_latest_input_bundle_v1', 'Unsupported parent bundle schema')
    required = {n + '.parquet' for n in MATRICES} | {'quotes-unmasked.parquet', 'companies.parquet',
                                                   'identity.json', 'signals.json', 'pending-signals.json'}
    require(required.issubset(manifest.get('files_sha256', {})), 'Parent manifest lacks required direct input hashes')
    for name, expected in manifest['files_sha256'].items():
        path = (base / name).resolve()
        require(path.is_relative_to(base), 'Parent manifest input escapes bundle')
        bind(path, root, refs, expected)
    anchor = manifest['end']; before = day(anchor); after = day(end)
    require(after > before, 'Extension end must follow parent end')
    if calendar is None:
        expected = pd.bdate_range(before + pd.Timedelta(days=1), after)
        require(list(expected) == [after], 'Multiple weekdays require an explicit observed market calendar')
        calendar_basis = 'one_following_weekday_confirmed_by_supplied_quotes_not_holiday_certification'
    else:
        dates = pd.read_parquet(bind(calendar, root, refs))['date']
        expected = pd.DatetimeIndex(pd.to_datetime(dates, errors='raise'))
        require(expected.is_unique and expected.is_monotonic_increasing and not expected.hasnans,
                'Invalid supplied market calendar')
        require(before in expected and list(expected[(expected > before) & (expected <= after)]) == [after],
                'Supplied calendar contains omitted market sessions')
        calendar_basis = 'caller_supplied_observed_market_calendar'
    frames = {name: pd.read_parquet(base / (name + '.parquet')).set_index('date') for name in MATRICES}
    for value in frames.values():
        value.index = pd.DatetimeIndex(value.index)
    days = frames['close-official'].index; ids = list(frames['close-official'].columns)
    require(days.is_unique and days.is_monotonic_increasing and not days.hasnans and days[-1] == before,
            'Invalid parent matrix calendar')
    require(str(days[0].date()) == manifest['start'], 'Parent calendar start differs from manifest')
    require('0050' in ids and len(ids) == len(set(ids)) and all(re.fullmatch(r'[1-9]\d{3}|0050', s) for s in ids),
            'Parent universe requires four-digit individual stocks plus 0050 only')
    require(all(f.index.equals(days) and list(f.columns) == ids for f in frames.values()),
            'Parent matrix axes differ')
    require(all(pd.api.types.is_bool_dtype(t) for t in frames['eligibility'].dtypes)
            and not frames['eligibility'].isna().any().any(), 'Original selector requires known parent eligibility')
    companies = pd.read_parquet(base / 'companies.parquet')
    require(not companies.stock_id.duplicated().any() and set(companies.stock_id) == set(ids) - {'0050'},
            'Company metadata differs from parent stock cohort')
    snapshot = pd.read_parquet(bind(market_snapshot, root, refs))
    identity, mask, identity_checks = extend_identity(read(base / 'identity.json'), companies, ids, snapshot,
                                                     anchor=anchor, end=end)
    raw = load_day(quotes, after, root, refs, quotes=True)
    aa = load_day(adj_anchor, before, root, refs).set_index('stock_id').close.reindex(ids)
    ae = load_day(adj_end, after, root, refs).set_index('stock_id').close.reindex(ids)
    newq = raw.loc[raw.stock_id.isin(ids)].copy()
    require(len(newq) > 0, 'No parent-cohort daily quotes in source')
    indexed = newq.set_index('stock_id').reindex(ids)
    require(bool(mask.iloc[0]['0050']) and np.isfinite(indexed.at['0050', 'close'])
            and indexed.at['0050', 'close'] > 0 and np.isfinite(aa['0050']) and aa['0050'] > 0
            and np.isfinite(ae['0050']) and ae['0050'] > 0,
            '0050 identity, quote and adjustment anchor are required for original-candidate completeness')
    missing = []
    for name in MATRICES:
        old = frames[name]
        if name == 'eligibility':
            extension = mask
        elif name in ('raw-close', 'raw-volume'):
            field = 'close' if name == 'raw-close' else 'volume'
            extension = pd.DataFrame([indexed[field].to_numpy()], columns=ids, index=[after]).where(mask)
        else:
            scale = old.loc[before] / aa
            valid = np.isfinite(scale) & old.loc[before].gt(0) & np.isfinite(aa) & aa.gt(0)
            values = (ae * scale).where(valid & np.isfinite(ae) & ae.gt(0))
            extension = pd.DataFrame([values.to_numpy()], columns=ids, index=[after]).where(mask)
            for sid in ids:
                if mask.at[after, sid] and pd.isna(extension.at[after, sid]):
                    missing.append(dict(field=name, stock_id=sid, reason='missing_positive_adjustment_or_anchor'))
        combined = pd.concat([old, extension])
        require(combined.iloc[:-1].equals(old), 'Historical matrix prefix changed: ' + name)
        frames[name] = combined
    oldq = pd.read_parquet(base / 'quotes-unmasked.parquet')
    require(not oldq.duplicated(['date', 'stock_id']).any()
            and pd.to_datetime(oldq.date).le(before).all(), 'Invalid parent daily quotes')
    combinedq = pd.concat([oldq, newq.loc[:, oldq.columns]], ignore_index=True)
    require(combinedq.iloc[:len(oldq)].equals(oldq), 'Historical quote prefix changed')
    require(not combinedq.duplicated(['date', 'stock_id']).any(), 'Duplicate combined quotes')
    ledger = read(base / 'signals.json'); terminal = read(base / ledger.get('pending_signal_file', 'pending-signals.json'))
    previous = ledger['entries']['median50m']; pending = terminal['entries']
    require(ledger.get('pending_signal_file', 'pending-signals.json') == 'pending-signals.json',
            'Extension requires the sealed standard pending-signal file')
    require(manifest.get('executable_candidate_count') == len(previous)
            and manifest.get('pending_candidate_count') == len(pending), 'Parent candidate counts differ')
    require(terminal.get('schema') == 'pending_last_close_signals_v1', 'Invalid parent terminal ledger')
    position = {str(d.date()): i for i, d in enumerate(days)}
    for event in previous:
        i = position.get(event['signal_date'], -1)
        require(i >= 0 and i + 1 < len(days) and event['entry_date'] == str(days[i+1].date()),
                'Parent executable candidate is not observed T+1')
    require(all(e['signal_date'] == anchor and e['entry_date'] is None for e in pending),
            'Parent pending candidate is not terminal close')
    events = previous + pending
    require(len({e['event_id'] for e in events}) == len(events), 'Duplicate parent candidate identity')
    features = candidate_features(frames, companies)
    reproduced, _ = original_candidates(frames, companies, identity, anchor, features=features)
    require(sorted(reproduced, key=lambda e: e['event_id']) == sorted(pending, key=lambda e: e['event_id']),
            'Original terminal candidates changed; do not claim unchanged candidate parameters')
    additions, rejections = original_candidates(frames, companies, identity, end, features=features)
    promoted = [dict(deepcopy(e), entry_date=end) for e in pending]
    executable = deepcopy(previous) + promoted
    require(executable[:len(previous)] == previous, 'Historical executable candidate prefix changed')
    merged_events = None
    if event_extension_report is not None:
        require('events.parquet' in manifest['files_sha256'], 'Parent events are not hash-bound')
        merged_events = load_event_extension(event_extension_report, anchor=anchor, end=end, base=base,
                                             manifest=manifest, root=root, refs=refs)
    # Recheck the source hashes after all reads; reject a mutable source race before publishing.
    for name, expected in list(refs.items()):
        bind(root / name, root, refs, expected)
    for name in CODE_SOURCES:
        if (root / name).exists():
            bind(root / name, root, refs)
    output.mkdir(parents=True)
    for name, frame in frames.items():
        frame.reset_index(names='date').to_parquet(output / (name + '.parquet'), index=False)
    combinedq.to_parquet(output / 'quotes-unmasked.parquet', index=False)
    shutil.copyfile(base / 'companies.parquet', output / 'companies.parquet')
    if merged_events is not None:
        merged_events.to_parquet(output / 'events.parquet', index=False)
    elif 'events.parquet' in manifest['files_sha256']:
        shutil.copyfile(base / 'events.parquet', output / 'events.parquet')
    write(output / 'identity.json', identity)
    write(output / 'signals.json', dict(entries={'median50m': executable}, pending_signal_file='pending-signals.json',
        candidate_parameters_changed=False, historical_executable_prefix_exact=True,
        terminal_promotions=len(promoted), account_rejections=rejections, live_qualified=False))
    write(output / 'pending-signals.json', dict(schema='pending_last_close_signals_v1', signal_date=end,
        entries=additions, next_session_observed=False, execution_inferred=False, live_qualified=False))
    coverage = dict(date=end, parent_end=anchor, parent_universe_stocks=len(ids)-1,
        snapshot_identity_checks=identity_checks, eligible_count=int(mask.iloc[0].sum()),
        eligible_missing_raw_ids=[sid for sid in ids if mask.at[after, sid]
                                  and not (np.isfinite(indexed.at[sid, 'close']) and indexed.at[sid, 'close'] > 0)],
        bridge_missing=missing, raw_source_rows=len(raw), cohort_quote_rows=len(newq),
        outside_cohort_observed_ids=sorted(sid for sid in raw.stock_id if re.fullmatch(r'[1-9]\d{3}', sid) and sid not in ids),
        extension_two_frame_adjustments_same_provider=True, officially_crosschecked_extension=False,
        historical_frames_unchanged=True, calendar_basis=calendar_basis)
    write(output / 'extension-coverage.json', coverage)
    write(output / 'preparation-gaps.json', dict(start=end, end=end, all_data_complete=False,
        events_extension_complete=merged_events is not None,
        corporate_events_verified_through=end if merged_events is not None else
            (anchor if manifest.get('events_extension_complete') else manifest.get('historical_end')),
        complete_historical_universe=False, market_identity_basis='provisional_current_snapshot',
        actual_fill_verified=False, poc_recomputation_required=True, live_qualified=False))
    result = dict(schema='poc_latest_input_bundle_v1', created_at=datetime.now(timezone.utc).isoformat(),
        parent_bundle=str(base.relative_to(root)), source_sha256=refs,
        start=manifest['start'], end=end, historical_end=manifest.get('historical_end'),
        historical_matrices_and_quotes_exact=True, historical_executable_candidate_prefix_exact=True,
        parent_terminal_candidates_reproduced=True, parent_terminal_promotions=len(promoted),
        candidate_parameters_changed=False, executable_candidate_count=len(executable), pending_candidate_count=len(additions),
        historical_candidate_count=manifest.get('historical_candidate_count', len(previous)),
        new_sessions=1, officially_crosschecked_extension=False, complete_historical_universe=False,
        corporate_events_extended=merged_events is not None, events_extension_complete=merged_events is not None,
        event_extension_report=str(Path(event_extension_report).resolve().relative_to(root)) if event_extension_report else None,
        actual_fill_verified=False,
        upstream_source_closure_reverified=False, live_qualified=False, unseen_validation=False,
        network_requests=0, finmind_requests=0, database_mutations=False,
        limitations=list(dict.fromkeys(manifest.get('limitations', []) + [
            'Only the frozen parent stock cohort is extended; newly observed out-of-cohort IPOs are reported, not silently included',
            'Current master market matches are provisional, not daily official identity or suspension certification',
            'Both appended adjusted frames use the same provider; their agreement is not independent source verification',
            ('Corporate events extend through the appended session; new POC still requires exact-tick evidence'
             if merged_events is not None else 'Corporate events are not extended; new POC needs separate event and exact-tick evidence'),
            'Signals at the terminal close have entry_date null; no future session or fill price is invented'])),
        files_sha256={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    write(output / 'manifest.json', result)
    (output / 'manifest.sha256').write_text(digest(output / 'manifest.json') + '\n')
    return result


def parser():
    result = ArgumentParser(description=__doc__)
    for flag in ('base', 'quotes', 'adj-anchor', 'adj-end', 'market-snapshot', 'output'):
        result.add_argument('--' + flag, type=Path, required=True)
    result.add_argument('--end', required=True)
    result.add_argument('--calendar', type=Path, help='Optional observed date-column market calendar; required across multiple weekdays')
    result.add_argument('--event-extension-report', type=Path, help='Hash-bound complete six-kind official action report for appended session')
    return result


if __name__ == '__main__':
    result = extend(**vars(parser().parse_args()))
    print(json.dumps({k: result[k] for k in ('end', 'parent_terminal_promotions', 'pending_candidate_count',
                                          'executable_candidate_count', 'network_requests')}, ensure_ascii=False))
