#!/usr/bin/env python3
"""Freeze causal RSI candidates for the registered cash-account comparison.

The scanner's existing Wilder RSI rule is reused unchanged. This preparation
does not execute trades, query providers, or infer a session after the source end.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from skills.strategy_scanner.data import load_bundle
from skills.strategy_scanner.engine import _compile_rules, _day, _prepare
from skills.strategy_scanner.public_rules import MIN_AMOUNT20, VERSION

START, END = '2024-01-02', '2026-10-02'
STRATEGY = 'rsi14_reclaim30'
SCHEMA = 'strategy_account_comparison_entries_v1'
MANIFEST_SCHEMA = 'strategy_account_comparison_inputs_v1'
DEFAULT_BUNDLE = ROOT / '.cache/poc-latest-20261003/inputs-v1'
DEFAULT_OUTPUT = ROOT / '.cache/strategy-account-comparison-20261007/inputs-v1'
PARAMETERS = dict(
    rsi_period=14, rsi_threshold=30, rsi_smoothing='Wilder_SMA_seed_alpha_1_over_14',
    rsi_comparison='current_ge_30_and_previous_lt_30',
    amount20_min=MIN_AMOUNT20, amount_basis='raw_close_times_shares_20_session_mean',
    signal_policy='known_first_day_only_T_close',
    entry_policy='next_observed_market_session_only',
    boundary_policy='include_previous_session_signal_entering_on_account_start',
    priority='signal_pub_amount20_desc_then_stock_id_asc',
    warmup='entire_hash_bound_source_history_no_truncation',
    threshold_tuning=False, portfolio_filtering=False,
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _inside(root, path):
    root = Path(root).resolve()
    path = Path(path)
    if '..' in path.parts:
        raise ValueError('Candidate source path must not contain traversal')
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Candidate source path escapes repository')
    return path


def prepare_entries(bars, calendar, *, start=START, end=END, batch_size=128):
    """Return candidate payload and complete observed calendar; no I/O or fills.

    Columns are independent for this rule, so bounded stock batches reuse the
    full scanner compiler without retaining all strategies for the full market.
    Batch size changes memory use only, not rule parameters or signal history.
    """
    start_day, end_day = _day(start), _day(end)
    days = pd.DatetimeIndex(calendar)
    if (days.hasnans or days.tz is not None or days.has_duplicates
            or not days.is_monotonic_increasing or not days.equals(days.normalize())):
        raise ValueError('Candidate calendar must contain ordered unique market dates')
    days = days[days <= end_day]
    if start_day > end_day or start_day not in days or end_day not in days:
        raise ValueError('Account endpoints must be observed market sessions')
    first = int(days.get_loc(start_day))
    if first == 0:
        raise ValueError('Account start needs a preceding observed signal session')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Candidate batch size must be a positive integer')
    signal_start = days[first - 1]
    dates = days.strftime('%Y-%m-%d').tolist()
    positions = {day: i for i, day in enumerate(dates)}
    stock_values = bars['stock_id']
    if not stock_values.map(lambda x: isinstance(x, str) and re.fullmatch(r'\d{4}', x) is not None).all():
        raise ValueError('Candidate stock codes must be four-digit strings')
    individual = sorted(sid for sid in stock_values.unique() if not sid.startswith('0'))
    entries, pending, matching_days, prior_unknown = [], [], 0, 0
    for offset in range(0, len(individual), batch_size):
        batch = individual[offset:offset + batch_size]
        f, actual_days, ids = _prepare(bars.loc[stock_values.isin(batch)], days, end_day)
        z, masks, _, _ = _compile_rules(f, actual_days, ids)
        match, known, _, _ = masks[STRATEGY]
        available = known & f['eligible'].eq(True).fillna(False)
        matched = match & available
        previous_known = available.shift(1, fill_value=False)
        first_signal = matched & previous_known & ~match.shift(1, fill_value=False)
        in_scope = (days >= signal_start) & (days <= end_day)
        matching_days += int(matched.loc[in_scope].to_numpy().sum())
        prior_unknown += int((matched & ~previous_known).loc[in_scope].to_numpy().sum())
        selected = first_signal.to_numpy(bool)
        selected[~in_scope] = False
        for i, j in zip(*np.where(selected)):
            sid, signal = ids[j], dates[i]
            next_day = dates[i + 1] if i + 1 < len(dates) else None
            amount = float(z['pub_amount20'].iat[i, j])
            if not np.isfinite(amount) or amount < MIN_AMOUNT20:
                raise ValueError('Compiled RSI candidate lacks its declared liquidity input')
            before_month = days[days < pd.Timestamp(signal[:7] + '-01')]
            cutoff = str(before_month[-1].date()) if len(before_month) else None
            row = dict(
                event_id=f'{STRATEGY}-{signal}-{sid}', signal_date=signal,
                entry_date=next_day, members=[sid], priority=amount,
                group_id=f'{STRATEGY}-{signal[:7]}', group_members=[sid],
                group_cutoff_date=cutoff,
                selection_reason='Wilder RSI14 reclaims 30; known first signal; raw amount20 >= 50m',
                leader_evidence=dict(
                    rsi14=float(z['pub_rsi14'].iat[i, j]),
                    previous_rsi14=float(z['pub_prior_rsi14'].iat[i, j]),
                    amount20=amount, ranking_basis='signal_pub_amount20',
                    information_cutoff=signal,
                ),
            )
            if next_day is None:
                if signal != end:
                    raise ValueError('Only the final observed signal may await entry')
                pending.append(row)
            elif start <= next_day <= end:
                if positions[next_day] != positions[signal] + 1:
                    raise ValueError('Candidate must enter on observed T+1')
                entries.append(row)
        del f, z, masks
    order = lambda row: (row['signal_date'], -row['priority'], row['members'][0])
    entries.sort(key=order)
    pending.sort(key=order)
    all_rows = entries + pending
    if len({r['event_id'] for r in all_rows}) != len(all_rows):
        raise ValueError('Duplicate RSI candidate identity')
    payload = dict(
        schema=SCHEMA, version=VERSION, strategy_id=STRATEGY, start=start, end=end,
        signal_start=str(signal_start.date()), entries=entries, pending_entries=pending,
        parameters=deepcopy(PARAMETERS),
        counts=dict(executable=len(entries), pending=len(pending),
                    boundary_before_start=sum(r['signal_date'] < start for r in entries),
                    matching_stock_days=matching_days, matched_prior_unknown=prior_unknown,
                    executable_by_signal_year=dict(Counter(r['signal_date'][:4] for r in entries))),
        source_history_start=dates[0], source_history_sessions=len(days),
        network_requests=0, database_mutations=False, live_qualified=False,
        unseen_validation=False,
    )
    return payload, dates


def write_inputs(output, payload, calendar, source_sha256, *, root=ROOT):
    """Publish immutable candidates only after verifying their bound source bytes."""
    output = _inside(root, output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new empty candidate output directory')
    for name, expected in source_sha256.items():
        if digest(_inside(root, name)) != expected:
            raise ValueError('Candidate input changed before publication: ' + name)
    value = deepcopy(payload)
    value['source_sha256'] = dict(sorted(source_sha256.items()))
    output.mkdir(parents=True, exist_ok=True)
    _write(output / 'rsi-entries.json', value)
    _write(output / 'calendar.json', calendar)
    manifest = dict(schema=MANIFEST_SCHEMA, strategy_id=STRATEGY,
                    start=value['start'], end=value['end'], parameters=value['parameters'],
                    counts=value['counts'], source_sha256=value['source_sha256'],
                    files_sha256={name: digest(output / name)
                                  for name in ('rsi-entries.json', 'calendar.json')})
    _write(output / 'manifest.json', manifest)
    (output / 'manifest.sha256').write_text(digest(output / 'manifest.json') + '\n')
    return manifest


def load_candidates(directory, *, verify_sources=True, root=ROOT):
    """Load bound candidate payload/calendar/manifest; no silent stale cache use."""
    directory = _inside(root, directory)
    manifest_path = directory / 'manifest.json'
    if digest(manifest_path) != (directory / 'manifest.sha256').read_text().strip():
        raise ValueError('Candidate manifest SHA mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != MANIFEST_SCHEMA or set(manifest.get('files_sha256', {})) != {'rsi-entries.json', 'calendar.json'}:
        raise ValueError('Unsupported candidate manifest')
    for name, expected in manifest['files_sha256'].items():
        if digest(directory / name) != expected:
            raise ValueError('Candidate output SHA mismatch: ' + name)
    payload = json.loads((directory / 'rsi-entries.json').read_text())
    calendar = json.loads((directory / 'calendar.json').read_text())
    if (payload.get('schema') != SCHEMA or payload.get('strategy_id') != STRATEGY
            or payload.get('parameters') != PARAMETERS
            or payload.get('source_sha256') != manifest.get('source_sha256')
            or any(payload.get(key) != manifest.get(key) for key in ('start', 'end', 'counts', 'parameters'))):
        raise ValueError('Candidate payload/manifest contract mismatch')
    if verify_sources:
        for name, expected in manifest['source_sha256'].items():
            if digest(_inside(root, name)) != expected:
                raise ValueError('Candidate source SHA mismatch: ' + name)
    positions = {day: i for i, day in enumerate(calendar)}
    if not calendar or len(positions) != len(calendar) or calendar != sorted(calendar) or calendar[-1] != payload['end']:
        raise ValueError('Candidate observed calendar changed')
    seen = set()
    for pending, rows in ((False, payload['entries']), (True, payload['pending_entries'])):
        if rows != sorted(rows, key=lambda r: (r['signal_date'], -r['priority'], r['members'][0])):
            raise ValueError('Candidate priority ordering changed')
        for row in rows:
            sid, signal = row['members'], row['signal_date']
            if len(sid) != 1 or not re.fullmatch(r'[1-9]\d{3}', sid[0]) or row['event_id'] in seen:
                raise ValueError('Candidate stock/event identity changed')
            seen.add(row['event_id'])
            if signal not in positions or not payload['signal_start'] <= signal <= payload['end']:
                raise ValueError('Candidate signal outside observed period')
            i = positions[signal]
            expected = calendar[i + 1] if i + 1 < len(calendar) else None
            if row['entry_date'] != expected or pending != (expected is None):
                raise ValueError('Candidate entry must be observed T+1 or terminal pending')
            if not pending and not payload['start'] <= expected <= payload['end']:
                raise ValueError('Executable candidate outside account period')
    if payload['counts']['executable'] != len(payload['entries']) or payload['counts']['pending'] != len(payload['pending_entries']):
        raise ValueError('Candidate counts changed')
    return payload, calendar, manifest


def prepare(bundle=DEFAULT_BUNDLE, output=DEFAULT_OUTPUT):
    bundle = _inside(ROOT, bundle)
    # Read the authenticated manifest's beginning, not an arbitrary 420-day seed.
    manifest_path = bundle / 'manifest.json'
    if digest(manifest_path) != (bundle / 'manifest.sha256').read_text().strip():
        raise ValueError('Source bundle manifest SHA mismatch')
    source = json.loads(manifest_path.read_text())
    data = load_bundle(bundle, start=source['start'], end=END)
    payload, calendar = prepare_entries(data['bars'], data['calendar'])
    payload['input_bundle'] = str(bundle.relative_to(ROOT))
    payload['source_provenance'] = deepcopy(data['provenance'])
    refs = {str((bundle / name).relative_to(ROOT)): value
            for name, value in data['provenance']['source_hashes'].items()}
    for path in [Path(__file__), *sorted((ROOT / 'skills/strategy_scanner').glob('*.py'))]:
        refs[str(path.relative_to(ROOT))] = digest(path)
    manifest = write_inputs(output, payload, calendar, refs)
    print(json.dumps(dict(output=str(Path(output).resolve()), **payload['counts'],
                          manifest_sha256=digest(Path(output) / 'manifest.json'), network_requests=0,
                          live_qualified=False), ensure_ascii=False))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.bundle, args.output)
