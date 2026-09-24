#!/usr/bin/env python3
"""Audit the quarantined 458-signal snapshot offline; never calculate returns."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import socket
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow
import scipy

from scripts.audit_signal_causality import CUTS, NAMES, digest, project, sha, transform
from scripts.prepare_million_signals import build_signals

BASE = ROOT / '.cache/five-axis-20260913/rebuild'
ORIGINAL = ROOT / '.cache/million-replay-signals'
SPEC = ROOT / 'docs/prereg_current_causality_20260925.md'
TEST = ROOT / 'tests/test_current_causality_20260925.py'
QUARANTINE = ROOT / 'artifacts/forward_simulation/price_chronology_20260913.json'


def read(path):
    return json.loads(Path(path).read_text())


def deterministic(result):
    """The one explicitly excluded field is wall-clock runtime, not evidence."""
    value = deepcopy(result)
    value['diffusion']['stats'].pop('seconds', None)
    return value


def as_of(result, cutoff):
    value = project(result, cutoff)
    value['group_leader_rejections'] = [
        {'month': group['month'], 'rows': [row for row in group['leader_rejections']
                                         if row['date'] <= cutoff]}
        for group in result['diffusion']['groups'] if group['month'] <= cutoff[:7]]
    return value


def cutoff_plan(index, start, end):
    """All month transitions and original quarter dates, selected without returns."""
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError('Cutoffs require a unique ordered calendar')
    selected = {}

    def add(day, label):
        if day < pd.Timestamp(start) or day > pd.Timestamp(end):
            raise ValueError('Cutoff is outside the sealed signal interval')
        position = index.get_loc(day)
        if position + 1 >= len(index):
            raise ValueError('Every cutoff requires a known next market session')
        key = str(day.date())
        selected.setdefault(key, {'cutoff': key, 'labels': [],
                                 'next_session': str(index[position + 1].date())})['labels'].append(label)

    for requested in CUTS:
        dates = index[index <= pd.Timestamp(requested)]
        if len(dates) == 0:
            raise ValueError('Original quarter cutoff predates the calendar')
        add(dates[-1], 'original_quarter:' + requested)
    for position, day in enumerate(index):
        if not pd.Timestamp(start) <= day <= pd.Timestamp(end):
            continue
        month = day.to_period('M')
        if position == 0 or index[position - 1].to_period('M') != month:
            add(day, 'month_first:' + str(month))
        if position + 1 < len(index) and index[position + 1].to_period('M') != month:
            add(day, 'month_last:' + str(month))
    add(pd.Timestamp(end), 'signal_end')
    return [selected[key] for key in sorted(selected)]


def verify_hashes(expected):
    failures = [str(path.relative_to(ROOT)) for path, value in expected.items()
                if not path.is_file() or sha(path) != value]
    if failures:
        raise ValueError('Sealed source/code changed or missing: ' + ', '.join(failures))


def provenance():
    manifest = read(BASE / 'manifest.json')
    identity = read(BASE / 'identity.json')
    required = {*NAMES, 'signals.json', 'summary.json', 'identity.json'}
    if not required.issubset(manifest['files_sha256']):
        raise ValueError('Rebuilt manifest is missing required source hashes')
    expected = {BASE / name: value for name, value in manifest['files_sha256'].items()}
    expected.update({ROOT / name: value for name, value in identity['input_sha256'].items()})
    companies_path = ORIGINAL / 'companies.parquet'
    if companies_path not in expected:
        raise ValueError('The rebuilt identity does not seal the original company cohort')
    expected[ROOT / 'scripts/prepare_five_axis.py'] = identity['code_sha256']
    expected[ROOT / 'docs/prereg_five_axis_20260913.md'] = identity['spec_sha256']
    for path in (BASE / 'manifest.json', ORIGINAL / 'manifest.json', SPEC, Path(__file__),
                 TEST, ROOT / 'scripts/audit_signal_causality.py',
                 ROOT / 'docs/prereg_signal_causality_20260912.md'):
        expected[path] = sha(path)
    verify_hashes(expected)
    return expected


def matrices(folder):
    result = [pd.read_parquet(folder / name).set_index('date') for name in NAMES]
    for frame in result:
        frame.index = pd.to_datetime(frame.index)
    return result


def verify_matrix_recipe(frames):
    original = matrices(ORIGINAL)
    applied = []
    for row in read(QUARANTINE)['quarantine']:
        sid, day = row['stock_id'], pd.Timestamp(row['date'])
        if sid not in original[0].columns or day not in original[0].index:
            continue
        for frame in original:
            frame.at[day, sid] = np.nan
        applied.append({'stock_id': sid, 'date': row['date']})
    for name, expected, actual in zip(NAMES, original, frames):
        if not expected.equals(actual):
            raise ValueError('Quarantine recipe cannot reproduce rebuilt matrix: ' + name)
    return {'passed': True, 'quarantined_rows': applied, 'matrix_count': len(frames)}


@contextmanager
def offline_only(result):
    def denied(*args, **kwargs):
        result['network_attempts_blocked'] += 1
        raise RuntimeError('Network access is prohibited by this offline audit')
    with patch.object(socket.socket, 'connect', denied), \
            patch.object(socket.socket, 'connect_ex', denied), \
            patch.object(socket.socket, 'sendto', denied), \
            patch.object(socket, 'getaddrinfo', denied), \
            patch.object(socket, 'create_connection', denied):
        yield


def run(output, *, max_seconds=600):
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; use a new path to preserve previous evidence')
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('max_seconds must be positive and finite')
    started = time.perf_counter()
    result = {'schema': 1, 'audit': 'current_causality_20260925',
              'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'passed': False, 'complete': False, 'live_qualified': False,
              'historical_universe_verified': False,
              'financial_publication_versions_verified': False, 'execution_verified': False,
              'publication_time_price_versions_verified': False,
              'historical_calendar_availability_verified': False,
              'portfolio_returns_computed': False, 'finmind_requests': 0,
              'network_attempts_blocked': 0, 'max_seconds': max_seconds, 'cases': [],
              'runtime_versions': {'python': platform.python_version(), 'numpy': np.__version__,
                                   'pandas': pd.__version__, 'scipy': scipy.__version__,
                                   'pyarrow': pyarrow.__version__},
              'scope': 'Causality of fixed rebuilt snapshot; no universe, publication or execution certification.'}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result['elapsed_seconds'] = round(time.perf_counter() - started, 3)
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(output)

    def budget():
        if time.perf_counter() - started > max_seconds:
            raise TimeoutError('Audit time budget exhausted; incomplete checks cannot pass')

    expected = {}
    try:
        with offline_only(result):
            result['stage'] = 'verify_provenance'
            expected = provenance()
            result['source_and_code_sha256'] = {str(path.relative_to(ROOT)): value
                                              for path, value in expected.items()}
            budget()
            frames = matrices(BASE)
            companies = pd.read_parquet(ORIGINAL / 'companies.parquet')
            sealed = read(BASE / 'signals.json')
            result['matrix_recipe'] = verify_matrix_recipe(frames)
            result['signal_interval'] = {'start': sealed['start'], 'end': sealed['signal_end']}
            result['input_matrix'] = {'first_date': str(frames[0].index[0].date()),
                                      'last_date': str(frames[0].index[-1].date()),
                                      'sessions': len(frames[0]), 'columns': len(frames[0].columns),
                                      'companies': len(companies)}
            result['cutoff_plan'] = cutoff_plan(frames[0].index, sealed['start'], sealed['signal_end'])
            result['expected_case_count'] = 2 * len(result['cutoff_plan'])
            result['stage'] = 'full_rebuild'
            budget()
            full = build_signals(*frames, companies, start=sealed['start'], signal_end=sealed['signal_end'])
            target_full, actual_full = deterministic(sealed), deterministic(full)
            different = [key for key in target_full.keys() | actual_full.keys()
                         if target_full.get(key) != actual_full.get(key)]
            result['full_rebuild'] = {'passed': not different, 'differing_sections': sorted(different),
                                      'expected_sha256': digest(target_full), 'actual_sha256': digest(actual_full)}
            result['accepted_candidates'] = len(full['entries'])
            result['rejected_candidates'] = len(full['rejections'])
            if different or len(full['entries']) != 458:
                raise ValueError('Full rebuilt 458-signal snapshot cannot be reproduced exactly')
            del target_full, actual_full
            save()
            result['stage'] = 'future_isolation'
            for planned in result['cutoff_plan']:
                cutoff = planned['cutoff']
                target = as_of(full, cutoff)
                target_hashes = {key: digest(value) for key, value in target.items()}
                for mode in ('truncate', 'mutate'):
                    budget()
                    tick = time.perf_counter()
                    altered = transform(frames, cutoff, mode)
                    if any(not original.loc[:cutoff].equals(changed.loc[:cutoff])
                           for original, changed in zip(frames, altered)):
                        raise ValueError('Future transformation modified historical input')
                    rebuilt = build_signals(*altered, companies, start=sealed['start'], signal_end=cutoff)
                    actual = as_of(rebuilt, cutoff)
                    differences = [key for key in target if actual[key] != target[key]]
                    result['cases'].append({**planned, 'mode': mode, 'passed': not differences,
                                            'accepted_candidates': len(actual['entries']),
                                            'rejected_candidates': len(actual['rejections']),
                                            'differing_sections': differences,
                                            'expected_sha256': digest(target), 'actual_sha256': digest(actual),
                                            'expected_sections_sha256': target_hashes,
                                            'actual_sections_sha256': {key: digest(value) for key, value in actual.items()},
                                            'elapsed_seconds': round(time.perf_counter() - tick, 3)})
                    save()
                    print(cutoff, mode, 'PASS' if not differences else 'FAIL ' + str(differences), flush=True)
                    del altered, rebuilt, actual
            budget()
            result['complete'] = len(result['cases']) == result['expected_case_count']
            result['passed'] = (result['complete'] and all(row['passed'] for row in result['cases'])
                                and result['network_attempts_blocked'] == 0)
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        result['passed'] = False
    finally:
        if expected:
            try:
                verify_hashes(expected)
                result['sources_unchanged_after_run'] = True
            except Exception as exc:
                result['sources_unchanged_after_run'] = False
                result['source_verification_error'] = str(exc)
                result['passed'] = False
        else:
            result['sources_unchanged_after_run'] = False
        result['status'] = 'passed' if result['passed'] else 'failed'
        save()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=600)
    args = parser.parse_args()
    report = run(args.output, max_seconds=args.max_seconds)
    print(json.dumps({key: report.get(key) for key in ('status', 'complete', 'accepted_candidates',
                     'rejected_candidates', 'expected_case_count', 'elapsed_seconds', 'error')}, indent=2))
    sys.exit(0 if report['passed'] else 1)
