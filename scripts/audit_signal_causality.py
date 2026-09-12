#!/usr/bin/env python3
"""Rebuild real sealed signals under preregistered truncation and future mutation."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.prepare_million_signals import build_signals

BASE = ROOT / '.cache/million-replay-signals'
SPEC = ROOT / 'docs/prereg_signal_causality_20260912.md'
NAMES = ('close-official.parquet', 'close-quality.parquet', 'raw-close.parquet', 'raw-volume.parquet')
CUTS = tuple(f'{year}-{month}' for year in range(2022, 2026)
             for month in ('03-31', '06-30', '09-30', '12-31')) + ('2026-03-31', '2026-06-30', '2026-09-08')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def project(result, cutoff):
    groups = []
    for original in result['diffusion']['groups']:
        if original['month'] <= cutoff[:7]:
            group = deepcopy(original)
            group.pop('leader_rejections', None)  # Later daily evidence is not frozen month-start information.
            groups.append(group)
    return dict(entries=[x for x in result['entries'] if x['signal_date'] <= cutoff],
                rejections=[x for x in result['rejections'] if x['signal_date'] <= cutoff],
                trend=[x for x in result['trend'] if x['date'] <= cutoff], groups=groups)


def transform(frames, cutoff, mode):
    index = frames[0].index
    cut = index.get_loc(pd.Timestamp(cutoff))
    if cut + 1 >= len(index):
        raise ValueError('A known next calendar session is required')
    result = []
    for frame in frames:
        if mode == 'truncate':
            altered = frame.iloc[:cut+2].copy()
            altered.iloc[-1, :] = np.nan
        elif mode == 'mutate':
            altered = frame.copy()
            rows = np.arange(len(index)-cut-1)[:, None]
            columns = np.arange(len(frame.columns))[None, :]
            multiplier = np.where((rows+columns) % 2 == 0, 8., .125)
            altered.iloc[cut+1:, :] = altered.iloc[cut+1:, :].to_numpy() * multiplier
        else:
            raise ValueError('Unknown future-data transformation')
        result.append(altered)
    return result


def run(output):
    if output.exists():
        raise ValueError('Output exists; use a new path, preserving previous evidence')
    started = time.perf_counter()
    manifest = json.loads((BASE/'manifest.json').read_text())
    source_names = (*NAMES, 'companies.parquet', 'signals.json')
    expected = {BASE/name: manifest['files_sha256'][name] for name in source_names}
    expected.update({ROOT/name: value for name, value in manifest['code_sha256'].items()})
    for path, value in expected.items():
        if not path.is_file() or sha(path) != value:
            raise ValueError('Sealed source/code changed: '+str(path))
    frames = [pd.read_parquet(BASE/name).set_index('date') for name in NAMES]
    for frame in frames:
        frame.index = pd.to_datetime(frame.index)
    companies = pd.read_parquet(BASE/'companies.parquet')
    sealed = json.loads((BASE/'signals.json').read_text())
    start, end = sealed['start'], sealed['signal_end']
    full = build_signals(*frames, companies, start=start, signal_end=end)
    identical = all(full[key] == sealed[key] for key in ('entries', 'rejections'))
    result = dict(spec_sha256=sha(SPEC), audit_code_sha256=sha(__file__),
                  source_sha256={str(p.relative_to(ROOT)): v for p, v in expected.items()},
                  manifest_sha256=sha(BASE/'manifest.json'), full_rebuild_matches_sealed=identical,
                  accepted_candidates=len(full['entries']), rejected_candidates=len(full['rejections']),
                  cases=[], live_qualified=False, finmind_requests=0,
                  scope='Signal causality on the supplied snapshot; not publication-time, universe or execution certification.')
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result['elapsed_seconds'] = round(time.perf_counter()-started, 3)
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        temp.replace(output)

    if not identical:
        result['passed'] = False
        save()
        return result
    for requested in CUTS:
        dates = frames[0].index[frames[0].index <= pd.Timestamp(requested)]
        cutoff = str(dates[-1].date())
        target = project(full, cutoff)
        for mode in ('truncate', 'mutate'):
            altered = transform(frames, cutoff, mode)
            rebuilt = build_signals(*altered, companies, start=start, signal_end=cutoff)
            actual = project(rebuilt, cutoff)
            differences = [key for key in target if actual[key] != target[key]]
            result['cases'].append(dict(requested_cutoff=requested, cutoff=cutoff, mode=mode,
                accepted_candidates=len(actual['entries']), passed=not differences,
                differing_sections=differences, expected_sha256=digest(target), actual_sha256=digest(actual)))
            save()
            print(cutoff, mode, 'PASS' if not differences else 'FAIL '+str(differences), flush=True)
            del altered, rebuilt, actual
    result['sources_unchanged_after_run'] = all(sha(path) == value for path, value in expected.items())
    result['passed'] = identical and result['sources_unchanged_after_run'] and all(x['passed'] for x in result['cases'])
    save()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'source_sha256')}, ensure_ascii=False, indent=2))
    sys.exit(0 if result['passed'] else 1)
