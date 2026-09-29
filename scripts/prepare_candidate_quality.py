#!/usr/bin/env python3
"""Freeze candidates and verify that truncating future inputs changes no past decisions."""
from pathlib import Path
import sys
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha
from skills.candidate_quality import generate_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior candidate ledger')
    base = ROOT/'.cache/partial-risk-2019-20260929/inputs-final'
    manifest = read(base/'manifest.json')
    for name, digest in manifest['files_sha256'].items():
        if sha(base/name) != digest:
            raise ValueError('Frozen input changed '+name)
    frames = {n: pd.read_parquet(base/(n+'.parquet')).set_index('date') for n in
        ('close-official', 'close-quality', 'raw-close', 'raw-volume', 'eligibility')}
    companies = pd.read_parquet(base/'companies.parquet')
    signals = read(base/'signals.json')['entries']
    entries = generate_candidates(frames, companies, signals, '2026-09-08')
    checks = []
    for cutoff in ('2020-06-30', '2022-12-30', '2024-06-28', '2025-12-31'):
        days = frames['close-official'].index
        last = days[days > pd.Timestamp(cutoff)][0]
        truncated = generate_candidates({k: v.loc[:last].copy() for k, v in frames.items()},
            companies, signals, cutoff)
        for arm in entries:
            if [e for e in entries[arm] if e['signal_date'] <= cutoff] != truncated[arm]:
                raise ValueError('Future truncation changes '+arm+':'+cutoff)
        checks.append(cutoff)
    refs = manifest['source_sha256'] | {str((base/n).relative_to(ROOT)): h for n, h in manifest['files_sha256'].items()}
    for p in (base/'manifest.json', Path(__file__), ROOT/'skills/candidate_quality.py',
              ROOT/'docs/prereg_candidate_quality_20260929.md'):
        refs[str(p.relative_to(ROOT))] = sha(p)
    write(args.output, dict(entries=entries, prefix_checks=checks, source_sha256=refs,
                           live_qualified=False, unseen_validation=False))
    print({k: len(v) for k, v in entries.items()}, flush=True)


if __name__ == '__main__':
    main()
