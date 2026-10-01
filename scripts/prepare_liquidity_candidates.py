#!/usr/bin/env python3
"""Freeze fixed liquidity candidates, binding to the original frozen inputs."""
from pathlib import Path
import argparse
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha
from skills.liquidity_candidates import filter_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise ValueError('Preserve the frozen candidate ledger')
    base = ROOT/'.cache/partial-risk-2019-20260929/inputs-final'
    manifest = read(base/'manifest.json')
    for name, digest in manifest['files_sha256'].items():
        if sha(base/name) != digest:
            raise ValueError('Frozen input changed: '+name)
    # Use the same unmasked raw quotes as the preceding independent-signal study.
    days = pd.read_parquet(base/'eligibility.parquet').set_index('date').index
    q = pd.read_parquet(base/'quotes-unmasked.parquet')
    raw, volume = [q.pivot(index='date', columns='stock_id', values=k).reindex(days)
                   for k in ('close', 'volume')]
    original = read(base/'signals.json')['entries']
    entries, decisions = filter_candidates(raw, volume, original, '2026-09-08')
    checks = []
    for cutoff in ('2020-06-30', '2022-12-30', '2024-06-28', '2025-12-31'):
        last = days[days > pd.Timestamp(cutoff)][0]
        prefix, diagnostic = filter_candidates(raw.loc[:last], volume.loc[:last], original, cutoff)
        if any([e for e in entries[arm] if e['signal_date'] <= cutoff] != prefix[arm]
               for arm in entries):
            raise ValueError('Future truncation changed candidates')
        if [e for e in decisions if e['signal_date'] <= cutoff] != diagnostic:
            raise ValueError('Future truncation changed feature evidence')
        checks.append(cutoff)
    refs = manifest['source_sha256'] | {
        str((base/name).relative_to(ROOT)): h for name, h in manifest['files_sha256'].items()}
    for p in (base/'manifest.json', Path(__file__), ROOT/'skills/liquidity_candidates.py',
              ROOT/'skills/liquidity_diagnostics.py', ROOT/'docs/prereg_liquidity_account_20261001.md'):
        refs[str(p.relative_to(ROOT))] = sha(p)
    write(output, dict(entries=entries, decisions=decisions, prefix_checks=checks,
        source_sha256=refs, live_qualified=False, unseen_validation=False))
    print({arm: len(rows) for arm, rows in entries.items()}, flush=True)


if __name__ == '__main__':
    main()
