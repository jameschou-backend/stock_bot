#!/usr/bin/env python3
"""Bind liquidity screens to the sealed expanded-universe candidate population."""
from pathlib import Path
import argparse
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha
from skills.liquidity_candidates import filter_candidates
from skills.liquidity_universe import expanded_entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    a.output.resolve().relative_to(ROOT)
    if a.output.exists():
        raise ValueError('Preserve the existing candidate ledger')
    base = ROOT/'.cache/partial-risk-2019-20260929/inputs-final'
    source = ROOT/'.cache/stock-universe-2019-20260929/signals-v2.json'
    parent = ROOT/'.cache/stock-universe-2019-20260929/final-a/report.json'
    report, prepared = read(parent), read(source)
    if sha(source) != report['source_sha256'][str(source.relative_to(ROOT))]:
        raise ValueError('Expanded candidate source changed')
    manifest = read(base/'manifest.json')
    refs = {str((base/name).relative_to(ROOT)): h for name, h in manifest['files_sha256'].items()}
    refs.update(prepared['source_sha256'])
    for name, digest in refs.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen input changed: '+name)
    days = pd.read_parquet(base/'eligibility.parquet').set_index('date').index
    q = pd.read_parquet(base/'quotes-unmasked.parquet')
    raw, volume = [q.pivot(index='date', columns='stock_id', values=k).reindex(days)
                   for k in ('close', 'volume')]
    original = prepared['entries']['liquid_universe']
    if len(original) != 37993:
        raise ValueError('Expanded population differs from the registered source')
    filtered, decisions = filter_candidates(raw, volume, original, '2026-09-08')
    entries = expanded_entries(filtered)
    checks = []
    for cutoff in ('2020-06-30', '2022-12-30', '2024-06-28', '2025-12-31'):
        last = days[days > pd.Timestamp(cutoff)][0]
        prefix, diagnostic = filter_candidates(raw.loc[:last], volume.loc[:last], original, cutoff)
        prefix = expanded_entries(prefix)
        if any([e for e in entries[arm] if e['signal_date'] <= cutoff] != prefix[arm] for arm in entries):
            raise ValueError('Future truncation changed candidates')
        if [e for e in decisions if e['signal_date'] <= cutoff] != diagnostic:
            raise ValueError('Future truncation changed feature evidence')
        checks.append(cutoff)
    for p in (source, parent, base/'manifest.json', Path(__file__), ROOT/'skills/liquidity_candidates.py',
              ROOT/'skills/liquidity_diagnostics.py', ROOT/'skills/liquidity_universe.py',
              ROOT/'docs/prereg_liquidity_universe_20261001.md'):
        refs[str(p.relative_to(ROOT))] = sha(p)
    write(a.output, dict(entries=entries, decisions=decisions, prefix_checks=checks,
        source_sha256=refs, live_qualified=False, unseen_validation=False))
    print({arm: len(rows) for arm, rows in entries.items()}, flush=True)


if __name__ == '__main__':
    main()
