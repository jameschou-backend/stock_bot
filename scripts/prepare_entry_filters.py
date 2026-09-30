#!/usr/bin/env python3
"""Freeze all entry-filter decisions and audit historical failures without tuning."""
from pathlib import Path
import argparse,json,sys
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read,write,sha
from skills.entry_filters import filter_features,apply_filters


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Choose new decision output')
    base=ROOT/'.cache/partial-risk-2019-20260929/inputs-final'
    manifest=read(base/'manifest.json');refs={str((base/'manifest.json').relative_to(ROOT)):sha(base/'manifest.json')}
    for n,h in manifest['files_sha256'].items():
        if sha(base/n)!=h:raise ValueError('Changed input '+n)
        refs[str((base/n).relative_to(ROOT))]=h
    frames={n:pd.read_parquet(base/(n+'.parquet')).set_index('date') for n in ('close-official','close-quality','raw-close','raw-volume','eligibility')}
    companies=pd.read_parquet(base/'companies.parquet');quotes=pd.read_parquet(base/'quotes-unmasked.parquet')
    source=ROOT/'.cache/stock-universe-2019-20260929/signals-v2.json';refs[str(source.relative_to(ROOT))]=sha(source)
    entries=read(source)['entries']['liquid_universe'];features=filter_features(frames,companies,quotes)
    selected,rows=apply_filters(entries,features);checks=[]
    for cutoff in ('2020-06-30','2022-12-30','2024-06-28','2025-12-31'):
        partial=filter_features({k:v.loc[:cutoff].copy() for k,v in frames.items()},companies,quotes.loc[quotes.date<=cutoff])
        sliced,detail=apply_filters(entries,partial,cutoff)
        if detail!=[r for r in rows if r['signal_date']<=cutoff]:raise ValueError('Filter leaks future '+cutoff)
        for arm in selected:
            if sliced[arm]!=[e for e in selected[arm] if e['signal_date']<=cutoff]:raise ValueError('Ordering changed')
        checks.append(cutoff)
    for path in (Path(__file__),ROOT/'skills/entry_filters.py',ROOT/'docs/prereg_entry_filters_20260930.md'):
        refs[str(path.relative_to(ROOT))]=sha(path)
    write(a.output,dict(entries=selected,decisions=rows,source_sha256=refs,prefix_checks=checks,
                       live_qualified=False,unseen_validation=False))
    print({k:len(v) for k,v in selected.items()},flush=True)


if __name__=='__main__':main()
