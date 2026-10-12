#!/usr/bin/env python3
"""Seal the original expanded-stock rule on the common extended daily inputs."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from skills.all_entry_accounts import digest, write, verify_refs
from skills.legacy_return_comparison import BASE, SPEC
from skills.stock_universe_2019 import generate


def prepare(output=BASE/'candidates-v1'):
    if output.exists():
        raise ValueError('Choose a new candidate directory')
    bundle = ROOT/'.cache/poc-latest-20261003/inputs-v1'
    manifest = json.loads((bundle/'manifest.json').read_text())
    refs = dict(manifest['source_sha256'])
    refs.update({str((bundle/n).relative_to(ROOT)):h for n,h in manifest['files_sha256'].items()})
    old_path = ROOT/'.cache/stock-universe-2019-20260929/signals-v2.json'
    old = json.loads(old_path.read_text())
    refs.update(old['source_sha256'])
    parent = ROOT/'.cache/partial-risk-2019-20260929/inputs-final/signals.json'
    for path in (Path(__file__), SPEC, old_path, parent, bundle/'manifest.json',
                 ROOT/'skills/stock_universe_2019.py', ROOT/'skills/legacy_return_comparison.py'):
        refs[str(path.relative_to(ROOT))] = digest(path)
    verify_refs(refs)
    frames = {n:pd.read_parquet(bundle/(n+'.parquet')).set_index('date') for n in
              ('close-official','close-quality','raw-close','raw-volume','eligibility')}
    companies = pd.read_parquet(bundle/'companies.parquet')
    signals = json.loads(parent.read_text())
    days = frames['close-official'].index
    groups = signals['diffusion']['groups']
    months = {r['month'] for r in groups}
    for month in sorted(set(str(d.to_period('M')) for d in days if d >= pd.Timestamp('2019-01-01')) - months):
        first = pd.Timestamp(month+'-01')
        groups.append(dict(month=month,cutoff_date=str(days[days<first][-1].date()),selected_ids=[],exclusions={}))
    entries = generate(frames,companies,signals,'2026-10-01')['liquid_universe']
    checks=[]
    for cutoff in ('2024-12-31','2025-12-31'):
        last=days[days>pd.Timestamp(cutoff)][0]
        prefix=generate({k:v.loc[:last].copy() for k,v in frames.items()},companies,signals,cutoff)['liquid_universe']
        if prefix != [e for e in entries if e['signal_date']<=cutoff]:
            raise ValueError('Future truncation changed candidates')
        checks.append(cutoff)
    previous={e['event_id']:e for e in old['entries']['liquid_universe'] if e['entry_date']>='2024-01-02'}
    current={e['event_id']:e for e in entries if '2024-01-02'<=e['entry_date']<='2026-09-09'}
    changes=dict(old_count=len(previous),current_count=len(current),
        added=sorted(current.keys()-previous.keys()),removed=sorted(previous.keys()-current.keys()),
        changed=[eid for eid in sorted(previous.keys()&current.keys()) if previous[eid]!=current[eid]])
    active=[e for e in entries if '2024-01-02'<=e['entry_date']<='2026-10-02']
    verify_refs(refs)
    payload=dict(entries=active,calendar=[str(d.date()) for d in days],prefix_checks=checks,
        old_candidate_comparison=changes,source_sha256=refs,outcomes_read=False,
        start='2024-01-02',end='2026-10-02',ranking='relative20_desc_event_id_asc')
    write(output/'candidates.json',payload)
    (output/'candidates.sha256').write_text(digest(output/'candidates.json')+'\n')
    print(json.dumps(dict(candidates=len(active),old_comparison={k:len(v) if isinstance(v,list) else v for k,v in changes.items()})))
    return payload


if __name__=='__main__': prepare()
