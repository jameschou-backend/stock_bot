#!/usr/bin/env python3
"""Fill uncomputed peer coordinates with the unchanged, causal peer definition.

Price matrices are byte-identical to the original peer study. Every previously
covered coordinate must reproduce before the expanded candidate set is sealed.
No account outcome or forward label is loaded.
"""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from skills.all_entry_accounts import BASE,BUNDLE,START,END,CONTEXT_REPORT,CONTEXT_SHA,digest,write,verify_refs,event
from scripts.research_sector_participation_20261003 import participation_features


def run(source=BASE/'inputs-v2', output=BASE/'inputs-v3'):
    source,output=Path(source).resolve(),Path(output).resolve()
    source.relative_to(BASE);output.relative_to(BASE)
    if output.exists():raise ValueError('Use a new candidate version')
    manifest=json.loads((source/'manifest.json').read_text())
    if digest(source/'manifest.json')!=(source/'manifest.sha256').read_text().strip():
        raise ValueError('Candidate source manifest changed')
    refs=dict(manifest['source_sha256'])
    refs[str((source/'manifest.json').relative_to(ROOT))]=digest(source/'manifest.json')
    for name,sha in manifest['files_sha256'].items():refs[str((source/name).relative_to(ROOT))]=sha
    if digest(CONTEXT_REPORT)!=CONTEXT_SHA:raise ValueError('Original context changed')
    context=json.loads(CONTEXT_REPORT.read_text());features=context['features']
    refs[features['path']]=features['sha256']
    for p in (Path(__file__),ROOT/'scripts/research_sector_participation_20261003.py',ROOT/'skills/surge_sector.py'):
        refs[str(p.relative_to(ROOT))]=digest(p)
    original=ROOT/'.cache/all-signals-2019-20261002/inputs/manifest.json'
    before=json.loads(original.read_text());financial=json.loads((BUNDLE/'manifest.json').read_text())
    refs[str(original.relative_to(ROOT))]=digest(original)
    frames={}
    for name in ('close-official','raw-close','raw-volume','eligibility'):
        filename=name+'.parquet';expected=before['files_sha256'][filename]
        if financial['files_sha256'][filename]!=expected:raise ValueError('Peer and account price matrices differ')
        refs[str((BUNDLE/filename).relative_to(ROOT))]=expected
        frames[name]=pd.read_parquet(BUNDLE/filename).set_index('date')
        frames[name].index=pd.to_datetime(frames[name].index)
    verify_refs(refs)
    rows=pd.read_parquet(ROOT/features['path'])
    rows=rows.loc[rows.cohort.eq('legacy_course_breakout') & rows.signal_date.between(START,END)].copy()
    observations=[dict(signal_id=f"peer-{r.signal_date}-{r.stock_id}",stock_id=r.stock_id,signal_date=r.signal_date)
                  for r in rows.itertuples()]
    print('complete all parent peer coordinates',len(observations),flush=True)
    filled,groups=participation_features(*(frames[n] for n in ('close-official','raw-close','raw-volume','eligibility')),observations)
    by_key={(r['stock_id'],r['signal_date']):r for r in filled}
    checked=0;added=Counter()
    for row in rows.itertuples():
        new=by_key[(row.stock_id,row.signal_date)]
        if row.peer_issue=='coordinate_not_covered':
            added[new['feature_issue'] or 'known']+=1
            continue
        old_issue=None if pd.isna(row.peer_issue) else row.peer_issue
        if old_issue!=new['feature_issue']:raise ValueError('Previously covered peer quality changed')
        if old_issue is None and (not np.isclose(row.peer_breadth_value,new['peer_above_ma20_fraction'],rtol=0,atol=1e-12)
                or not np.isclose(row.peer_turnover_multiple,new['peer_share_multiple'],rtol=0,atol=1e-12)):
            raise ValueError('Previously covered peer value changed')
        checked+=1
    calendar=json.loads((source/'calendar.json').read_text());positions={day:i for i,day in enumerate(calendar)}
    parents=json.loads((source/'legacy_course_breakout.json').read_text())
    parent_map={(r['members'][0],r['signal_date']):r for r in parents['entries']+parents['pending_entries']}
    strategy='entry_peer_narrow';entries=[];pending=[];counts=Counter(parent_events=len(rows))
    for row in rows.itertuples():
        peer=by_key[(row.stock_id,row.signal_date)]
        if peer['feature_issue'] is not None or pd.isna(row.market_breadth):
            counts['unknown_context']+=1;continue
        if peer['peer_above_ma20_fraction']<.5 or bool(row.market_breadth):continue
        parent=parent_map[(row.stock_id,row.signal_date)]
        i=positions[row.signal_date];next_day=calendar[i+1] if i+1<len(calendar) else None
        if next_day is not None and next_day>END:next_day=None
        candidate=event(strategy,row.stock_id,row.signal_date,next_day,parent['priority'])
        (entries if next_day else pending).append(candidate)
    order=lambda r:(r['signal_date'],-r['priority'],r['members'][0])
    entries.sort(key=order);pending.sort(key=order);counts.update(executable=len(entries),pending=len(pending))
    output.mkdir(parents=True)
    for name,sha in manifest['files_sha256'].items():
        if name!=strategy+'.json':shutil.copyfile(source/name,output/name)
    write(output/(strategy+'.json'),dict(strategy_id=strategy,entries=entries,pending_entries=pending,
        counts=dict(counts),poc_gate_on_demand=False))
    evidence=output/'peer-completion'
    details=dict(schema='all_parent_peer_completion_v1',previously_covered_exact=checked,
        filled_missing_coordinates=sum(added.values()),added_coordinate_statuses=dict(added),
        total_parent_events=len(rows),candidate_counts=dict(counts),outcomes_read=False,
        definitions_unchanged=True,financial_matrices_identical=True,live_qualified=False)
    for name,value in [('features.json',filled),('monthly-peers.json',groups),('audit.json',details)]:
        path=evidence/name;refs[str(path.relative_to(ROOT))]=write(path,value)
    for name in manifest['files_sha256']:
        manifest['files_sha256'][name]=digest(output/name)
        if name!=strategy+'.json' and manifest['files_sha256'][name]!=digest(source/name):
            raise ValueError('Unrelated strategy candidates changed')
    manifest['counts'][strategy]=dict(counts);manifest['source_sha256']=refs
    manifest['peer_completion']=details
    verify_refs(refs)
    h=write(output/'manifest.json',manifest);(output/'manifest.sha256').write_text(h+'\n')
    print(json.dumps(details,ensure_ascii=False),flush=True)
    return details


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=BASE/'inputs-v2')
    parser.add_argument('--output',type=Path,default=BASE/'inputs-v3')
    args=parser.parse_args();run(args.source,args.output)
