#!/usr/bin/env python3
"""New, independently sealed inputs for the five requested research axes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.config import load_config
from app.finmind import fetch_dataset
from app.file_lock import file_lock
from scripts.research_exit_scenarios import read, write, sha, encoded
from scripts.prepare_million_signals import build_signals

OUTPUT = ROOT / '.cache/five-axis-20260913'
SPEC = ROOT / 'docs/prereg_five_axis_20260913.md'
AUDIT = ROOT / 'artifacts/forward_simulation/price_chronology_20260913.json'
SOURCE = ROOT / '.cache/million-replay-signals'


def source_files():
    meta = read(SOURCE/'manifest.json')
    files = {str((SOURCE/name).relative_to(ROOT)):value for name,value in meta['files_sha256'].items()}
    files.update({k:v for k,v in meta['code_sha256'].items()})
    refs = read(ROOT/'.cache/cash-allocation-inputs/manifest.json')['references']
    files.update({r['path']:r['sha256'] for r in refs.values()})
    files[str(AUDIT.relative_to(ROOT))] = sha(AUDIT)
    for name,digest in files.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Sealed input changed: '+name)
    return files


def rebuild(output=OUTPUT):
    tick = time.monotonic()
    files = source_files()
    target = output/'rebuild'
    fixed = dict(input_sha256=files, spec_sha256=sha(SPEC), code_sha256=sha(__file__))
    if (target/'identity.json').exists() and read(target/'identity.json') != fixed:
        raise ValueError('Rebuild source changed; choose a new output')
    if (target/'manifest.json').exists():
        verify_rebuild(output)
        return read(target/'summary.json')
    write(target/'identity.json', fixed)
    frames = {name:pd.read_parquet(SOURCE/(name+'.parquet')).set_index('date')
              for name in ('raw-close','raw-volume','close-official','close-quality')}
    for frame in frames.values():
        frame.index = pd.to_datetime(frame.index)
    companies = pd.read_parquet(SOURCE/'companies.parquet')
    original = read(SOURCE/'signals.json')
    rebuilt = build_signals(frames['close-official'], frames['close-quality'], frames['raw-close'],
                              frames['raw-volume'], companies)
    if encoded(rebuilt['entries']) != encoded(original['entries']):
        raise ValueError('Original signal generation cannot reproduce sealed entries')
    changes = []
    for row in read(AUDIT)['quarantine']:
        sid, day = row['stock_id'], pd.Timestamp(row['date'])
        if sid not in frames['raw-close'] or day not in frames['raw-close'].index:
            continue
        before = {k:frame.at[day,sid] for k,frame in frames.items()}
        for frame in frames.values():
            frame.at[day,sid] = float('nan')
        changes.append(dict(stock_id=sid,date=row['date'], previous={k:float(v) if pd.notna(v) else None for k,v in before.items()}))
    clean = build_signals(frames['close-official'], frames['close-quality'], frames['raw-close'],
                          frames['raw-volume'], companies)
    for name,frame in frames.items():
        frame.rename_axis('date').reset_index().to_parquet(target/(name+'.parquet'),index=False)
    write(target/'signals.json', clean)
    old = {r['event_id']:r for r in original['entries']}
    new = {r['event_id']:r for r in clean['entries']}
    groups_old = {g['month']:g for g in rebuilt['diffusion']['groups']}
    groups_new = {g['month']:g for g in clean['diffusion']['groups']}
    changed_groups = [m for m in groups_old if encoded(groups_old[m]) != encoded(groups_new.get(m))]
    result = dict(original_entries=len(old), rebuilt_entries=len(new),
        added=sorted(new.keys()-old.keys()), removed=sorted(old.keys()-new.keys()),
        changed_entries=[sid for sid in old.keys() & new.keys() if encoded(old[sid])!=encoded(new[sid])],
        changed_group_months=changed_groups, group_months=len(groups_new), quarantined_matrix_rows=changes,
        companies=len(companies), full_historical_membership_verified=False,
        original_generation_reproduced=True, elapsed_seconds=round(time.monotonic()-tick,3),
        limitation='Full monthly generation on the original cohort; missing delisted members and historical listing intervals remain unresolved')
    write(target/'summary.json',result)
    if source_files()!=files:
        raise ValueError('Rebuild input changed during computation')
    write(target/'manifest.json',dict(files_sha256={p.name:sha(p) for p in target.iterdir()
        if p.is_file() and p.name!='manifest.json'}))
    return result


def verify_rebuild(output=OUTPUT):
    folder = output/'rebuild'
    meta = read(folder/'manifest.json')
    for name,value in meta['files_sha256'].items():
        if sha(folder/name)!=value:
            raise ValueError('Rebuilt snapshot changed: '+name)
    return meta


def financial(output=OUTPUT):
    verify_rebuild(output)
    entries = read(output/'rebuild/signals.json')['entries']
    ids = sorted({r['members'][0] for r in entries})
    folder = output/'financial';folder.mkdir(parents=True,exist_ok=True)
    plan = dict(dataset='TaiwanStockFinancialStatements',ids=ids,start='2020-01-01',end='2026-09-09')
    if (folder/'plan.json').exists() and read(folder/'plan.json')!=plan:
        raise ValueError('Financial sample changed')
    write(folder/'plan.json',plan)
    # Include the already authorized 2330 schema probe in the lifetime budget.
    ledger = read(folder/'attempts.json') if (folder/'attempts.json').exists() else {'calls':['2330-schema-probe']}
    todo = [sid for sid in ids if not (folder/(sid+'.parquet')).exists()]
    token = load_config().finmind_token
    def one(sid):
        frame = fetch_dataset(plan['dataset'],date(2020,1,1),date(2026,9,9),data_id=sid,
                              token=token,max_retries=0,timeout=30)
        if not frame.empty:
            if not {'date','stock_id','type','value'}.issubset(frame) or set(frame.stock_id.astype(str))!={sid}:
                raise ValueError('Unexpected financial source identity: '+sid)
            dates=pd.to_datetime(frame.date)
            if dates.isna().any() or not dates.between('2020-01-01','2026-09-09').all():
                raise ValueError('Wrong financial source dates: '+sid)
        frame.to_parquet(folder/(sid+'.parquet'),index=False)
        return len(frame)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0,len(todo),4):
            batch=todo[i:i+4]
            if len(ledger['calls'])+len(batch)>400:
                raise ValueError('Financial request budget exhausted')
            ledger['calls'].extend(batch);write(folder/'attempts.json',ledger)
            futures=[(sid,pool.submit(one,sid)) for sid in batch]
            errors=[]
            for sid,future in futures:
                try: print('financial',sid,future.result(),flush=True)
                except Exception as exc: errors.append(exc)
            if errors: raise errors[0]
    write(folder/'manifest.json',dict(plan=plan, calls_reserved=len(ledger['calls']),
        fetched_at=datetime.now(timezone.utc).isoformat(), publication_timestamps_verified=False,
        files_sha256={p.name:sha(p) for p in folder.iterdir() if p.suffix=='.parquet'}))
    return dict(companies=len(ids),calls=len(ledger['calls']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--financial',action='store_true')
    args=parser.parse_args()
    with file_lock(args.output/'prepare.lock',timeout=0):
        print(financial(args.output) if args.financial else rebuild(args.output))
