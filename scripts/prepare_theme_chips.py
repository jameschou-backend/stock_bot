#!/usr/bin/env python3
"""Bounded all-market weekly holder snapshots; no production database writes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.file_lock import file_lock
from app.finmind import fetch_dataset
from app.config import load_config
from scripts.research_exit_scenarios import sha, write
from scripts.audit_current_causality_20260925 import matrices, BASE


def load_complete(output, plan):
    """A repeat collection must not rewrite a manifest used by frozen research."""
    path=output/'manifest.json'
    if not path.exists():return None
    saved=json.loads(path.read_text())
    if any(saved.get(k)!=v for k,v in plan.items()) or set(saved['files'])!=set(plan['dates']):
        raise ValueError('Completed snapshot manifest differs from its plan')
    for day,record in saved['files'].items():
        meta=json.loads((output/f'{day}.json').read_text())
        if sha(output/f'{day}.parquet')!=record['sha256'] or any(
                record.get(k)!=meta.get(k) for k in ('date','rows','sha256','retrieved_at','cache_hit')):
            raise ValueError('Completed holder snapshot changed: '+day)
    return dict(saved,requests_this_run=0,reused_complete=True)


def run(output, supplement=None):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'.cache'):
        raise ValueError('Snapshots must stay in the project cache')
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output/'prepare.lock', timeout=0):
        calendar = matrices(BASE)[0].index
        calendar = calendar[calendar >= pd.Timestamp('2021-11-01')]
        days = pd.Series(calendar, index=calendar).groupby(calendar.to_period('W-FRI')).max()
        # A partial current week is not a weekly closing balance.
        days = days[days.index.end_time.normalize() <= calendar[-1]]
        if len(days) > 260:
            raise ValueError('Collection exceeds the preregistered request budget')
        query_days = [str(day.date()) for day in days]
        calendar_source = None
        if supplement:
            parent = Path(supplement).resolve()
            if not parent.is_relative_to(ROOT/'.cache'):
                raise ValueError('Supplement requires local original snapshots')
            source = ROOT/'.cache/holder-case-2492/TaiwanStockHoldingSharesPer.parquet'
            actual = pd.to_datetime(pd.read_parquet(source,columns=['date']).date).drop_duplicates()
            actual = actual[(actual>=calendar[0])&(actual<=calendar[-1])]
            original = json.loads((parent/'plan.json').read_text())['dates']
            query_days = sorted(set(actual.dt.strftime('%Y-%m-%d'))-set(original))
            if len(query_days)+len(original)>280:
                raise ValueError('Combined collection budget exceeded')
            calendar_source = dict(path=str(source.relative_to(ROOT)),sha256=sha(source),
                                   original_plan_sha256=sha(parent/'plan.json'))
        plan = dict(schema='theme_chip_inputs_v1', dataset='TaiwanStockHoldingSharesPer',
                    dates=query_days, max_requests=len(query_days), workers=4,
                    prereg_sha256=sha(ROOT/'docs/prereg_theme_chips_20260927.md'),calendar_source=calendar_source)
        plan_path = output/'plan.json'
        if plan_path.exists():
            original_plan=json.loads(plan_path.read_text())
            if any(original_plan.get(k)!=plan.get(k) for k in (
                    'schema','dataset','dates','max_requests','workers','calendar_source')):
                raise ValueError('Existing request plan differs')
            plan=original_plan  # Preserve the original registration, including amendments' chronology.
        else:
            write(plan_path, plan)
        complete=load_complete(output,plan)
        if complete is not None:return complete
        token = load_config().finmind_token
        def collect(day):
            path = output/f'{day}.parquet'; meta_path = output/f'{day}.json'
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta['date'] != day or sha(path) != meta['sha256']:
                    raise ValueError('Changed holder snapshot: '+day)
                return dict(meta, reused=True)
            if path.exists():
                raise ValueError('Unsealed holder snapshot: '+day)
            frame = fetch_dataset('TaiwanStockHoldingSharesPer', pd.Timestamp(day).date(),
                token=token, requests_per_hour=5400, max_retries=0, timeout=45)
            if not frame.empty:
                if not {'date','stock_id','HoldingSharesLevel','people','percent','unit'}.issubset(frame):
                    raise ValueError('Holder response schema changed')
                if set(pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')) != {day}:
                    raise ValueError('Holder response date differs from requested date')
            temp = path.with_suffix('.tmp'); frame.to_parquet(temp, index=False); temp.replace(path)
            meta = dict(date=day, rows=len(frame), sha256=sha(path),
                        retrieved_at=frame.attrs.get('retrieved_at'), cache_hit=frame.attrs.get('cache_hit',False))
            write(meta_path, meta)
            return dict(meta, reused=False)
        results=[];started=time.perf_counter()
        # Batches bound in-flight work when quota/network errors halt the run.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for first in range(0,len(query_days),4):
                results.extend(list(pool.map(collect,query_days[first:first+4])))
                if len(results)%20==0 or len(results)==len(query_days):
                    print(json.dumps(dict(completed=len(results),planned=len(query_days),
                        empty=sum(r['rows']==0 for r in results),elapsed=round(time.perf_counter()-started,1))),flush=True)
        result=dict(**plan,files={r['date']:r for r in results},
                    elapsed_seconds=round(time.perf_counter()-started,3),
                    requests_this_run=sum(not r['reused'] and not r['cache_hit'] for r in results))
        write(output/'manifest.json',result)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--supplement',type=Path)
    args=p.parse_args();r=run(args.output,args.supplement)
    print(json.dumps({k:r[k] for k in ('max_requests','requests_this_run','elapsed_seconds')}))
