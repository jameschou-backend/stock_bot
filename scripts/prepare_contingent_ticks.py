#!/usr/bin/env python3
"""Fill only the published dependency inventory's board gaps, in a new cache."""
from pathlib import Path
import argparse
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.contingent_execution_ui import load_report
from app.file_lock import file_lock
from scripts.research_intraday_limit import TickCache, OUTPUT as OLD
from scripts.research_exit_scenarios import read,write,sha
from skills.replay_market_feeds import ReplayDataUnavailable
from app.finmind import FinMindError

OUTPUT=ROOT/'.cache/contingent-ticks-20260914'


def run(prepare=False):
    report=load_report()
    requests=sorted({(r['date'],e['stock_id'],e['market']) for c in report['cases'].values()
        for r in c['rows'] for e in r['evidence'] if e['channel']=='board'})
    OUTPUT.mkdir(parents=True,exist_ok=True)
    identity=dict(inventory_sha256=sha(ROOT/'artifacts/forward_simulation/contingent_audit_20260914.json'),requests=requests)
    identity=read_json_identity(identity)
    path=OUTPUT/'identity.json'
    if path.exists() and read(path)!=identity:raise ValueError('Source inventory changed')
    write(path,identity)
    cache=TickCache(OUTPUT/'ticks',online=prepare,maximum=36)
    rows=[]
    for day,sid,market in requests:
        path=cache.root/f'{sid}-{day}.parquet';meta=path.with_suffix('.json')
        old=OLD/'ticks'/path.name
        if not path.exists() and old.exists():
            # Verify original bytes against the sealed manifest before copying.
            manifest=read(OLD/'manifest.json')['files_sha256']
            for p in (old,old.with_suffix('.json')):
                if manifest[str(p.relative_to(OLD))]!=sha(p):raise ValueError('Original tick evidence changed')
                shutil.copyfile(p,cache.root/p.name)
        try:
            frame,digest=cache.get(sid,day,market)
            row=dict(date=day,stock_id=sid,market=market,completed=True,rows=len(frame),
                path=str(path.relative_to(ROOT)),sha256=digest,metadata_sha256=sha(meta))
        except (ReplayDataUnavailable,FinMindError) as exc:
            # Leave completed requests durable; stop on service/quota failure.
            row=dict(date=day,stock_id=sid,market=market,completed=False,reason=str(exc))
            rows.append(row);write(OUTPUT/'summary.json',dict(rows=rows,all_completed=False,requests_this_run=cache.calls))
            raise
        rows.append(row)
        print(sid,day,'rows',len(frame),'requests',cache.calls,flush=True)
    result=dict(rows=rows,all_completed=True,requests_this_run=cache.calls,
        requests_lifetime=read(cache.root/'budget.json')['reserved'] if (cache.root/'budget.json').exists() else 0,
        source_inventory_sha256=identity['inventory_sha256'],historical_odd_verified=False,
        live_qualified=False)
    write(OUTPUT/'summary.json',result)
    print('completed',len(rows),'requests_this_run',cache.calls,flush=True)


def read_json_identity(value):
    import json
    return json.loads(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
    with file_lock(OUTPUT/'run.lock',timeout=1):run(args.prepare)
