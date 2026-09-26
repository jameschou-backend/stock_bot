#!/usr/bin/env python3
"""Acquire six bounded sources for an unchanged earlier-period ETF replication."""
from datetime import date,datetime,timezone
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.config import load_config
from app.file_lock import file_lock
from app.finmind import fetch_dataset
from scripts.research_exit_scenarios import read,write,sha

OUT=ROOT/'.cache/index-earlier-20260927/sources-v1'
REQUESTS=(('00631L','TaiwanStockPrice'),('00631L','TaiwanStockPriceLimit'),
    ('0050','TaiwanStockPrice'),('0050','TaiwanStockPriceAdj'),('0050','TaiwanStockPriceLimit'),('0050','TaiwanStockDividend'))

def run():
    plan=dict(start='2014-11-01',end='2021-12-31',requests=[list(r) for r in REQUESTS],
        maximum_attempts=6,max_retries=0,database_writes=0,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),
            ROOT/'docs/prereg_index_earlier_period_20260927.md',ROOT/'skills/index_exposure_replay.py',
            ROOT/'artifacts/forward_simulation/index_exposure_20260927.json',
            ROOT/'app/finmind.py',ROOT/'app/finmind_cache.py',ROOT/'app/rate_limiter.py')})
    if (OUT/'plan.json').exists() and read(OUT/'plan.json')!=plan:
        raise ValueError('Earlier study acquisition identity changed')
    write(OUT/'plan.json',plan)
    attempts=read(OUT/'attempts.json') if (OUT/'attempts.json').exists() else []
    config=load_config()
    for sid,dataset in REQUESTS:
        path=OUT/(sid+'-'+dataset+'.json')
        if path.exists():
            if not path.with_suffix('.sha256').exists() or sha(path)!=path.with_suffix('.sha256').read_text().strip():
                raise ValueError('Earlier source changed')
            continue
        if len(attempts)>=6:raise ValueError('Earlier study acquisition ceiling reached')
        attempts.append(dict(stock_id=sid,dataset=dataset,requested_at=datetime.now(timezone.utc).isoformat()))
        write(OUT/'attempts.json',attempts)
        frame=fetch_dataset(dataset,date(2014,11,1),date(2021,12,31),data_id=sid,
            token=config.finmind_token,requests_per_hour=6000,max_retries=0,timeout=30)
        if frame.empty or 'stock_id' not in frame or set(frame.stock_id.astype(str))!={sid}:
            raise ValueError('Earlier source identity mismatch or empty')
        write(path,dict(dataset=dataset,stock_id=sid,data=json.loads(frame.to_json(orient='records')),
            attrs={k:frame.attrs.get(k) for k in ('retrieved_at','cache_hit','source')}))
        path.with_suffix('.sha256').write_text(sha(path)+'\n')
        print(sid,dataset,len(frame),min(frame.date),max(frame.date),flush=True)
    write(OUT/'manifest.json',dict(schema='index_earlier_acquisition_v1',database_writes=0,
        files_sha256={p.name:sha(p) for p in OUT.glob('*.json') if p.name!='manifest.json'}))
    (OUT/'manifest.sha256').write_text(sha(OUT/'manifest.json')+'\n')

if __name__=='__main__':
    with file_lock(OUT/'.acquire.lock',timeout=0):run()
