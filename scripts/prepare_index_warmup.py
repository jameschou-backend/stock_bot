#!/usr/bin/env python3
"""Two bounded incremental requests for a 250-session signal warmup."""
from datetime import date, datetime, timezone
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.config import load_config
from app.file_lock import file_lock
from app.finmind import fetch_dataset
from scripts.research_exit_scenarios import read,write,sha

OUT=ROOT/'.cache/index-exposure-20260927/warmup-v1'
DATASETS=('TaiwanStockPrice','TaiwanStockPriceAdj')

def run():
    plan=dict(stock_id='0050',start='2020-11-01',end='2021-02-01',maximum_attempts=2,
        datasets=list(DATASETS),max_retries=0,purpose='warmup_prefix_and_overlap_only',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),
            ROOT/'docs/prereg_index_exposure_20260927.md',ROOT/'app/finmind.py',
            ROOT/'app/finmind_cache.py',ROOT/'app/rate_limiter.py')})
    if (OUT/'plan.json').exists() and read(OUT/'plan.json')!=plan:
        raise ValueError('Warmup acquisition identity changed')
    write(OUT/'plan.json',plan)
    attempts=read(OUT/'attempts.json') if (OUT/'attempts.json').exists() else []
    config=load_config()
    for dataset in DATASETS:
        path=OUT/(dataset+'.json')
        if path.exists():
            if not path.with_suffix('.sha256').exists() or sha(path)!=path.with_suffix('.sha256').read_text().strip():
                raise ValueError('Warmup source changed')
            continue
        if len(attempts)>=2:raise ValueError('Warmup acquisition attempt ceiling reached')
        attempts.append(dict(dataset=dataset,requested_at=datetime.now(timezone.utc).isoformat()))
        write(OUT/'attempts.json',attempts)
        frame=fetch_dataset(dataset,date(2020,11,1),date(2021,2,1),data_id='0050',
            token=config.finmind_token,requests_per_hour=6000,max_retries=0,timeout=30)
        if frame.empty or 'stock_id' not in frame or set(frame.stock_id.astype(str))!={'0050'}:
            raise ValueError('Warmup identity mismatch or no data')
        write(path,dict(dataset=dataset,stock_id='0050',data=json.loads(frame.to_json(orient='records')),
            attrs={k:frame.attrs.get(k) for k in ('retrieved_at','cache_hit','source')}))
        path.with_suffix('.sha256').write_text(sha(path)+'\n')
        print(dataset,len(frame),min(frame.date),max(frame.date),flush=True)
    write(OUT/'manifest.json',dict(schema='index_warmup_v1',database_writes=0,
        files_sha256={p.name:sha(p) for p in OUT.glob('*.json') if p.name!='manifest.json'}))
    (OUT/'manifest.sha256').write_text(sha(OUT/'manifest.json')+'\n')

if __name__=='__main__':
    with file_lock(OUT/'.acquire.lock',timeout=0):run()
