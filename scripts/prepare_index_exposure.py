#!/usr/bin/env python3
"""Fetch three explicitly registered ETF datasets, with no DB writes or retries."""
from datetime import date, datetime, timezone
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.config import load_config
from app.file_lock import file_lock
from app.finmind import fetch_dataset
from scripts.research_exit_scenarios import write,read,sha

OUT=ROOT/'.cache/index-exposure-20260927/sources-v1'
DATASETS=('TaiwanStockPrice','TaiwanStockPriceAdj','TaiwanStockPriceLimit')
SPEC=ROOT/'docs/prereg_index_exposure_20260927.md'


def run():
    plan=dict(stock_id='00631L',instrument_type='leveraged_ETF',
        stock_id_exception='Officially listed six-character ETF; not an ordinary four-digit stock',
        datasets=list(DATASETS),start='2021-01-01',end='2026-09-09',maximum_attempts=3,
        max_retries=0,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            (SPEC,Path(__file__),ROOT/'app/finmind.py',ROOT/'app/finmind_cache.py',ROOT/'app/rate_limiter.py')})
    if (OUT/'plan.json').exists() and read(OUT/'plan.json')!=plan:
        raise ValueError('ETF acquisition code or preregistration changed; preserve the old directory')
    write(OUT/'plan.json',plan)
    attempts=read(OUT/'attempts.json') if (OUT/'attempts.json').exists() else []
    config=load_config()
    for dataset in DATASETS:
        path=OUT/(dataset+'.json')
        if path.exists():
            if not path.with_suffix('.sha256').exists() or sha(path)!=path.with_suffix('.sha256').read_text().strip():
                raise ValueError('Existing ETF source is incomplete or changed')
            continue
        if len(attempts)>=plan['maximum_attempts']:
            raise ValueError('Acquisition attempt ceiling reached; retained failed attempts require review')
        attempts.append(dict(dataset=dataset,requested_at=datetime.now(timezone.utc).isoformat()))
        write(OUT/'attempts.json',attempts)
        frame=fetch_dataset(dataset,date.fromisoformat(plan['start']),date.fromisoformat(plan['end']),
            data_id=plan['stock_id'],token=config.finmind_token,requests_per_hour=6000,
            max_retries=0,timeout=30)
        if frame.empty or 'stock_id' not in frame or set(frame.stock_id.astype(str))!={'00631L'}:
            raise ValueError('ETF provider returned empty or mismatched identity data')
        rows=json.loads(frame.to_json(orient='records'))
        write(path,dict(dataset=dataset,stock_id=plan['stock_id'],start=plan['start'],end=plan['end'],
            provider='FinMind',data=rows,attrs={k:frame.attrs.get(k) for k in ('retrieved_at','cache_hit','source')}))
        path.with_suffix('.sha256').write_text(sha(path)+'\n')
        print(dataset,len(frame),min(frame.date),max(frame.date),flush=True)
    manifest=dict(schema='index_exposure_acquisition_v1',database_writes=0,
        files_sha256={p.name:sha(p) for p in OUT.glob('*.json') if p.name!='manifest.json'})
    if (OUT/'manifest.json').exists() and read(OUT/'manifest.json')!=manifest:
        raise ValueError('Acquisition manifest changed')
    write(OUT/'manifest.json',manifest)
    (OUT/'manifest.sha256').write_text(sha(OUT/'manifest.json')+'\n')
    return manifest


if __name__=='__main__':
    with file_lock(OUT/'.acquire.lock',timeout=0):run()
