#!/usr/bin/env python3
"""Refresh each quarantined tape once through the shared FinMind adapter.

Never overwrite a sealed tape, synthesize missing trades, or loosen comparison.
Acquisition is explicit and bounded to the 42 identities in the parent report.
"""
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from skills.board_tape_reconciliation import digest, summarize_ticks, reconcile, verify_report

PARENT = ROOT/'artifacts/forward_simulation/board_tape_reconciliation_20260925.json'
CACHE = ROOT/'.cache/board-conflict-resolution-20260925'


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n'


def sequence(frame):
    return [(str(row.date),str(row.stock_id),str(row.Time),
        str(Decimal(str(row.deal_price)).normalize()),int(row.volume),str(row.TickType))
        for row in frame.itertuples(index=False)]


def compare(old, new, identity, official):
    before_order, after_order = sequence(old), sequence(new)
    before, after = Counter(before_order), Counter(after_order)
    summary = summarize_ticks(new, identity['stock_id'], identity['date'], identity['market'])
    return dict(added_records=sum((after-before).values()), removed_records=sum((before-after).values()),
        payload_unchanged=before_order==after_order, same_record_multiset=before==after,
        new_rows=len(new),
        result=reconcile(summary,official))


def prepare(parent, cache=CACHE):
    from app.file_lock import file_lock
    cache=Path(cache)
    cache.mkdir(parents=True,exist_ok=True)
    with file_lock(cache/'.acquisition.lock',timeout=0):
        _prepare(parent,cache)


def _prepare(parent,cache):
    from app.config import load_config
    from app.finmind import fetch_dataset
    cache=Path(cache)
    cache.mkdir(parents=True,exist_ok=True)
    rows=[r for r in parent['rows'] if r['status']=='daily_aggregate_conflict']
    if len(rows)!=42:
        raise ValueError('This acquisition plan is restricted to the original 42 conflicts')
    ledger=cache/'request-ledger.json'
    state=json.loads(ledger.read_text()) if ledger.exists() else dict(maximum=42,attempts={})
    expected={f"{r['stock_id']}-{r['date']}":dict(dataset='TaiwanStockPriceTick',
        data_id=r['stock_id'],start_date=r['date']) for r in rows}
    if (state.get('maximum')!=42 or not isinstance(state.get('attempts'),dict)
            or not set(state['attempts']).issubset(expected)):
        raise ValueError('Acquisition ledger differs from the fixed conflict plan')
    for key,item in state['attempts'].items():
        if item.get('query')!=expected[key] or item.get('status') not in ('started','received'):
            raise ValueError('Acquisition identity/status differs')
        if item['status']=='received':
            source=(ROOT/item['path']).resolve()
            if not source.is_relative_to(ROOT) or digest(source)!=item['sha256']:
                raise ValueError('Previously acquired source changed')
    config=load_config()
    for row in rows:
        sid,day=row['stock_id'],row['date']; key=f'{sid}-{day}'
        if key in state['attempts']:
            continue
        query=dict(dataset='TaiwanStockPriceTick',data_id=sid,start_date=day)
        state['attempts'][key]=dict(query=query,started_at=datetime.now(timezone.utc).isoformat(),status='started')
        ledger.write_text(encoded(state))
        # max_retries=0 means the 42-row plan permits at most 42 adapter attempts.
        frame=fetch_dataset('TaiwanStockPriceTick',date.fromisoformat(day),data_id=sid,
            token=config.finmind_token,max_retries=0,force_refresh=True,timeout=40)
        raw=cache/(key+'.json')
        with raw.open('x') as stream:
            stream.write(encoded(dict(query=query,retrieved_at=frame.attrs['retrieved_at'],
                source='finmind',data=frame.to_dict('records'))))
        state['attempts'][key].update(status='received',path=str(raw.relative_to(ROOT)),
            sha256=digest(raw),rows=len(frame))
        ledger.write_text(encoded(state))
        print(json.dumps(dict(stock_id=sid,date=day,rows=len(frame),attempts=len(state['attempts']))),flush=True)


def build(cache=CACHE):
    parent=verify_report(PARENT,ROOT)
    cache=Path(cache)
    state=json.loads((cache/'request-ledger.json').read_text())
    refs={str(PARENT.relative_to(ROOT)):digest(PARENT),
          str((cache/'request-ledger.json').relative_to(ROOT)):digest(cache/'request-ledger.json'),
          **parent['input_sha256'],**parent['code_sha256']}
    rows=[]
    for original in parent['rows']:
        if original['status']!='daily_aggregate_conflict': continue
        key=f"{original['stock_id']}-{original['date']}"
        item=state['attempts'].get(key,{})
        if item.get('status')!='received':
            rows.append(dict(stock_id=original['stock_id'],date=original['date'],status='refresh_missing'))
            continue
        path=(ROOT/item['path']).resolve()
        if not path.is_relative_to(ROOT) or digest(path)!=item['sha256']:
            raise ValueError('Refreshed source hash mismatch')
        raw=json.loads(path.read_text())
        if raw['query']!=dict(dataset='TaiwanStockPriceTick',data_id=original['stock_id'],start_date=original['date']):
            raise ValueError('Refreshed source query identity mismatch')
        refs[item['path']]=item['sha256']
        old=pd.read_parquet(ROOT/original['tape_path'])
        new=pd.DataFrame(raw['data'])
        result=compare(old,new,original,original['official'])
        rows.append(dict(stock_id=original['stock_id'],date=original['date'],market=original['market'],
            old_source=original['tape_path'],new_source=item['path'],
            status='refreshed_aggregate_matched' if result['result']['same_scope_aggregate_matched'] else 'conflict_persists',**result))
    return dict(schema='provider_tape_conflict_resolution_v1',rows=rows,
        summary=dict(required=len(rows),refreshed=sum(r['status']!='refresh_missing' for r in rows),
            aggregate_matched=sum(r['status']=='refreshed_aggregate_matched' for r in rows),
            conflicts=sum(r['status']=='conflict_persists' for r in rows),
            unchanged=sum(r.get('payload_unchanged',False) for r in rows)),
        adapter_attempts=len(state['attempts']),input_sha256=refs,
        code_sha256={str(Path(__file__).relative_to(ROOT)):digest(__file__)},
        strict_data_ready=False,live_qualified=False,own_order_fill_proven=False,
        limitations=['A repeated provider response does not replace independent exchange evidence.',
            'Daily total differences cannot locate or synthesize missing intraday records.',
            'The sealed inputs and historical returns are unchanged.'])


def verify(path):
    path=Path(path)
    if digest(path)!=path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Conflict report hash mismatch')
    report=json.loads(path.read_text())
    if report!=build(): raise ValueError('Conflict evidence reproduction differs')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--output',type=Path,default=CACHE/'report.json')
    args=parser.parse_args()
    if args.fetch and args.verify: parser.error('Do not mix acquisition and offline verification')
    if args.fetch: prepare(verify_report(PARENT,ROOT))
    if args.verify: value=verify(args.output)
    else:
        value=build()
        with args.output.open('x') as stream: stream.write(encoded(value))
        args.output.with_suffix('.sha256').write_text(digest(args.output)+'\n')
    print(json.dumps(value['summary'],ensure_ascii=False))
