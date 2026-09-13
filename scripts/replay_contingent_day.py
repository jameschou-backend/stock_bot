#!/usr/bin/env python3
"""Run or exactly verify one explicit plan with supplied, identified market tapes."""
from decimal import Decimal
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.research_exit_scenarios import read,write,sha,encoded
from skills.contingent_replay import Tape,replay_day
from skills.intraday_limit_replay import normalize_ticks
from skills.replay_market_feeds import ReplayDataUnavailable

CODE=['skills/contingent_execution.py','skills/contingent_replay.py',
      'scripts/replay_contingent_day.py','skills/intraday_limit_replay.py',
      'docs/prereg_contingent_replay_20260914.md']


def load_tape(item,base):
    path=(base/item['path']).resolve()
    if not path.exists():raise ReplayDataUnavailable('Required source file missing: '+str(path))
    digest=sha(path)
    if digest!=item['sha256']:raise ValueError('Source file hash mismatch')
    if item['format']=='finmind_board':
        meta_path=path.with_suffix('.json')
        if not meta_path.exists():raise ReplayDataUnavailable('Board metadata missing')
        if sha(meta_path)!=item['metadata_sha256']:raise ValueError('Board metadata hash mismatch')
        meta=read(meta_path)
        expected=dict(dataset='TaiwanStockPriceTick',data_id=item['stock_id'],start_date=item['date'])
        if meta['raw_sha256']!=digest or meta['query']!=expected:
            raise ValueError('Board query identity mismatch')
        frame=normalize_ticks(pd.read_parquet(path),item['stock_id'],item['date'],item['market'])
        rows=[]
        for row in frame.itertuples(index=False):
            price=Decimal(str(row.price))*100
            if price!=price.to_integral_value():raise ValueError('Unrepresentable price cents')
            rows.append((int(row.time.value//1000),int(price),int(row.shares),True))
        tape=Tape(item['stock_id'],'board',item['market'],item['date'],tuple(rows),
                  'https://api.finmindtrade.com/api/v4/data',digest,False)
    elif item['format']=='normalized_auction_v1':
        data=read(path)
        if (data['schema']!='normalized_auction_v1' or data['timezone']!='Asia/Taipei'
                or data['quantity_unit']!='shares' or data['price_unit']!='TWD_cents'
                or data['channel'] not in ('board','odd')):
            raise ValueError('Explicit market, volume and price units required')
        if data['session_complete'] is not True:
            raise ReplayDataUnavailable('Source does not declare a complete session')
        if not data['rows'] and data.get('no_trades_confirmed') is not True:
            raise ReplayDataUnavailable('Empty source without confirmed no-trade session')
        for key in ('stock_id','date','market'):
            if data[key]!=item[key]:raise ValueError('Auction source query identity mismatch')
        if any(r['record_type'] not in ('trade','trial') for r in data['rows']):
            raise ValueError('Unknown auction record type')
        rows=tuple((r['time_us'],r['price_cents'],r['shares'],r['record_type']=='trade') for r in data['rows'])
        tape=Tape(data['stock_id'],data['channel'],data['market'],data['date'],rows,
                  data['source_url'],digest,data['synthetic'])
    else:raise ValueError('Unsupported tape format; no silent fallback')
    tape.validate(item['date'])
    return tape


def run(plan_path,output,verify=False):
    plan_path=Path(plan_path).resolve();output=Path(output).resolve()
    document=read(plan_path);tapes={};missing=[]
    for item in document['sources']:
        try:tape=load_tape(item,plan_path.parent)
        except ReplayDataUnavailable as exc:
            missing.append(str(exc));continue
        key=(tape.stock_id,tape.channel)
        if key in tapes:raise ValueError('Duplicate market tape')
        tapes[key]=tape
    result=replay_day(document['spec'],tapes)
    if missing:
        result=dict(date=document['spec']['date'],scope='single_day_contingent_execution',
            completed=False,total_return=None,live_qualified=False,missing=missing,events=[],fills=[])
    result.update(plan_sha256=sha(plan_path),code_sha256={p:sha(ROOT/p) for p in CODE},
                  independent_source_audit_completed=False,network_calls=0)
    if verify:
        if encoded(read(output))!=encoded(result):raise ValueError('Offline single-day replay differs')
    else:
        if output.exists() and read(output)!=result:raise ValueError('Output is immutable; choose a new path')
        write(output,result)
    print(dict(completed=result['completed'],fills=len(result['fills']),verified=verify,network_calls=0))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args();run(a.plan,a.output,a.verify)
