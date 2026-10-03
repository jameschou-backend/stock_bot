#!/usr/bin/env python3
"""Bounded current-request official extension; no DB writes or origin-hold resets."""
from __future__ import annotations
import argparse
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import requests
from app.file_lock import file_lock
from app.twse_client import TWSEError
from skills import official_adj_factors as actions
from skills.market_input_validation import parse_market_day
from skills.official_daily_acquisition import (OfficialDailyAcquisition, create_plan,
    request_item, digest, read, require, _write, _epoch, _security_response)

START, END = '2026-09-10', '2026-10-02'
CACHE = ROOT/'.cache/poc-latest-20261003/official-v1'
CALENDAR = '.cache/all-signals-2019-20261002/inputs/close-official.parquet'
OLD_META = '.cache/market-input-repair-20261002/quote-evidence/official-sources.json'
ACTION_CACHE = '.cache/million-replay-inputs/sources'
SCOPE = dict(purpose='Extend frozen POC four-arm research and HTML through 2026-10-02',
             start=START,end=END,markets=['TWSE','TPEX'],
             datasets=['full_market_daily_tables','ex_rights','capital_reduction','par_value_change'])
COLUMNS = ['source_id','market','date','stock_id','name','open','high','low','close',
           'volume','volume_scope','table_category']


def stamp(clock=None):
    return datetime.fromtimestamp(clock(),timezone.utc).isoformat() if clock else datetime.now(timezone.utc).isoformat()


def sealed(path, value):
    if path.exists():
        require(read(path)==value and path.with_suffix('.sha256').read_text().strip()==digest(path),
                'Immutable output changed: '+str(path))
    else:
        _write(path,value,exclusive=True)
        path.with_suffix('.sha256').write_text(digest(path)+'\n')


def relative(root,path):
    path=Path(path).resolve()
    require(path.is_relative_to(root.resolve()),'Evidence must remain inside repository')
    return str(path.relative_to(root.resolve()))


def dates_from_calendar(frame):
    days=sorted(pd.to_datetime(frame['date']).dt.strftime('%Y-%m-%d').unique())
    days=[d for d in days if START<=d<=END]
    require(days and days[-1]==END,'Calendar does not cover extension end')
    return days


def action_items():
    out=[]
    for kind,_,_ in actions.FETCH_SPECS:
        market=kind.split('_')[0].upper(); fmt='%Y%m%d' if market=='TWSE' else '%Y/%m/%d'
        params=dict(startDate=date.fromisoformat(START).strftime(fmt),
                    endDate=date.fromisoformat(END).strftime(fmt),response='json')
        out.append(dict(kind=kind,market=market,url=getattr(actions,kind.upper()+'_URL'),
            method='GET' if market=='TWSE' else 'POST',params=params,start=START,end=END))
    return out


def initialize(root=ROOT,cache=CACHE):
    root,cache=Path(root).resolve(),Path(cache).resolve(); relative(root,cache)
    if (cache/'inventory.json').exists():
        inventory=read(cache/'inventory.json')
        require(digest(cache/'inventory.json')==(cache/'inventory.sha256').read_text().strip(),'Inventory changed')
        for p,h in inventory['source_sha256'].items(): require(digest(root/p)==h,'Inventory source changed')
        return inventory
    old=read(root/OLD_META)
    require(digest(root/OLD_META)==(root/OLD_META).with_suffix('.sha256').read_text().strip(),'Old source index changed')
    days=dates_from_calendar(pd.read_parquet(root/CALENDAR,columns=['date']))
    refs={p:digest(root/p) for p in (CALENDAR,OLD_META,'scripts/prepare_poc_latest_official.py',
        'skills/official_daily_acquisition.py','skills/market_input_validation.py','skills/official_adj_factors.py')}
    reusable=[]
    for sid,source in old['sources'].items():
        if source.get('date') not in days: continue
        require(source['market'] in ('TWSE','TPEX'),'Unexpected cached market')
        for path,hash_key in ((source['path'],'sha256'),(source['receipt'],'receipt_sha256')):
            require(digest(root/path)==source[hash_key],'Cached source changed'); refs[path]=source[hash_key]
        require(source['url']==request_item(source['market'],source['date'])['url'],'Cached endpoint differs')
        rows=parse_market_day(read(root/source['path']),source['market'],source['date'])
        require(len(rows)==source['rows'],'Cached row count differs')
        reusable.append(dict(source_id=sid,**source))
    found={(r['market'],r['date']) for r in reusable}
    require(len(found)==len(reusable),'Duplicate cached market days')
    missing=[request_item(m,d) for d in days for m in ('TWSE','TPEX') if (m,d) not in found]
    old_actions={}
    for kind,_,_ in actions.FETCH_SPECS:
        paths=sorted((root/ACTION_CACHE).glob('actions-'+kind+'-*.meta.json'))
        require(paths,'Missing frozen action schema example')
        p=max(paths,key=lambda x:read(x)['query_end']); meta=read(p); raw=p.with_name(p.name.replace('.meta.json','.json'))
        require(digest(raw)==meta['sha256'],'Prior corporate action source changed')
        refs[relative(root,p)]=digest(p); refs[relative(root,raw)]=digest(raw)
        old_actions[kind]=dict(path=relative(root,raw),meta_path=relative(root,p),query_end=meta['query_end'],
                              extension_covered=meta['query_start']<=START and meta['query_end']>=END)
    for host in ('www.twse.com.tw','www.tpex.org.tw'):
        p=root/'.cache/official-origin-holds'/(host+'.json')
        if p.exists(): refs[relative(root,p)]=digest(p)
    inventory=dict(schema='poc_latest_official_inventory_v1',start=START,end=END,days=days,
        required_market_days=len(days)*2,reusable=reusable,missing=missing,old_actions=old_actions,
        source_sha256=refs,network_requests=0)
    sealed(cache/'inventory.json',inventory)
    require(missing,'No missing days; acquisition plan unnecessary')
    create_plan(root,cache/'daily',missing,{relative(root,cache/'inventory.json'):digest(cache/'inventory.json'),**refs})
    sealed(cache/'actions-plan.json',dict(schema='poc_latest_actions_plan_v1',entries=action_items(),
         max_requests=6,automatic_retries=0,minimum_start_interval_seconds=3.1))
    return inventory


class CurrentRequestAcquisition(OfficialDailyAcquisition):
    """Keep the original transport/security guards; authorize only this new go."""
    def _blocks(self,state,hold):
        # Python 3.10 fromisoformat rejects Z; normalize a copy, never edit evidence.
        return [dict(b,observed_at=b['observed_at'].replace('Z','+00:00'))
                for b in super()._blocks(state,hold)]

    def _auth(self):
        require(self.authorization is not None,'Current go authorization required')
        record=read(self.authorization); plan=self.cache.parent/'actions-plan.json'
        require(record.get('schema')=='official_daily_authorization_v1'
            and record.get('scope')=='poc_latest_official_extension'
            and record.get('task_scope')==SCOPE and record.get('user_request')=='go'
            and record.get('security_bypass_authorized') is False
            and record.get('plan_sha256')==self.plan_hash
            and record.get('actions_plan_sha256')==digest(plan)
            and digest(plan)==plan.with_suffix('.sha256').read_text().strip(),
            'Authorization does not match current go scope and immutable plans')
        require(0<=self.clock()-_epoch(record['created_at'])<=86400,'Current authorization is stale')
        return dict(path=relative(self.root,self.authorization),sha256=digest(self.authorization),created_at=record['created_at'])


def authorize(root,cache,user_request):
    require(user_request=='go','Record the actual current go, not a prior user quotation')
    p=cache/'authorization.json'
    if not p.exists():
        sealed(p,dict(schema='official_daily_authorization_v1',scope='poc_latest_official_extension',
            user_request='go',task_scope=SCOPE,security_bypass_authorized=False,created_at=stamp(),
            plan_sha256=digest(cache/'daily/plan.json'),actions_plan_sha256=digest(cache/'actions-plan.json')))
    return p


def client(root,cache,**kwargs):
    proofs={}
    for p in sorted((cache/'daily/receipts').glob('*.json')):
        r=read(p)
        if r.get('accepted') and r.get('request_kind')=='single_normal_probe': proofs[r['market']]=p
    return CurrentRequestAcquisition(root,cache/'daily',authorization_path=cache/'authorization.json',
                                      recovery_proofs=proofs,**kwargs)


def parse_actions(root,inventory,item,payload):
    """Require full named schema and no silently skipped four-digit source rows."""
    kind=item['kind']; old=read(root/inventory['old_actions'][kind]['path'])
    stat=str(payload.get('stat','')).lower()
    explicit_empty=any(s in stat for s in ('沒有符合條件','查無資料'))
    if explicit_empty:
        require(not payload.get('data') and not any(t.get('data') for t in payload.get('tables',[])),
                'Empty status conflicts with event rows')
        return actions.events_to_dataframe([])
    require(stat=='ok','Corporate action response lacks successful status')
    tables=[payload] if item['market']=='TWSE' else payload.get('tables')
    old_tables=[old] if item['market']=='TWSE' else old.get('tables')
    require(isinstance(tables,list) and len(tables)==len(old_tables)==1,'Corporate action table missing/ambiguous')
    source_rows=[]
    for t,expected in zip(tables,old_tables):
        require(t.get('fields')==expected.get('fields') and isinstance(t.get('data'),list),'Corporate action fields changed')
        require(all(isinstance(r,list) and len(r)==len(t['fields']) for r in t['data']),'Malformed action row')
        if 'totalCount' in t: require(t['totalCount']==len(t['data']),'Corporate action row count incomplete')
        source_rows+=t['data']
    for k in ('strDate','endDate'):
        if k in payload:
            require(payload[k]==(START if k=='strDate' else END).replace('-',''),'Corporate action response interval differs')
    if 'params' in payload:
        require(all(payload['params'].get(k)==v for k,v in item['params'].items()),'Corporate action echoed params differ')
    parser=dict((k,p) for k,p,_ in actions.FETCH_SPECS)[kind]
    events=parser(payload)
    actions.validate_events_in_range(events,date.fromisoformat(START),date.fromisoformat(END),kind)
    require(len(events)==len(source_rows),'Corporate action parser skipped source rows')
    four=[e for e in events if re.fullmatch(r'\d{4}',e.stock_id)]
    require(all(e.ratio is not None and actions.RATIO_LO<=e.ratio<=actions.RATIO_HI for e in four),
            'Corporate action has invalid adjustment ratio')
    return actions.events_to_dataframe(events)


def action_probe(c,inventory,item):
    """One normal existing-endpoint request, append-only, same shared origin lock."""
    base=c.cache.parent/'actions'/item['kind']; attempt=base/'attempt.json'; receipt=base/'receipt.json'
    with file_lock(base/'acquisition.lock'):
        if receipt.exists():
            require(digest(receipt)==receipt.with_suffix('.sha256').read_text().strip(),'Action receipt changed')
            return read(receipt)
        require(not attempt.exists(),'Interrupted action attempt; automatic retry forbidden')
        require(item in read(c.cache.parent/'actions-plan.json')['entries'],'Action outside plan')
        state_path,lock_path,hold_path=c._origin_paths(item)
        with file_lock(lock_path):
            state=read(state_path) if state_path.exists() else {}; auth=c._auth()
            require(not state.get('in_flight'),'Interrupted origin request must be reviewed')
            blocks=c._blocks(state,hold_path)
            require(all(_epoch(auth['created_at'])>_epoch(b['observed_at']) for b in blocks),'Origin stopped after current go')
            key=sha256((auth['sha256']+'\n'+item['url']).encode()).hexdigest()
            require(key not in state.get('probes',{}),'Action endpoint probe already consumed')
            wait=max(0,c.interval-(c.clock()-state.get('last_start_epoch',float('-inf'))))
            if wait: c.sleep(wait+.001)
            require(c._auth()==auth,'Authorization changed during wait')
            require(all(_epoch(auth['created_at'])>_epoch(b['observed_at']) for b in c._blocks(state,hold_path)),
                    'Origin stopped during action wait')
            started=stamp(c.clock)
            record=dict(schema='poc_latest_action_receipt_v1',**item,started_at=started,
                authorization=auth,actions_plan_sha256=digest(c.cache.parent/'actions-plan.json'),
                request_kind='single_normal_probe',accepted=False,http_status=None,
                automatic_redirects_disabled=True,redirect_statuses=[],automatic_retries=0)
            _write(attempt,record,exclusive=True)
            state.update(last_start_epoch=c.clock(),in_flight=dict(started_at=started,identity=key))
            state.setdefault('probes',{})[key]=dict(started_at=started,identity=key); _write(state_path,state)
            response=None
            try:
                kwargs={'params' if item['method']=='GET' else 'data':item['params']}
                response=c.session.request(item['method'],item['url'],timeout=(10,30),allow_redirects=False,**kwargs)
            except requests.RequestException as exc:
                response=getattr(exc,'response',None)
                record.update(status='transport_error_no_retry',error_type=type(exc).__name__,
                              exception_response_present=response is not None)
            if response is not None:
                raw=base/'raw.json';raw.write_bytes(response.content)
                denied=_security_response(response)
                record.update(http_status=response.status_code,raw_path=relative(c.root,raw),raw_sha256=digest(raw),
                    security_denied=denied,redirect_statuses=[r.status_code for r in getattr(response,'history',[])])
                if denied: record['status']='origin_stopped'
                elif response.status_code!=200:record['status']='http_error_no_retry'
                elif record.get('exception_response_present'):record['status']='exception_response_no_retry'
                else:
                    try:
                        frame=parse_actions(c.root,inventory,item,json.loads(response.content))
                        record.update(status='verified_corporate_action_interval',accepted=True,events=len(frame))
                    except (ValueError,KeyError,TypeError,TWSEError) as exc:
                        record.update(status='schema_error_no_retry',error_type=type(exc).__name__,error=str(exc))
            record['retrieved_at']=stamp(c.clock);sealed(receipt,record)
            if record.get('security_denied'):
                state['stopped']=dict(observed_at=record['retrieved_at'],kind='security_or_rate_limit',
                    receipt_path=relative(c.root,receipt),receipt_sha256=digest(receipt))
            state.pop('in_flight',None);_write(state_path,state)
            return record


def normalize(root,cache):
    inventory=initialize(root,cache); c=client(root,cache)
    refs=dict(inventory['source_sha256']);sources={};parts=[]
    for s in inventory['reusable']:
        sources[s['source_id']]={k:v for k,v in s.items() if k!='source_id'}
    for p in sorted((cache/'daily/receipts').glob('*.json')):
        if not read(p).get('accepted'):continue
        r=c._receipt(p);sid=r['identity']
        sources[sid]=dict(market=r['market'],date=r['date'],rows=r['rows'],path=r['raw_path'],
            sha256=r['raw_sha256'],receipt=r['receipt_path'],receipt_sha256=r['receipt_sha256'],
            volume_scope=r['volume_scope'],http_status=200,retrieved_at=r['retrieved_at'],url=r['url'])
    for sid,s in sources.items():
        rows=parse_market_day(read(root/s['path']),s['market'],s['date'])
        parts.extend(dict(source_id=sid,**r) for r in rows.values())
    observed={(s['market'],s['date']) for s in sources.values()}
    require(len(observed)==len(sources),'Duplicate normalized market dates')
    missing=[dict(market=m,date=d) for d in inventory['days'] for m in ('TWSE','TPEX') if (m,d) not in observed]
    frame=pd.DataFrame(parts,columns=COLUMNS); events=[];action_status=[]
    for item in action_items():
        p=cache/'actions'/item['kind']/'receipt.json'
        status=dict(kind=item['kind'],start=START,end=END,complete=False,status='not_acquired')
        if p.exists():
            r=read(p); require(digest(p)==p.with_suffix('.sha256').read_text().strip(),'Action receipt changed')
            status['status']=r['status']
            if r.get('accepted'):
                require(all(r.get(k)==v for k,v in item.items()) and r.get('http_status')==200
                    and r.get('automatic_redirects_disabled') is True and not r.get('security_denied')
                    and r.get('redirect_statuses')==[] and r['actions_plan_sha256']==digest(cache/'actions-plan.json'),
                    'Action receipt provenance differs')
                require(digest(root/r['raw_path'])==r['raw_sha256'],'Action raw changed')
                f=parse_actions(root,inventory,item,read(root/r['raw_path']))
                require(len(f)==r['events'],'Action count differs');events.append(f)
                status.update(complete=True,events=len(f))
        action_status.append(status)
    combined=pd.concat(events,ignore_index=True) if events else actions.events_to_dataframe([])
    keys=['stock_id','event_date','market','source','event_type']
    require(not combined.duplicated(keys).any(),'Overlapping corporate action records')
    # New outputs are immutable; rerun to a new output directory after adding evidence.
    out=cache/'normalized';out.mkdir(exist_ok=True)
    require(not (cache/'report.json').exists(),'Final report already exists; preserve sealed result')
    frame.to_parquet(out/'official-normalized.parquet',index=False)
    combined.to_parquet(out/'events.parquet',index=False)
    sealed(out/'official-sources.json',dict(schema='poc_latest_official_sources_v1',sources=sources))
    for p in cache.rglob('*'):
        if p.is_file() and 'normalized' not in p.parts and p.suffix not in ('.lock',): refs[relative(root,p)]=digest(p)
    for s in sources.values(): refs[s['path']]=s['sha256'];refs[s['receipt']]=s['receipt_sha256']
    outputs={relative(root,p):digest(p) for p in out.iterdir() if p.is_file()}
    report=dict(schema='poc_latest_official_extension_v1',start=START,end=END,
        required_market_days=inventory['required_market_days'],accepted_market_days=len(observed),
        missing_market_days=missing,daily_tables_extension_complete=not missing,
        normalized_path=relative(root,out/'official-normalized.parquet'),sources_path=relative(root,out/'official-sources.json'),
        events_path=relative(root,out/'events.parquet'),corporate_events_extension_complete=all(s['complete'] for s in action_status),
        corporate_action_coverage=action_status,event_count=len(combined),source_sha256=refs,output_sha256=outputs,
        network_requests=len(list((cache/'daily/attempts').glob('*.json')))+len(list((cache/'actions').glob('*/attempt.json'))),
        finmind_requests=0,database_mutations=0,live_qualified=False,
        limitations=['TWSE daily shares include all daily sessions; TPEx legacy no1430 is ordinary_session.',
            'Corporate events are reference-price adjustments, not full cash/share entitlement accounting.'])
    sealed(cache/'report.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['inventory','plan','probe','fetch','actions','normalize'])
    p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--market',choices=['TWSE','TPEX']);p.add_argument('--user-request')
    a=p.parse_args();cache=a.cache.resolve();inventory=initialize(ROOT,cache)
    if a.mode in ('inventory','plan'):
        print(json.dumps({k:inventory[k] for k in ('required_market_days','days')},ensure_ascii=False));return
    if a.mode=='normalize':
        r=normalize(ROOT,cache);print(json.dumps({k:v for k,v in r.items() if k not in ('source_sha256','output_sha256')},ensure_ascii=False));return
    require(a.market,'Choose one market')
    if a.mode=='probe':authorize(ROOT,cache,a.user_request)
    c=client(ROOT,cache)
    if a.mode=='actions':
        for item in action_items():
            if item['market']!=a.market:continue
            r=action_probe(c,inventory,item);print(json.dumps(dict(kind=item['kind'],status=r['status'],accepted=r['accepted'])),flush=True)
            if not r['accepted']:raise SystemExit(2)
        return
    selected=[i for i in inventory['missing'] if i['market']==a.market]
    for old in (c.cache/'receipts').glob('*.json'):
        r=read(old)
        require(r.get('market')!=a.market or r.get('accepted') is True,'Prior market request failed; no automatic continuation')
    for item in selected:
        if a.mode=='fetch' and ((c.cache/'attempts'/(item['identity']+'.json')).exists() or (c.cache/'receipts'/(item['identity']+'.json')).exists()):continue
        r=c.probe(item,allow_probe=True) if a.mode=='probe' else c.fetch(item)
        print(json.dumps(dict(market=item['market'],date=item['date'],accepted=r['accepted'],status=r['status'],receipt=r.get('receipt_path'))),flush=True)
        if not r['accepted']:raise SystemExit(2)
        if a.mode=='probe':break

if __name__=='__main__':main()
