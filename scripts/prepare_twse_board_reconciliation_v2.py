#!/usr/bin/env python3
"""Bounded, resumable official TWSE preparation; no FinMind or trading operations."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import argparse
import json
import os
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import requests
from app.file_lock import file_lock
from skills.board_tape_reconciliation import digest
from skills.backtest_data_evidence import verify_report
from scripts.audit_board_tape_reconciliation import required_items

CACHE=ROOT/'.cache/twse-board-reconciliation-v2-20260925'
DATA=ROOT/'artifacts/forward_simulation/backtest_data_completion_20260925.json'
KINDS={'total':'afterTrading/MI_INDEX','intraday_odd':'afterTrading/TWTC7U',
       'after_odd':'afterTrading/TWT53U','fixed':'afterTrading/BFT41U',
       'block_single':'block/BFIAUU','block_basket':'block/BFIAUU'}
MAX_REQUESTS=3000
DAY_KIND_ORDER=('total','intraday_odd','after_odd','fixed','block_single','block_basket')
ORIGIN_HOLD_RELATIVE='.cache/official-origin-holds/www.twse.com.tw.json'


def encode(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)
def relative(path):return str(Path(path).resolve().relative_to(ROOT))
def query(kind,day):
    params=dict(date=day.replace('-',''),response='json')
    if kind=='total':params['type']='ALLBUT0999'
    elif kind=='fixed':params['selectType']='ALL'
    elif kind=='block_single':params['selectType']='S'
    elif kind=='block_basket':params['selectType']='M'
    return dict(kind=kind,date=day,url='https://www.twse.com.tw/rwd/zh/'+KINDS[kind],params=params)
def identity(item):return sha256(encode({k:item[k] for k in ('url','params')}).encode()).hexdigest()


def dispatch_order(entries):
    """Complete dates together without modifying the immutable acquisition plan."""
    return sorted(entries,key=lambda item:(item['date'],DAY_KIND_ORDER.index(item['kind'])))


def plan(cache=CACHE):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    path=cache/'plan.json'
    if path.exists():
        result=json.loads(path.read_text())
        if digest(path)!=path.with_suffix('.sha256').read_text().strip():raise ValueError('Preparation plan changed')
        if digest(ROOT/result['data_path'])!=result['data_sha256']:raise ValueError('Required-session input changed')
        return result
    data=verify_report(DATA,ROOT)
    items=[r for r in required_items(data) if r['market']=='TWSE']
    days=sorted({r['date'] for r in items});sources={}
    # Index only timestamped official metadata. Default selectors are normalized
    # only for endpoints with explicit returned selector fields.
    for meta in sorted((ROOT/'.cache').glob('**/*.source.json')):
        try:
            m=json.loads(meta.read_text());url=m.get('url','');parsed=urlparse(url)
            if parsed.hostname!='www.twse.com.tw':continue
            name=parsed.path.removeprefix('/rwd/zh/')
            if name not in KINDS.values():continue
            params={k:v[0] for k,v in parse_qs(parsed.query).items()}
            raw=meta.with_name(meta.name.removesuffix('.source.json'))
            if not raw.exists():raw=meta.with_name(meta.name.removesuffix('.source.json')+'.json')
            if not raw.exists() or m.get('sha256')!=digest(raw):continue
            payload=json.loads(raw.read_text());day=payload.get('date','')
            if len(day)!=8 or day[:4]<'2022':continue
            day=f'{day[:4]}-{day[4:6]}-{day[6:]}'
            if day not in days or str(payload.get('stat')).lower()!='ok':continue
            for kind in KINDS:
                if KINDS[kind]!=name:continue
                expected=query(kind,day)
                if kind in ('fixed','block_single','block_basket'):
                    selector=payload.get('selectType')
                    if selector!=expected['params']['selectType']:continue
                    params.setdefault('selectType',selector)
                if params!=expected['params']:continue
                sources[(kind,day)]=dict(raw_path=relative(raw),raw_sha256=digest(raw),
                    metadata_path=relative(meta),metadata_sha256=digest(meta),record_wrapper=False)
        except (ValueError,TypeError,KeyError,OSError):continue
    for day in days:
        candidates=[ROOT/f'.cache/sector-account-sources-r2-20260925/inputs/execution-feeds/odd-twse-{day}.raw.json']
        if not candidates[0].exists():candidates=sorted((ROOT/'.cache').glob(f'**/odd-twse-{day}.raw.json'))
        for raw in candidates:
            try:
                record=json.loads(raw.read_text());payload=record['payload'];expected=query('intraday_odd',day)
                if (record.get('url')!=expected['url'] or record.get('params')!=expected['params']
                        or record.get('http_status')!=200 or payload.get('date')!=day.replace('-','')
                        or payload.get('type')!='ALL' or str(payload.get('stat')).lower()!='ok'):continue
                sources[('intraday_odd',day)]=dict(raw_path=relative(raw),raw_sha256=digest(raw),record_wrapper=True)
                break
            except (OSError,ValueError,TypeError,KeyError):continue
    entries=[]
    # Discover nonempty baskets early without needing another speculative query.
    for kind in ('block_basket','total','after_odd','fixed','block_single','intraday_odd'):
        for day in days:
            item=query(kind,day);item['identity']=identity(item)
            if (kind,day) in sources:item['reused']=sources[(kind,day)]
            entries.append(item)
    result=dict(schema='twse_board_preparation_plan_v2',data_path=relative(DATA),data_sha256=digest(DATA),
        prepared_at=datetime.now(timezone.utc).isoformat(),max_http_requests=MAX_REQUESTS,min_start_interval_seconds=1.5,
        max_attempts_per_identity=3,required_stock_days=len(items),required_dates=len(days),
        base_missing_requests=sum('reused' not in i for i in entries),entries=entries)
    path.write_text(encode(result)+'\n');path.with_suffix('.sha256').write_text(digest(path)+'\n')
    return result


class OfficialAccessDenied(RuntimeError):
    pass


class Fetcher:
    def __init__(self,cache,max_requests=MAX_REQUESTS,*,allow_security_retry=False,min_interval=3.1):
        self.cache=Path(cache);self.raw=self.cache/'raw';self.raw.mkdir(parents=True,exist_ok=True)
        self.path=self.cache/'requests.jsonl';self.events=[]
        if self.path.exists():self.events=[json.loads(line) for line in self.path.read_text().splitlines()]
        if not 1<=max_requests<=MAX_REQUESTS:raise ValueError('Hard official request cap exceeded')
        self.max_requests=max_requests
        if min_interval<1.5:raise ValueError('Official interval cannot be shorter than 1.5 seconds')
        self.min_interval=min_interval
        self.allow_security_retry=allow_security_retry
        self.session=requests.Session()
        self.session.headers['User-Agent']='stock-bot historical-data reconciliation/2'
    def add(self,row):
        with self.path.open('a') as f:f.write(encode(row)+'\n');f.flush()
        self.events.append(row)
    def attempts(self):return [e for e in self.events if e['event']=='start']
    def hold_origin(self,item,result):
        """Persist rejection evidence before making a cross-cache stop visible."""
        folder=self.cache/'security-denials';folder.mkdir(exist_ok=True)
        receipt=folder/f"{item['identity']}-{result['attempt']}.json"
        document=dict(schema='official_security_denial_receipt_v1',request=item,response=result)
        with receipt.open('x') as stream:
            stream.write(encode(document)+'\n');stream.flush();os.fsync(stream.fileno())
        hold=dict(schema='official_origin_hold_v1',origin='https://www.twse.com.tw',status='blocked',
            reason='Official origin refused the request; automatic cross-cache dispatch is stopped.',
            observed_at=result['retrieved_at'],no_automatic_recovery=True,
            evidence_sha256={relative(receipt):digest(receipt),result['raw_path']:result['raw_sha256']})
        path=ROOT/ORIGIN_HOLD_RELATIVE;path.parent.mkdir(parents=True,exist_ok=True)
        with file_lock(path.with_suffix('.lock'),timeout=5):
            if path.exists():return
            with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as stream:
                stream.write(encode(hold)+'\n');stream.flush();os.fsync(stream.fileno());temporary=stream.name
            os.replace(temporary,path)
    def fetch(self,item):
        parsed=urlparse(item['url'])
        documentation=(item.get('response_format')=='html' and parsed.scheme=='https'
            and parsed.netloc=='www.twse.com.tw' and parsed.path=='/zh/trading/block/bfiauu-detail.html')
        if not documentation and item['url'].split('?')[0].startswith('https://www.twse.com.tw/rwd/zh/') is not True:
            raise ValueError('Only the explicit official HTTPS endpoint is permitted')
        key=item['identity'];previous=[e for e in self.events if e.get('identity')==key]
        for event in previous:
            if event['event']=='finish' and event.get('accepted'):
                if digest(ROOT/event['raw_path'])!=event['raw_sha256']:raise ValueError('Cached official payload changed')
                return event
        for hold_path in (self.cache/'origin-hold.json',ROOT/ORIGIN_HOLD_RELATIVE):
            if not hold_path.exists():continue
            hold=json.loads(hold_path.read_text())
            for path,expected in hold['evidence_sha256'].items():
                source=(ROOT/path).resolve()
                if not source.is_relative_to(ROOT) or digest(source)!=expected:raise ValueError('Origin hold evidence changed')
            raise OfficialAccessDenied('Shared official-origin hold is active; no new request or recovery probe dispatched')
        denied_indices=[i for i,e in enumerate(self.events) if e['event']=='finish' and (e.get('security_denied')
                        or e.get('http_status') in (401,403,307,428))]
        recovery_index=max((i for i,e in enumerate(self.events) if e['event']=='security_recovery'),default=-1)
        security_probe=False
        if denied_indices and denied_indices[-1]>recovery_index:
            denial=self.events[denied_indices[-1]]
            latest_denial=datetime.fromisoformat(denial['retrieved_at']).timestamp()
            if time.time()<latest_denial+300:
                raise OfficialAccessDenied('Official security cooldown has not elapsed; no request dispatched')
            if not self.allow_security_retry or key!=denial['identity']:
                raise OfficialAccessDenied('Origin blocked: only an explicitly authorized same-identity recovery probe is allowed')
            if any(e.get('security_recovery_probe') for e in self.events[recovery_index+1:] if e['event']=='start'):
                raise OfficialAccessDenied('The single recovery probe was already consumed; origin remains blocked')
            security_probe=True
        tries=sum(e['event']=='start' for e in previous)
        while tries<3:
            if len(self.attempts())>=self.max_requests:raise ValueError('Total official request budget exhausted')
            # A non-transient completed response is not automatically retried.
            if previous and previous[-1]['event']=='finish' and not previous[-1].get('retryable'):
                last=previous[-1]
                raw=(ROOT/last['raw_path']).read_bytes() if last.get('raw_path') else b''
                denied=last.get('security_denied') or b'FOR SECURITY REASONS' in raw
                if not (denied and self.allow_security_retry):return last
            latest=max((e['epoch'] for e in self.attempts()),default=0)
            time.sleep(max(0,self.min_interval-(time.time()-latest)))
            tries+=1
            start=dict(event='start',identity=key,attempt=tries,epoch=time.time(),**{k:item[k] for k in ('kind','date','url','params')})
            if security_probe:start['security_recovery_probe']=True
            self.add(start)
            result=dict(event='finish',identity=key,attempt=tries,retrieved_at=datetime.now(timezone.utc).isoformat(),
                accepted=False,retryable=False)
            try:
                response=self.session.get(item['url'],params=item['params'],timeout=(10,30),allow_redirects=False)
                raw=self.raw/f'{key}-{tries}.json';raw.write_bytes(response.content)
                result.update(http_status=response.status_code,raw_path=relative(raw),raw_sha256=digest(raw),bytes=len(response.content),
                    retryable=response.status_code in (408,429,500,502,503,504))
                result['security_denied']=(300<=response.status_code<400 or response.status_code in (401,403,428) or b'FOR SECURITY REASONS' in response.content
                    or '因為安全性考量'.encode() in response.content)
                result['automatic_redirects_disabled']=True
                result['final_url']=getattr(response,'url',item['url'])
                result['redirect_statuses']=[r.status_code for r in getattr(response,'history',[])]
                if result['security_denied']:result['retryable']=False
                if response.status_code==200 and not result['security_denied']:
                    try:
                        if documentation:
                            result['accepted']=(b'<html' in response.content.lower() and b'BFIAUU' in response.content)
                        else:
                            payload=response.json()
                            result['accepted']=(isinstance(payload,dict) and payload.get('date')==item['date'].replace('-','')
                                and str(payload.get('stat')).lower()=='ok')
                        if not result['accepted']:result['semantic_error']='response_date_or_status_mismatch'
                    except ValueError:result['semantic_error']='not_json'
            except requests.RequestException as exc:
                result.update(error_type=type(exc).__name__,retryable=True)
            self.add(result);previous.append(result)
            if result.get('security_denied'):self.hold_origin(item,result)
            print(encode(dict(kind=item['kind'],date=item['date'],attempt=tries,total_http=len(self.attempts()),
                         accepted=result['accepted'],retryable=result['retryable'])),flush=True)
            if security_probe:
                if result['accepted']:
                    self.add(dict(event='security_recovery',identity=key,epoch=time.time(),
                        recovered_at=datetime.now(timezone.utc).isoformat()))
                    return result
                raise OfficialAccessDenied('Single recovery probe failed; origin remains blocked')
            if result.get('security_denied'):
                raise OfficialAccessDenied('Official security page: stop this origin; retain the rejected response and cool down')
            if result['accepted'] or not result['retryable']:return result
            time.sleep(min(60,10*tries))
        if previous[-1]['event']=='start':
            result=dict(event='finish',identity=key,attempt=tries,accepted=False,retryable=False,
                        error_type='interrupted_attempt_limit_exhausted',retrieved_at=datetime.now(timezone.utc).isoformat())
            self.add(result);return result
        return previous[-1]


def run(cache=CACHE,only=None,allow_security_retry=False):
    cache=Path(cache);p=plan(cache)
    with file_lock(cache/'fetch.lock',timeout=1):
        client=Fetcher(cache,p['max_http_requests'],allow_security_retry=allow_security_retry);results=[]
        for item in dispatch_order(p['entries']):
            if only and item['kind'] not in only:continue
            if 'reused' in item:
                r=item['reused']
                if digest(ROOT/r['raw_path'])!=r['raw_sha256']:raise ValueError('Reused source changed')
                results.append(dict(identity=item['identity'],accepted=True,reused=True,**r));continue
            result=client.fetch(item);results.append(result)
        summary=dict(schema='twse_board_preparation_progress_v2',plan_sha256=digest(cache/'plan.json'),
            max_http_requests=p['max_http_requests'],actual_http_requests=len(client.attempts()),
            accepted=sum(r['accepted'] for r in results),blocked=sum(not r['accepted'] for r in results),
            results=results)
        (cache/'progress.json').write_text(encode(summary)+'\n')
        print(encode({k:v for k,v in summary.items() if k!='results'}),flush=True)
        return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,default=CACHE)
    parser.add_argument('--plan-only',action='store_true');parser.add_argument('--retry-security-after-cooldown',action='store_true');parser.add_argument('--only',nargs='*',choices=list(KINDS))
    args=parser.parse_args()
    if args.plan_only:
        p=plan(args.cache);print(encode({k:v for k,v in p.items() if k!='entries'}))
    else:run(args.cache,args.only,args.retry_security_after_cooldown)
