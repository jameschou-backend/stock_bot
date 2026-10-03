#!/usr/bin/env python3
"""One reviewed normal probe per odd-lot endpoint, then bounded completion."""
from datetime import date, datetime, timezone
from pathlib import Path
from hashlib import sha256
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import requests
from app.file_lock import file_lock
from skills.official_daily_acquisition import (
    OfficialDailyAcquisition, AcquisitionBlocked, _write, _bound, _epoch as _existing_epoch,
    _security_response, digest, read, require,
)
from skills.replay_market_feeds import parse_odd, URLS, ReplayDataUnavailable

FIRST = {'TWSE': ('2025-02-13','6558'), 'TPEX': ('2025-06-23','5475')}
PREREG = 'docs/prereg_volume_profile_odd_completion_20261003.md'
PREREG_SHA = 'dedf8fcfe0671d6a52b5ba90b04bdd3c6e79835128e8fb9ae5bf3331f90830cf'
BASE = '.cache/volume-profile-account-20261003/odd-completion-v1'
DEST = '.cache/market-input-repair-20261002/execution-v1'


def _epoch(stamp):
    """Accept standard UTC Z on Python 3.10 without rewriting stored evidence."""
    if isinstance(stamp,str) and stamp.endswith('Z'):
        stamp=stamp[:-1]+'+00:00'
    return _existing_epoch(stamp)


def item(market,day):
    market=market.upper()
    require(market in FIRST,'Only the two preregistered odd-lot endpoints are allowed')
    d=date.fromisoformat(day)
    require(day==d.isoformat() and '2024-01-02'<=day<='2026-09-09','Date outside fixed account period')
    return dict(market=market,date=day,url=URLS[market.lower()],
                params=dict(date=d.strftime('%Y%m%d' if market=='TWSE' else '%Y/%m/%d'),response='json'))


class OddCompletion:
    def __init__(self,root=ROOT,*,session=None,clock=time.time,sleep=time.sleep):
        self.root=Path(root).resolve();self.cache=self.root/BASE
        self.clock,self.sleep=clock,sleep;self.refs={}
        self.session=session or requests.Session()
        if session is None:
            # No environment proxy, cookies, supplied credentials or custom UA.
            self.session.trust_env=False
        require(digest(self.root/PREREG)==PREREG_SHA,'Odd completion preregistration changed')

    def _auth(self,create=False):
        path=self.cache/'authorization.json'
        if create and not path.exists():
            value=dict(schema='vp_odd_authorization_v1',urls=URLS,maximum_attempts=50,
                normal_probes={k:list(v) for k,v in FIRST.items()},minimum_interval_seconds=3.1,retries=0,
                scope='user-authorized Volume Profile account data completion; parent explicitly reviewed one normal endpoint probe',
                global_hold_remains=True,security_bypass_authorized=False,
                created_at=datetime.fromtimestamp(self.clock(),timezone.utc).isoformat(),
                prereg_path=PREREG,prereg_sha256=PREREG_SHA)
            _write(path,value,exclusive=True)
        require(path.exists(),'Explicit --probe is required before endpoint completion')
        value=read(path)
        require(value.get('schema')=='vp_odd_authorization_v1' and value.get('urls')==URLS
                and value.get('maximum_attempts')==50 and value.get('normal_probes')=={k:list(v) for k,v in FIRST.items()}
                and value.get('prereg_sha256')==PREREG_SHA
                and value.get('minimum_interval_seconds')==3.1 and value.get('retries')==0
                and value.get('global_hold_remains') is True
                and value.get('security_bypass_authorized') is False,
                'Endpoint authorization changed')
        require(0<=self.clock()-_epoch(value['created_at'])<=86400,'Scoped authorization expired')
        return dict(path=str(path.relative_to(self.root)),sha256=digest(path),created_at=value['created_at'])

    def _requirement(self,market,day,stock_id,evidence):
        require(isinstance(stock_id,str) and len(stock_id)==4 and stock_id.isdigit(),'Four-digit required stock')
        path=_bound(self.root,evidence);value=read(path)
        require(value.get('completed') is False and value.get('summary') is None
                and value.get('reason')=='Offline official odd cache missing (holds retained): odd:'+market.lower()+':'+day,
                'A retained failed account must establish this exact odd-lot demand')
        plans=value.get('partial_journal',{}).get('day_plans',[])
        matches=[p for p in plans if p.get('date')==day and p.get('stock_id')==stock_id
                 and p.get('planned_qty',0)>0 and p.get('odd_qty',0)>0]
        require(matches,'Required stock has no precommitted positive odd-share order')
        return dict(path=str(path.relative_to(self.root)),sha256=digest(path),stock_id=stock_id,plans=matches)

    def _receipt(self,path):
        value,record,rows,refs=self._receipt_closure(path,set())
        for name,expected in refs.items():
            require(name not in self.refs or self.refs[name]==expected,'Previously bound odd evidence changed')
            self.refs[name]=expected
        return value,record,rows

    def _receipt_closure(self,path,seen):
        """Verify immutable receipt ancestors without authorizing any network I/O."""
        path=_bound(self.root,path)
        require(path not in seen,'Cyclic odd recovery evidence')
        seen=seen|{path};refs={}
        def bind(p,expected=None):
            p=_bound(self.root,p);actual=digest(p)
            require(expected is None or actual==expected,'Odd evidence ancestor changed: '+str(p))
            refs[str(p.relative_to(self.root))]=actual
            return p
        bind(path);bind(path.with_suffix('.sha256'))
        require(path.with_suffix('.sha256').read_text().strip()==digest(path),'Receipt changed')
        value=read(path);request=item(value['market'],value['date'])
        require(path==self.cache/'receipts'/(value['market']+'-'+value['date']+'.json'),
                'Receipt is outside its exact endpoint/date identity')
        require(value.get('schema')=='vp_odd_receipt_v1' and value.get('accepted') is True
                and all(value.get(k)==v for k,v in request.items())
                and value.get('http_status')==200 and value.get('security_denied') is False
                and value.get('redirect_statuses')==[] and value.get('automatic_redirects_disabled') is True
                and value.get('global_hold_unchanged') is True,
                'A complete successful exact-endpoint receipt is required')
        raw=bind(value['raw_path'],value['raw_sha256'])
        require(raw==self.cache/'raw'/(value['market']+'-'+value['date']+'.bin'),
                'Raw response identity differs')
        attempt_path=bind(self.cache/'attempts'/(value['market']+'-'+value['date']+'.json'))
        attempt=read(attempt_path)
        require(attempt.get('schema')=='vp_odd_attempt_v1'
                and all(value.get(k)==v for k,v in attempt.items() if k!='schema'),
                'Receipt differs from its pre-I/O attempt')
        auth=value['authorization'];auth_path=bind(auth['path'],auth['sha256']);authorization=read(auth_path)
        require(auth_path==self.cache/'authorization.json'
                and authorization.get('schema')=='vp_odd_authorization_v1'
                and authorization.get('urls')==URLS and authorization.get('maximum_attempts')==50
                and authorization.get('normal_probes')=={k:list(v) for k,v in FIRST.items()}
                and authorization.get('minimum_interval_seconds')==3.1 and authorization.get('retries')==0
                and authorization.get('global_hold_remains') is True
                and authorization.get('security_bypass_authorized') is False
                and authorization.get('created_at')==auth['created_at']
                and authorization.get('prereg_path')==PREREG
                and authorization.get('prereg_sha256')==PREREG_SHA,
                'Receipt authorization scope differs')
        bind(self.root/PREREG,PREREG_SHA)
        start=_epoch(value['started_at']);end=_epoch(value['retrieved_at'])
        require(0<=start-_epoch(auth['created_at'])<=86400 and end>=start,
                'Receipt was outside its authorization time')
        bind(self.cache/'source-snapshots'/(value['helper_sha256']+'.py'),value['helper_sha256'])
        demand=value['requirement'];bind(demand['path'],demand['sha256'])
        require(self._requirement(value['market'],value['date'],demand['stock_id'],demand['path'])==demand,
                'Receipt differs from its precommitted positive odd-share requirement')
        _,_,hold=OfficialDailyAcquisition._origin_paths(self,request)
        if value.get('global_hold_sha256') is not None:
            bind(hold,value['global_hold_sha256']);hold_data=read(hold)
            require(hold_data.get('status')=='blocked','Original global hold changed')
            for name,expected in hold_data.get('evidence_sha256',{}).items():bind(name,expected)
        proof=value.get('recovery_proof')
        if value.get('request_kind')=='single_normal_probe':
            require((value['date'],demand['stock_id'])==FIRST[value['market']] and proof is None,
                    'Receipt was not the reviewed first endpoint probe')
        else:
            require(value.get('request_kind')=='necessary_missing_day' and isinstance(proof,dict),
                    'Followup lacks endpoint recovery proof')
            proof_path=bind(proof['path'],proof['sha256'])
            previous,_,_,parents=self._receipt_closure(proof_path,seen)
            require(previous['request_kind']=='single_normal_probe'
                    and previous['market']==value['market'] and previous['url']==value['url']
                    and previous['authorization']==auth and previous['retrieved_at']==proof['retrieved_at']
                    and 0<=start-_epoch(previous['retrieved_at'])<=86400,
                    'Recovery proof scope or time differs')
            refs.update(parents)
        record=dict(schema=1,provider=value['market'].lower(),day=value['date'],url=request['url'],params=request['params'],
                    http_status=200,retrieved_at=value['retrieved_at'],payload=json.loads(raw.read_bytes()))
        rows=parse_odd(record,value['market'].lower(),value['date'])
        require(len(rows)==value['rows'] and value['requirement']['stock_id'] in rows,'Receipt stock/count differs')
        return value,record,rows,refs

    def verify_cached(self,path):
        """Read a published wrapper and bind its entire new-helper source chain."""
        path=_bound(self.root,path);wrapper=read(path)
        value,record,rows=self._receipt(_bound(self.root,wrapper['source_receipt']))
        expected=dict(record,source_receipt=wrapper['source_receipt'],
            source_receipt_sha256=digest(self.root/wrapper['source_receipt']),
            source_raw=value['raw_path'],source_raw_sha256=value['raw_sha256'])
        require(path==self.root/DEST/('odd-'+record['provider']+'-'+record['day']+'.json')
                and wrapper==expected,'Published odd wrapper differs from verified source closure')
        self.refs[str(path.relative_to(self.root))]=digest(path)
        return dict(record=wrapper,rows=rows,source_sha256=dict(self.refs))

    def _proof(self,market,blocks,auth):
        path=self.cache/'receipts'/(market+'-'+FIRST[market][0]+'.json')
        require(path.exists(),'This endpoint needs its explicitly reviewed normal probe first')
        value,_,_=self._receipt(path)
        require(value.get('request_kind')=='single_normal_probe' and value.get('market')==market and value.get('authorization')==auth,
                'Recovery proof is from another authorization or endpoint')
        stamp=_epoch(value['retrieved_at'])
        require(0<=self.clock()-stamp<=86400,'Endpoint proof expired')
        require(all(stamp>_epoch(b['observed_at']) for b in blocks),'Newer origin stop invalidates proof')
        return dict(path=str(path.relative_to(self.root)),sha256=digest(path),retrieved_at=value['retrieved_at'])

    def fetch(self,market,day,stock_id,evidence,*,probe=False):
        request=item(market,day);market=request['market'];url=request['url'];key_name=market+'-'+day
        requirement=self._requirement(market,day,stock_id,evidence)
        if probe:require((day,stock_id)==FIRST[market],'Only the reviewed first ordinary probe is authorized')
        receipt_path=self.cache/'receipts'/(key_name+'.json')
        attempt_path=self.cache/'attempts'/(key_name+'.json')
        state_path,origin_lock,hold_path=OfficialDailyAcquisition._origin_paths(self,request)
        with file_lock(self.cache/'acquisition.lock',timeout=0):
            if receipt_path.exists():
                require(receipt_path.with_suffix('.sha256').read_text().strip()==digest(receipt_path),'Receipt changed')
                if not read(receipt_path).get('accepted'):
                    return dict(read(receipt_path),resumed_without_retry=True)
                value,record,rows=self._receipt(receipt_path)
                require(stock_id in rows,'Requested stock missing from accepted cached date')
                self._publish(receipt_path,record)
                return dict(value,resumed_without_retry=True)
            require(not attempt_path.exists(),'Interrupted odd attempt; automatic retry forbidden')
            require(len(list((self.cache/'attempts').glob('*.json')))<50,'Persistent 50-attempt budget exhausted')
            auth=self._auth(create=probe)
            with file_lock(origin_lock):
                state=read(state_path) if state_path.exists() else dict(probes={})
                blocks=OfficialDailyAcquisition._blocks(self,state,hold_path)
                key=sha256((auth['sha256']+'\n'+url).encode()).hexdigest()
                proof=None
                if probe:
                    require(key not in state.get('probes',{}),'Single normal probe already consumed')
                    require(all(_epoch(auth['created_at'])>_epoch(b['observed_at']) for b in blocks),
                            'Origin stopped after this authorization; do not probe')
                else:proof=self._proof(market,blocks,auth)
                wait=max(0,3.1-(self.clock()-state.get('last_start_epoch',float('-inf'))))
                if wait:self.sleep(wait+.001)
                require(self._auth()==auth,'Authorization changed while waiting')
                blocks=OfficialDailyAcquisition._blocks(self,state,hold_path)
                if not probe:proof=self._proof(market,blocks,auth)
                else:require(all(_epoch(auth['created_at'])>_epoch(b['observed_at']) for b in blocks),'New origin stop during probe wait')
                hold_hash=digest(hold_path) if hold_path.exists() else None
                started=self.clock();stamp=datetime.fromtimestamp(started,timezone.utc).isoformat()
                helper_hash=digest(Path(__file__))
                helper_snapshot=self.cache/'source-snapshots'/(helper_hash+'.py')
                if helper_snapshot.exists():require(digest(helper_snapshot)==helper_hash,'Helper snapshot changed')
                else:
                    helper_snapshot.parent.mkdir(parents=True,exist_ok=True)
                    with helper_snapshot.open('xb') as stream:stream.write(Path(__file__).read_bytes())
                    require(digest(helper_snapshot)==helper_hash,'Helper changed before dispatch')
                attempt=dict(schema='vp_odd_attempt_v1',**request,requirement=requirement,authorization=auth,
                    started_at=stamp,request_kind='single_normal_probe' if probe else 'necessary_missing_day',
                    recovery_proof=proof,helper_sha256=helper_hash,global_hold_sha256=hold_hash)
                _write(attempt_path,attempt,exclusive=True)
                state['last_start_epoch']=started;state['in_flight']=dict(started_at=stamp,identity='vp-odd-'+key_name)
                if probe:state.setdefault('probes',{})[key]=dict(started_at=stamp,identity='vp-odd-'+key_name)
                _write(state_path,state)
                result=dict(attempt,schema='vp_odd_receipt_v1',accepted=False,http_status=None,
                            automatic_redirects_disabled=True,redirect_statuses=[],security_denied=False)
                try:
                    if hasattr(self.session,'cookies'):self.session.cookies.clear()
                    response=self.session.get(url,params=request['params'],timeout=(10,30),allow_redirects=False)
                    raw=self.cache/'raw'/(key_name+'.bin');raw.parent.mkdir(parents=True,exist_ok=True)
                    with raw.open('xb') as stream:stream.write(response.content)
                    security=_security_response(response)
                    result.update(http_status=response.status_code,raw_path=str(raw.relative_to(self.root)),
                        raw_sha256=digest(raw),bytes=len(response.content),security_denied=security,
                        redirect_statuses=[r.status_code for r in getattr(response,'history',[])])
                    if security:result['status']='origin_stopped'
                    elif response.status_code!=200:result['status']='http_error_no_retry'
                    else:
                        try:
                            record=dict(schema=1,provider=market.lower(),day=day,url=url,params=request['params'],
                                http_status=200,retrieved_at=stamp,payload=json.loads(response.content))
                            rows=parse_odd(record,market.lower(),day)
                            require(stock_id in rows,'Required stock absent from official odd table')
                            result.update(accepted=True,status='verified_intraday_odd_day',rows=len(rows),
                                          required_stock_row=rows[stock_id])
                        except (ReplayDataUnavailable,ValueError,KeyError,TypeError,UnicodeDecodeError) as exc:
                            result.update(status='schema_error_no_retry',error_type=type(exc).__name__)
                except requests.RequestException as exc:
                    response=getattr(exc,'response',None)
                    result.update(status='transport_error_no_retry',error_type=type(exc).__name__,
                                  exception_response_present=response is not None)
                    if response is not None:
                        raw=self.cache/'raw'/(key_name+'.bin');raw.parent.mkdir(parents=True,exist_ok=True)
                        with raw.open('xb') as stream:stream.write(response.content)
                        result.update(http_status=response.status_code,raw_path=str(raw.relative_to(self.root)),
                            raw_sha256=digest(raw),bytes=len(response.content),security_denied=_security_response(response),
                            redirect_statuses=[r.status_code for r in getattr(response,'history',[])])
                        if result['security_denied']:result['status']='origin_stopped'
                result['retrieved_at']=datetime.fromtimestamp(self.clock(),timezone.utc).isoformat()
                result['global_hold_unchanged']=(digest(hold_path) if hold_path.exists() else None)==hold_hash
                if not result['global_hold_unchanged']:result.update(accepted=False,status='global_hold_changed')
                _write(receipt_path,result,exclusive=True)
                receipt_path.with_suffix('.sha256').write_text(digest(receipt_path)+'\n')
                for p in (attempt_path,receipt_path,receipt_path.with_suffix('.sha256'),self.cache/'authorization.json',
                          self.root/PREREG,Path(__file__),helper_snapshot,self.root/requirement['path']):
                    self.refs[str(p.relative_to(self.root))]=digest(p)
                if result.get('raw_path'):self.refs[result['raw_path']]=result['raw_sha256']
                if result['security_denied']:
                    state['stopped']=dict(observed_at=result['retrieved_at'],kind='security_or_rate_limit',
                        receipt_path=str(receipt_path.relative_to(self.root)),receipt_sha256=digest(receipt_path))
                state.pop('in_flight',None);_write(state_path,state)
                if result['accepted']:
                    _,record,_=self._receipt(receipt_path);self._publish(receipt_path,record)
                return result

    def _publish(self,receipt_path,record):
        value,verified,_=self._receipt(receipt_path)
        require(record==verified,'Publish record differs from verified receipt')
        dest=self.root/DEST/('odd-'+record['provider']+'-'+record['day']+'.json')
        record=dict(record,source_receipt=str(receipt_path.relative_to(self.root)),
            source_receipt_sha256=digest(receipt_path),source_raw=value['raw_path'],source_raw_sha256=value['raw_sha256'])
        if dest.exists():
            require(read(dest)==record,'Existing execution evidence differs; never replace')
        else:_write(dest,record,exclusive=True)
        paths=[self.root/PREREG,Path(__file__),self.cache/'authorization.json',receipt_path,
               receipt_path.with_suffix('.sha256'),self.root/value['raw_path'],
               self.root/value['requirement']['path'],dest]
        if value.get('recovery_proof'):paths.append(self.root/value['recovery_proof']['path'])
        hold=self.root/'.cache/official-origin-holds'/(('www.twse.com.tw' if value['market']=='TWSE' else 'www.tpex.org.tw')+'.json')
        if hold.exists():paths.append(hold)
        for p in paths:self.refs[str(p.relative_to(self.root))]=digest(p)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--market',choices=('TWSE','TPEX'),required=True)
    parser.add_argument('--date',required=True)
    parser.add_argument('--stock-id',required=True)
    parser.add_argument('--evidence',required=True,type=Path)
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    result=OddCompletion().fetch(args.market,args.date,args.stock_id,args.evidence,probe=args.probe)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['accepted'] else 2)
