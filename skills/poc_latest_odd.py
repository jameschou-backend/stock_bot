"""Scoped normal odd-lot requests for the 2026-09-10..10-02 continuation.

The existing per-origin dispatch lock and security holds remain authoritative.
No retries, redirects, alternate identities or inferred zero volumes.
"""
from datetime import date, datetime, timezone
from pathlib import Path
import json
import time
import requests
from app.file_lock import file_lock
from skills.official_daily_acquisition import OfficialDailyAcquisition, _security_response, _write, _epoch as _base_epoch, digest, read
from skills.replay_market_feeds import parse_odd, URLS, ReplayDataUnavailable

START, END = '2026-09-10', '2026-10-02'
PREREG = 'docs/prereg_poc_latest_20261003.md'


def _epoch(stamp):
    return _base_epoch(stamp[:-1]+'+00:00' if stamp.endswith('Z') else stamp)


def wrapper(record, payload):
    return dict(schema=1,provider=record['market'].lower(),day=record['date'],
        url=record['url'],params=record['params'],http_status=record['http_status'],
        retrieved_at=record.get('retrieved_at') or record['started_at'],payload=payload)


def requirement(day, sid, engine):
    plans=list(getattr(engine,'day_plans',{}).values())
    found=[p for p in plans if p.get('date')==day and p.get('stock_id')==sid
           and p.get('planned_qty',0)>0 and p.get('odd_qty',0)>0]
    if not found: raise ValueError('Odd acquisition requires a precommitted positive odd-share order')
    return found


class LatestOddData:
    def __init__(self, root, *, online=False, session=None, clock=time.time, sleep=time.sleep):
        self.root=Path(root).resolve();self.cache=self.root/'.cache/poc-latest-20261003/odd-v1'
        self.online=online;self.refs={};self.clock=clock;self.sleep=sleep
        self.session=session or requests.Session()
        if session is None:self.session.trust_env=False
        self.cache.mkdir(parents=True,exist_ok=True)
        self.auth=self.cache/'authorization.json'
        expected=dict(schema='poc_latest_odd_authorization_v1',user_request='go',
            scope='extend POC account through 2026-10-02 using necessary preplanned odd-lot orders',
            start=START,end=END,urls=URLS,maximum_attempts=30,minimum_interval_seconds=3.1,
            retries=0,security_bypass_authorized=False,prereg_sha256=digest(self.root/PREREG))
        if not self.auth.exists():
            _write(self.auth,dict(expected,created_at=datetime.fromtimestamp(clock(),timezone.utc).isoformat()),exclusive=True)
        value=read(self.auth)
        if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Latest odd authorization changed')
        self.authorization=dict(path=str(self.auth.relative_to(self.root)),sha256=digest(self.auth),created_at=value['created_at'])
        self.mark(self.auth);self.mark(self.root/PREREG)

    def mark(self,path,expected=None):
        path=Path(path).resolve();path.relative_to(self.root);h=digest(path)
        key=str(path.relative_to(self.root))
        if expected is not None and h!=expected:raise ValueError('Odd source hash changed: '+key)
        if key in self.refs and self.refs[key]!=h:raise ValueError('Odd source mutated: '+key)
        self.refs[key]=h;return h

    def cached(self,path,day,market,seen=None):
        seen=set() if seen is None else set(seen)
        if path in seen:raise ValueError('Cyclic odd evidence')
        seen.add(path)
        r=read(path);self.mark(path);self.mark(path.with_suffix('.sha256'))
        if path.with_suffix('.sha256').read_text().strip()!=digest(path):raise ValueError('Odd receipt hash mismatch')
        if not r.get('accepted'):raise ReplayDataUnavailable('Prior odd request failed; no automatic retry: '+market+' '+day)
        if (r.get('schema')!='poc_latest_odd_receipt_v1' or r.get('date')!=day or r.get('market')!=market
                or r.get('url')!=URLS[market.lower()] or not START<=day<=END
                or r.get('http_status')!=200 or r.get('security_denied') is not False
                or r.get('automatic_redirects_disabled') is not True
                or r.get('global_hold_unchanged') is not True):raise ValueError('Odd source identity mismatch')
        self.mark(self.root/r['raw_path'],r['raw_sha256'])
        self.mark(self.root/r['authorization']['path'],r['authorization']['sha256'])
        self.mark(self.root/r['demand_path'],r['demand_sha256'])
        if r['authorization']!=self.authorization:raise ValueError('Odd authorization differs')
        start,end=_epoch(r['started_at']),_epoch(r['retrieved_at'])
        if not 0<=start-_epoch(self.authorization['created_at'])<=86400 or end<start:
            raise ValueError('Odd evidence authorization time differs')
        attempt=self.cache/'attempts'/(market+'-'+day+'.json');self.mark(attempt,r['attempt_sha256'])
        a=read(attempt)
        if any(r.get(k)!=v for k,v in a.items() if k!='accepted'):raise ValueError('Odd receipt differs from dispatch')
        helper=self.cache/'source-snapshots'/(r['helper_sha256']+'.py');self.mark(helper,r['helper_sha256'])
        demand=read(self.root/r['demand_path'])
        if demand['date']!=day or not any(p.get('date')==day and p.get('stock_id')==demand['stock_id']
                and p.get('odd_qty',0)>0 and p.get('planned_qty',0)>0 for p in demand['precommitted_plans']):
            raise ValueError('Odd source demand differs')
        _,_,hold=OfficialDailyAcquisition._origin_paths(self,r)
        if r['global_hold_sha256'] is not None:
            self.mark(hold,r['global_hold_sha256'])
            for name,h in read(hold).get('evidence_sha256',{}).items():self.mark(self.root/name,h)
        proof=r.get('recovery_proof')
        if r['request_kind']=='necessary_preplanned_day':
            if not isinstance(proof,dict):raise ValueError('Odd endpoint proof absent')
            parent=self.root/proof['path'];self.mark(parent,proof['sha256']);prior=read(parent)
            if prior['request_kind']!='single_normal_probe' or prior['url']!=r['url'] or _epoch(prior['retrieved_at'])>start:
                raise ValueError('Odd endpoint proof invalid')
            self.cached(parent,prior['date'],market,seen)
        elif r['request_kind']!='single_normal_probe' or proof is not None:raise ValueError('Unknown odd request kind')
        rows=parse_odd(wrapper(r,read(self.root/r['raw_path'])),market.lower(),day)
        if len(rows)!=r['rows'] or demand['stock_id'] not in rows:raise ValueError('Odd stock or row count differs')
        return rows

    def get(self,day,sid,market,engine=None):
        market=market.upper();stamp=date.fromisoformat(day)
        if market.lower() not in URLS or not START<=day<=END:raise ValueError('Latest odd request outside scope')
        if len(sid)!=4 or not sid.isdigit():raise ValueError('Require four-digit security')
        key=market+'-'+day;receipt=self.cache/'receipts'/(key+'.json')
        if receipt.exists():
            rows=self.cached(receipt,day,market)
            if sid not in rows:raise ReplayDataUnavailable('Odd table lacks stock: '+key+' '+sid)
            return rows[sid]
        if not self.online:raise ReplayDataUnavailable('Latest official odd day missing: '+key)
        plans=requirement(day,sid,engine)
        demand=self.cache/'demands'/(key+'-'+sid+'.json')
        if not demand.exists():_write(demand,dict(date=day,stock_id=sid,precommitted_plans=plans),exclusive=True)
        request=dict(url=URLS[market.lower()],market=market,date=day)
        state_path,origin_lock,hold=OfficialDailyAcquisition._origin_paths(self,request)
        attempt=self.cache/'attempts'/(key+'.json');proof_path=self.cache/('proof-'+market+'.json')
        with file_lock(self.cache/('run-'+market+'.lock')),file_lock(origin_lock):
            if receipt.exists():return self.get(day,sid,market,engine)
            if attempt.exists():raise ReplayDataUnavailable('Previous odd attempt unfinished; no retry')
            if len(list((self.cache/'attempts').glob('*.json')))>=30:raise ReplayDataUnavailable('Latest odd budget exhausted')
            if not 0<=self.clock()-_epoch(self.authorization['created_at'])<=86400:raise ReplayDataUnavailable('Odd authorization expired')
            state=read(state_path) if state_path.exists() else dict(probes={})
            blocks=OfficialDailyAcquisition._blocks(self,state,hold)
            is_probe=not proof_path.exists();proof=None
            if is_probe:
                if any(_epoch(b['observed_at'])>=_epoch(self.authorization['created_at']) for b in blocks):
                    raise ReplayDataUnavailable('Newer origin stop prohibits odd probe')
            else:
                proof=read(proof_path);self.mark(self.root/proof['path'],proof['sha256'])
                recovered=read(self.root/proof['path'])
                if not recovered['accepted'] or recovered['url']!=request['url']:raise ReplayDataUnavailable('Odd endpoint has no successful proof')
                self.cached(self.root/proof['path'],recovered['date'],market)
                if any(_epoch(b['observed_at'])>=_epoch(recovered['retrieved_at']) for b in blocks):
                    raise ReplayDataUnavailable('Newer origin stop invalidates odd endpoint proof')
            wait=max(0,3.1-(self.clock()-state.get('last_start_epoch',0)))
            if wait:self.sleep(wait+.001)
            now=datetime.fromtimestamp(self.clock(),timezone.utc).isoformat()
            params=dict(date=stamp.strftime('%Y%m%d' if market=='TWSE' else '%Y/%m/%d'),response='json')
            helper_hash=digest(Path(__file__));helper=self.cache/'source-snapshots'/(helper_hash+'.py')
            if not helper.exists():
                helper.parent.mkdir(parents=True,exist_ok=True);helper.write_bytes(Path(__file__).read_bytes())
            record=dict(schema='poc_latest_odd_receipt_v1',**request,params=params,authorization=self.authorization,
                request_kind='single_normal_probe' if is_probe else 'necessary_preplanned_day',started_at=now,
                demand_path=str(demand.relative_to(self.root)),demand_sha256=digest(demand),accepted=False,
                global_hold_sha256=digest(hold) if hold.exists() else None,helper_sha256=helper_hash,recovery_proof=proof)
            _write(attempt,record,exclusive=True)
            record['attempt_sha256']=digest(attempt)
            state['last_start_epoch']=self.clock();state['in_flight']=dict(started_at=now,identity=key)
            _write(state_path,state)
            try:
                response=self.session.get(request['url'],params=params,timeout=(10,30),allow_redirects=False)
                raw=self.cache/'raw'/(key+'.json');raw.parent.mkdir(parents=True,exist_ok=True)
                with raw.open('xb') as stream:stream.write(response.content)
                denied=_security_response(response)
                record.update(raw_path=str(raw.relative_to(self.root)),raw_sha256=digest(raw),http_status=response.status_code,
                    security_denied=denied,automatic_redirects_disabled=True)
                if denied:record['status']='origin_stopped'
                elif response.status_code!=200:record['status']='http_error_no_retry'
                else:
                    rows=parse_odd(wrapper(record,json.loads(response.content)),market.lower(),day)
                    if not rows:raise ValueError('Empty odd table')
                    record.update(accepted=True,status='verified_odd_market_day',rows=len(rows))
            except (requests.RequestException,ReplayDataUnavailable,ValueError,KeyError,TypeError) as exc:
                record.update(status='request_failed_no_retry',error_type=type(exc).__name__)
            record['retrieved_at']=datetime.fromtimestamp(self.clock(),timezone.utc).isoformat()
            record['global_hold_unchanged']=(digest(hold) if hold.exists() else None)==record['global_hold_sha256']
            if not record['global_hold_unchanged']:
                record.update(accepted=False,status='global_hold_changed')
            _write(receipt,record,exclusive=True);receipt.with_suffix('.sha256').write_text(digest(receipt)+'\n')
            if record.get('security_denied'):
                state['stopped']=dict(observed_at=record['retrieved_at'],kind='security_or_rate_limit',
                    receipt_path=str(receipt.relative_to(self.root)),receipt_sha256=digest(receipt))
            state.pop('in_flight',None);_write(state_path,state)
            if is_probe:
                _write(proof_path,dict(path=str(receipt.relative_to(self.root)),sha256=digest(receipt)),exclusive=True)
        rows=self.cached(receipt,day,market)
        if sid not in rows:raise ReplayDataUnavailable('Odd table lacks stock: '+key+' '+sid)
        return rows[sid]
