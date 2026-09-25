#!/usr/bin/env python3
"""Finish basket acquisition and offline-replay the complete TWSE ordinary audit."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from app.file_lock import file_lock
from skills.board_tape_reconciliation import digest, summarize_ticks, reconcile, verify_report as verify_v1
from skills.board_tape_reconciliation_v2 import basket_requests,basket_key,parse_twse_complete,parse_basket_detail,verify_report
from scripts.prepare_twse_board_reconciliation_v2 import CACHE,Fetcher,identity,encode,relative,dispatch_order,plan

BASE=ROOT/'artifacts/forward_simulation/board_tape_reconciliation_20260925.json'
OUTPUT=ROOT/'artifacts/forward_simulation/board_tape_reconciliation_v2_20260925.json'
CODE=['skills/board_tape_reconciliation_v2.py','scripts/audit_twse_board_reconciliation_v2.py',
      'scripts/prepare_twse_board_reconciliation_v2.py']


class Sources:
    def __init__(self,cache,refs):
        self.cache=Path(cache);self.refs=refs
        self.plan=json.loads(self.read(self.cache/'plan.json'))
        if digest(self.cache/'plan.json')!=self.read(self.cache/'plan.sha256').strip():raise ValueError('Plan hash differs')
        self.events=[json.loads(line) for line in self.read(self.cache/'requests.jsonl').splitlines()]
        self.success={}
        for event in self.events:
            if event['event']=='finish' and event.get('accepted'):self.success[event['identity']]=event
        self.entries={(v['date'],v['kind']):v for v in self.plan['entries']}
    def read(self,path):
        path=Path(path).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('Source escapes repository')
        self.refs[relative(path)]=digest(path)
        return path.read_text()
    def payload(self,item):
        source=item.get('reused') or self.success.get(item['identity'])
        if not source:return None
        raw_path=ROOT/source['raw_path']
        text=self.read(raw_path)
        if digest(raw_path)!=source['raw_sha256']:raise ValueError('Official source hash differs')
        if source.get('metadata_path'):
            meta=ROOT/source['metadata_path'];self.read(meta)
            if digest(meta)!=source['metadata_sha256']:raise ValueError('Reused source metadata differs')
        value=json.loads(text)
        return value['payload'] if source.get('record_wrapper') else value
    def base(self,day):
        return {kind:self.payload(item) for (stamp,kind),item in self.entries.items() if stamp==day}


def detail_item(day,request):
    # Official bfiauu-detail.html supplies only these four source identities;
    # selectType=M asks for the parent list again and must not be sent here.
    params=dict(request['identity'],response='json')
    item=dict(kind='block_detail',date=day,url='https://www.twse.com.tw/rwd/zh/block/BFIAUU',params=params)
    item['identity']=identity(item)
    return item


def receipt_payload(item,receipt):
    source=item.get('reused') or receipt
    if not source or not source.get('accepted',bool(item.get('reused'))):return None
    raw=ROOT/source['raw_path']
    if digest(raw)!=source['raw_sha256']:raise ValueError('Acquisition payload changed')
    if source.get('metadata_path') and digest(ROOT/source['metadata_path'])!=source['metadata_sha256']:
        raise ValueError('Acquisition metadata changed')
    payload=json.loads(raw.read_text())
    return payload['payload'] if source.get('record_wrapper') else payload


def fetch_complete_days(cache=CACHE,allow_security_retry=False):
    """Acquire each date's components and constituents before the next date."""
    cache=Path(cache);base=plan(cache)
    with file_lock(cache/'fetch.lock',timeout=1):
        client=Fetcher(cache,base['max_http_requests'],allow_security_retry=allow_security_retry)
        by_day={}
        for item in dispatch_order(base['entries']):by_day.setdefault(item['date'],[]).append(item)
        progress=dict(schema='twse_complete_day_preparation_v2',plan_sha256=digest(cache/'plan.json'),
            dispatch_order='date_then_components_then_basket_details',days=[],actual_http_requests=len(client.attempts()))
        def checkpoint():
            progress['actual_http_requests']=len(client.attempts())
            (cache/'complete-day-progress.json').write_text(encode(progress)+'\n')
        try:
            for day,entries in by_day.items():
                parts={}
                for item in entries:
                    receipt=None if item.get('reused') else client.fetch(item)
                    parts[item['kind']]=receipt_payload(item,receipt)
                missing=[kind for kind,value in parts.items() if value is None]
                result=dict(date=day,status='missing',missing=missing,validated_baskets=0)
                if not missing:
                    details={}
                    for request in basket_requests(parts['block_basket'],day):
                        item=detail_item(day,request);receipt=client.fetch(item)
                        payload=receipt_payload(item,receipt)
                        if payload is None:raise ValueError('Required basket detail request did not return dated JSON')
                        # First semantic failure stops the batch; do not repeat
                        # a wrongly interpreted endpoint across every basket.
                        parse_basket_detail(payload,request,day)
                        details[basket_key(request)]=payload
                    parse_twse_complete(parts,details,day)
                    result.update(status='fully_decomposed',validated_baskets=len(details))
                progress['days'].append(result);checkpoint()
                print(encode(dict(event='date_completed',**result)),flush=True)
        except Exception as exc:
            progress['stopped']=dict(error_type=type(exc).__name__,message=str(exc),at=datetime.now(timezone.utc).isoformat())
            checkpoint();raise
        return progress


def fetch_baskets(cache=CACHE):
    cache=Path(cache)
    with file_lock(cache/'fetch.lock',timeout=1):
        source=Sources(cache,{})
        client=Fetcher(cache,source.plan['max_http_requests'])
        requests=[]
        for day in sorted({day for day,kind in source.entries}):
            parent=source.payload(source.entries[(day,'block_basket')])
            if parent is None:raise ValueError('Acquire all parent basket lists before constituent phase')
            for request in basket_requests(parent,day):
                item=detail_item(day,request)
                requests.append(dict(day=day,parent=request,item=item))
        plan=dict(schema='twse_basket_constituent_plan_v2',parent_plan_sha256=digest(cache/'plan.json'),
                  required_baskets=len(requests),entries=requests)
        path=cache/'basket-plan.json'
        if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Basket plan changed')
        path.write_text(encode(plan)+'\n');path.with_suffix('.sha256').write_text(digest(path)+'\n')
        results=[]
        for entry in requests:
            receipt=client.fetch(entry['item'])
            if not receipt.get('accepted'):raise ValueError('Official basket response unavailable: '+str(entry['item']))
            payload=json.loads((ROOT/receipt['raw_path']).read_text())
            # Stop after a schema failure so an incorrect endpoint is not repeated
            # against every basket. Its raw response and charged attempt remain.
            parsed=parse_basket_detail(payload,entry['parent'],entry['day'])
            results.append(dict(identity=entry['item']['identity'],basket_key=basket_key(entry['parent']),
                                date=entry['day'],security_count=len(parsed)))
        result=dict(schema='twse_basket_preparation_v2',baskets=len(results),actual_http_requests=len(client.attempts()),
                    results=results)
        (cache/'basket-progress.json').write_text(encode(result)+'\n')
        print(encode({k:v for k,v in result.items() if k!='results'}),flush=True)
        return result


def run(cache=CACHE,output=OUTPUT,verify=False):
    cache,output=Path(cache),Path(output)
    baseline=verify_v1(BASE,ROOT)
    refs=dict(baseline['input_sha256']);refs.update(baseline['code_sha256'])
    refs[relative(BASE)]=digest(BASE);refs[relative(BASE.with_suffix('.sha256'))]=digest(BASE.with_suffix('.sha256'))
    source=Sources(cache,refs)
    origin_hold=None
    if (cache/'origin-hold.json').exists():
        origin_hold=json.loads(source.read(cache/'origin-hold.json'))
        for name,expected in origin_hold['evidence_sha256'].items():
            path=ROOT/name;source.read(path)
            if digest(path)!=expected:raise ValueError('Origin hold evidence changed')
    # The local hold is the immutable point-in-time copy. The shared guard is
    # checked by Fetcher, but later authorized source recovery must not rewrite
    # this report's historical evidence.
    for event in source.events:
        if event['event']=='finish' and event.get('raw_path'):
            raw_path=ROOT/event['raw_path'];source.read(raw_path)
            if digest(raw_path)!=event['raw_sha256']:raise ValueError('Acquisition response changed')
    rows=[dict(r) for r in baseline['rows'] if r['market']!='TWSE']
    current=[r for r in baseline['rows'] if r['market']=='TWSE']
    by_day={};failures={};known_baskets=[];known_basket_dates=0;successful_dates=0
    for day in sorted({r['date'] for r in current}):
        parent=source.payload(source.entries[(day,'block_basket')])
        if parent is None:continue
        known_basket_dates+=1
        for request in basket_requests(parent,day):
            item=detail_item(day,request)
            known_baskets.append(dict(date=day,identity=item['identity'],basket_key=basket_key(request),
                acquired=item['identity'] in source.success,parent=request,item=item))
    for day in sorted({r['date'] for r in current}):
        parts=source.base(day)
        missing=[k for k,v in parts.items() if v is None]
        if missing:
            failures[day]='Missing required official components: '+','.join(missing);continue
        details={}
        try:
            needed=basket_requests(parts['block_basket'],day)
            for request in needed:
                payload=source.payload(detail_item(day,request))
                if payload is None:raise ValueError('Basket constituent source not acquired: '+basket_key(request))
                details[basket_key(request)]=payload
            by_day[day]=parse_twse_complete(parts,details,day);successful_dates+=1
        except (ValueError,KeyError,TypeError) as exc:
            failures[day]=type(exc).__name__+': '+str(exc)
    for previous in current:
        day,sid=previous['date'],previous['stock_id'];path=ROOT/previous['tape_path']
        source.read(path) if path.suffix=='.json' else refs.update({relative(path):digest(path)})
        if digest(path)!=previous['tape_sha256']:raise ValueError('Sealed tick source changed')
        raw=pd.read_parquet(path);tape=summarize_ticks(raw,sid,day,'TWSE')
        official=by_day.get(day,{}).get(sid)
        if official is None:
            result=dict(status='official_daily_source_missing' if day not in by_day else 'official_stock_row_missing',
                reason=failures.get(day,'Official stock row missing'),tape=tape,same_scope_aggregate_matched=False,
                transaction_count_verified=False,tick_sequence_complete=False,accepted_for_strict_replay=False,
                own_order_fill_proven=False,live_qualified=False)
        else:
            result=reconcile(tape,official)
            if result['status']=='daily_aggregate_conflict':
                result['diagnosis']=dict(raw_rows=len(raw),normalization_dropped_rows=tape['normalization_dropped_rows'],
                    raw_all_session_shares=tape['raw_all_session_shares'],fixed_price_shares=tape['fixed_price_shares'],
                    unknown_session_rows=tape['unknown_session_rows'],raw_duplicate_rows=int(raw.duplicated().sum()),
                    duplicate_rows_removed=False,official_minus_regular_shares=official['shares']-tape['shares'],
                    official_minus_regular_amount_cents=official['amount_cents']-tape['amount_cents'],
                    classification='share_quantity_conflict' if 'shares' in result['differences'] else 'amount_or_price_conflict',
                    cause='unresolved_provider_vs_exchange_aggregate_discrepancy',no_claim_of_specific_missing_transaction=True)
        rows.append(dict(date=day,stock_id=sid,market='TWSE',tape_path=previous['tape_path'],
                         tape_sha256=previous['tape_sha256'],**result))
    rows.sort(key=lambda r:(r['date'],r['stock_id']))
    unresolved_base=[i for i in source.plan['entries'] if 'reused' not in i and i['identity'] not in source.success]
    attempted={e['identity'] for e in source.events if e['event']=='start'}
    def count(group):
        return dict(required_sessions=len(group),same_scope_aggregate_matched=sum(r['same_scope_aggregate_matched'] for r in group),
            aggregate_conflicts=sum(r['status']=='daily_aggregate_conflict' for r in group),
            share_quantity_conflicts=sum('shares' in r.get('differences',{}) for r in group),
            amount_only_conflicts=sum(set(r.get('differences',{}))=={'amount_cents'} for r in group),
            independent_daily_source_missing=sum(r['status'].endswith('_missing') for r in group),
            transaction_count_verified=0,tick_sequence_complete=0,accepted_for_strict_replay=0)
    result=dict(schema='board_tape_reconciliation_v2',as_of='2026-09-25',scope='20_frozen_account_required_stock_days',
        summary=count(rows),markets={m:count([r for r in rows if r['market']==m]) for m in ('TWSE','TPEX')},rows=rows,
        quarantine=[dict(date=r['date'],stock_id=r['stock_id'],market=r['market'],tape_sha256=r['tape_sha256'],
                reason='official_daily_aggregate_conflict',differences=r['differences'],diagnosis=r['diagnosis'])
                for r in rows if r['status']=='daily_aggregate_conflict'],
        official_acquisition=dict(base_missing_planned=source.plan['base_missing_requests'],
            origin_hold=origin_hold,acquisition_complete=not failures,
            status='blocked_by_official_origin' if origin_hold else ('complete' if not failures else 'incomplete'),
            actual_http_requests=sum(e['event']=='start' for e in source.events),max_http_requests=source.plan['max_http_requests'],
            request_count_semantics='recorded_transport_invocations; historical auto-redirect hops were not instrumented',
            http_wire_request_count_exact=all(e.get('automatic_redirects_disabled') is True
                for e in source.events if e['event']=='finish' and e.get('http_status')),
            transport_invocations_by_kind=dict(Counter(e['kind'] for e in source.events if e['event']=='start')),
            remaining_base_identities=len(unresolved_base),
            previously_attempted_unresolved_base_identities=sum(i['identity'] in attempted for i in unresolved_base),
            known_minimum_remaining_requests=len(unresolved_base)+sum(not b['acquired'] for b in known_baskets),
            reused_base_components=sum('reused' in e for e in source.plan['entries']),
            required_twse_dates=source.plan['required_dates'],fully_decomposed_twse_dates=successful_dates,
            parent_basket_dates_available=known_basket_dates,
            required_basket_details_discovered=len(known_baskets),
            acquired_basket_details=sum(b['acquired'] for b in known_baskets),
            basket_discovery_complete=known_basket_dates==source.plan['required_dates'],
            pending_discovered_basket_details=[b for b in known_baskets if not b['acquired']],failed_dates=failures,
            response_acceptance_is_not_session_coverage=True,
            successful_response_attempts=sum(e['event']=='finish' and e.get('accepted') for e in source.events),
            blocked_response_attempts=sum(e['event']=='finish' and not e.get('accepted') for e in source.events),
            undispatched_base_identities=sum('reused' not in i and not any(e['event']=='start' and e['identity']==i['identity']
                for e in source.events) for i in source.plan['entries'])),
        input_sha256=refs,code_sha256={p:digest(ROOT/p) for p in CODE},
        preparation_finmind_requests=0,network_requests=0,database_mutations=0,
        strict_data_ready=False,live_qualified=False,own_order_fill_proven=False,
        limitations=baseline['limitations']+[
            'TPEX findings are carried unchanged from v1; later correction research is a separate evidence artifact.',
            'TWSE nonempty basket blocks are explicitly decomposed using source-provided identities and exact parent totals.',
            'Daily-total matches still do not prove tick sequence completeness, executable queue position or own fill.'])
    text=json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if verify:
        if output.read_text()!=text:raise ValueError('Full board offline reproduction differs')
        verify_report(output,ROOT)
    else:
        if output.exists() and output.read_text()!=text:raise ValueError('Result is immutable; choose another output')
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text)
        output.with_suffix('.sha256').write_text(digest(output)+'\n')
    print(encode(dict(summary=result['summary'],markets=result['markets'],verified=verify)))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--fetch-baskets',action='store_true');p.add_argument('--verify',action='store_true')
    p.add_argument('--fetch-complete-days',action='store_true');p.add_argument('--retry-security-after-cooldown',action='store_true')
    args=p.parse_args()
    if args.fetch_complete_days:
        if args.verify or args.fetch_baskets:p.error('Choose one acquisition or offline verification operation')
        fetch_complete_days(args.cache,args.retry_security_after_cooldown)
    elif args.fetch_baskets:
        if args.verify:p.error('Online preparation and offline verification are separate commands')
        fetch_baskets(args.cache)
    else:run(args.cache,args.output,args.verify)
