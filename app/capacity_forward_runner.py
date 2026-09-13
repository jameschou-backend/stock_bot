"""Bounded three-account forward workflow, outside all previous sealed runners."""
from datetime import time
from pathlib import Path
import json
import sys
import time as walltime
from app import capacity_forward as policy
from app import forward_automation as base, forward_journal as j, forward_portfolio as p
from app import forward_simulation as sim, forward_corporate_audit as corporate
from app import forward_market_quotes as market, forward_halts as halts
from app.file_lock import file_lock


def books(root):
    paths={role:Path(root)/(role+'.sqlite3') for role in policy.ROLES}
    for role,path in paths.items(): policy.verify(path,role)
    return paths


def refresh(paths,clock=j.now,budget=12):
    if type(budget) is not int or not 1<=budget<=12: raise ValueError('三帳本合計更新上限為1至12')
    calls=reused=0
    for path in paths.values():
        if calls>=budget: break
        r=corporate.refresh(path,clock=clock,request_budget=budget-calls)
        calls+=r['calls'];reused+=r['reused']
    return dict(calls=calls,reused=reused)


def observe(paths,clock,sleeper,seconds,fetcher):
    refresh_result=refresh(paths,clock)
    queries=[];ready={};outputs=[]
    for role,path in paths.items():
        rows=policy.verify(path,role)
        if role!='benchmark' and policy.risk(rows)['blocked']:
            policy.pause(path,True,'整戶風險條件限制新增；保留既有出場委託',clock)
            rows=policy.verify(path,role)
        audit=corporate.inspect(path,clock=clock,rows=rows)
        if audit['blocked']:
            outputs.append(dict(role=role,status='blocked',reason='公司行動來源缺漏或待核對事件'))
            continue
        state=p.state(rows);day=str(clock().astimezone(p.TZ).date())
        current=[o for o in state['orders'].values() if not o['closed'] and o['filled']<o['qty'] and o['session']==day]
        lookup=base.markets(sorted({o['stock_id'] for o in current}))
        for o in current:
            key=(lookup[o['stock_id']],o['stock_id'],o['channel'])
            if key not in queries: queries.append(key)
        ready[role]=path
        outputs.append(dict(role=role,status='collect',orders=len(current)))
    if len(queries)>24: raise ValueError('行情種類超過24組，停止非預期抓取')
    for iteration in range(2):
        for key in queries:
            observation=fetcher(*key)
            for role,path in ready.items():
                result=sim.match(path,observation,clock)
                outputs.append(dict(role=role,status='observation',query=list(key),
                    fills=len(result.get('fills',[])),reasons=result.get('reasons',[])))
        if iteration==0 and queries: sleeper(seconds)
    return dict(refresh=refresh_result,observations=outputs,http_queries=len(queries)*2)


def _signals(root):
    base.command([sys.executable,'scripts/prepare_capacity_signals.py','--root',str(root)],Path(root)/'prepare.log')
    from app.forward_service import freeze_today
    from datetime import datetime
    source=Path(root)/'capacity-signals'/str(datetime.now(p.TZ).date())/'signals.json'
    return dict(hash=freeze_today(Path(root)/'signals.sqlite3',source)['hash'])


def _checked(value):
    if isinstance(value,dict) and value.get('entry_block'):
        raise ValueError(value['entry_block'])
    return value


def status(root=policy.ROOT,clock=j.now):
    paths=books(root);rows={role:policy.verify(path,role) for role,path in paths.items()}
    summaries={role:p.summary(path,clock) for role,path in paths.items()}
    reports={}
    for role,path in paths.items():
        audit=corporate.inspect(path,clock=clock,rows=rows[role])
        plans=[r for r in rows[role] if r['kind']=='capacity_plan']
        reports[role]=dict(cash=summaries[role]['cash'],nav=summaries[role]['nav'],price_date=summaries[role]['price_date'],
            fill_count=summaries[role]['fill_count'],holdings=summaries[role]['holdings'],orders=summaries[role]['orders'],
            source_blocked=audit['blocked'],issues=audit['issues'],risk=policy.risk(rows[role]) if role!='benchmark' else None,
            plan=plans[-1]['body'] if plans else None,head=rows[role][-1]['hash'])
    required=['observe','expire_strategy','expire_control','expire_benchmark','pipeline','signals',
              'close_strategy','close_control','close_benchmark','plan_strategy','plan_control','plan_benchmark']
    daily={}
    for r in base._runs(root):
        if j.timestamp(r['recorded_at'])>clock(): continue
        b=r['body'];daily.setdefault(b['date'],{})[b['stage']]=b
    days=[]
    for day,stages in sorted(daily.items()):
        missing=[]
        for stage in required:
            value=stages.get(stage,{})
            if value.get('status')!='ok': missing.append(stage);continue
            if stage=='observe':
                results=value.get('detail',{}).get('observations',[])
                if not all(any(x.get('role')==role and x.get('status')=='collect' for x in results) for role in policy.ROLES):
                    missing.append(stage)
        days.append(dict(date=day,completed=not missing,missing=missing))
    return dict(version=policy.RULES,observed_at=clock().isoformat(),books=reports,days=days,
        completed_days=sum(d['completed'] for d in days),live_qualified=False,broker_execution_verified=False,
        classification=sim.RULES['classification'],
        comparison={role:halts.compare(paths[role],paths['benchmark']) for role in ('strategy','control')},
        remaining=['完整歷史母體及獨立還原價未通過資格','尚需未見期間及真實券商成交對帳',
                   '前向限價規則與歷史收盤成交假設不同，報酬不能直接沿用',
                   '20%整戶回撤暫定值僅供模擬，實盤水位須使用者選定'])


def export(root=policy.ROOT,clock=j.now):
    result=status(root,clock)
    target=Path(root)/'latest-report.json';temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(result,ensure_ascii=False,indent=2));temp.replace(target)
    return result


def run(root=policy.ROOT,observe_seconds=20,clock=j.now,sleeper=walltime.sleep,fetcher=market.fetch):
    if type(observe_seconds) is not int or not 15<=observe_seconds<=30: raise ValueError('觀察間隔須15至30秒')
    root=Path(root)
    with file_lock(root/'.run.lock',timeout=0):
        paths=books(root);now=clock().astimezone(p.TZ)
        if not base.calendar_day(now.date()): return dict(status='closed_market',stages=[])
        stages=[]
        if time(9)<=now.time()<time(13,30):
            stages.append(base._stage(root,'observe',lambda:observe(paths,clock,sleeper,observe_seconds,fetcher),
                                      clock,retry_seconds=60,once=False))
        elif time(13,30)<=now.time()<time(20):
            for role,path in paths.items():
                stages.append(base._stage(root,'expire_'+role,lambda path=path:base._cancel_day(path,clock),clock))
            if now.hour>=18:
                # Explicit source selection; never silently switches an official-source failure.
                pipeline=base._stage(root,'pipeline',lambda:base.command(
                    ['env','INGEST_PRICES_SOURCE=finmind','make','pipeline'],root/'pipeline.log'),clock)
                stages.append(pipeline)
                if pipeline['status'] not in ('ok','already_done'): return dict(status='needs_attention',stages=stages)
                stages.append(base._stage(root,'signals',lambda:_signals(root),clock))
                refresh(paths,clock)
                # New-candidate failure must not prevent independently verifiable exits.
                for role,path in paths.items():
                    settled=base._stage(root,'close_'+role,lambda path=path:base.close(path,clock=clock),clock)
                    stages.append(settled)
                    if settled['status'] in ('ok','already_done'):
                        stages.append(base._stage(root,'plan_'+role,
                            lambda path=path:_checked(policy.save_plans(path,root/'signals.sqlite3',clock)),clock))
        if stages: export(root,clock)
        blocked=any(s['status'] in ('blocked','cooldown') or any(x.get('status')=='blocked'
            for x in (s.get('detail') or {}).get('observations',[]) if isinstance(x,dict))
            for s in stages if isinstance(s.get('detail') or {},dict))
        return dict(status='needs_attention' if blocked else 'ok' if stages else 'outside_window',stages=stages)
