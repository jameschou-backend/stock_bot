"""Cash-only scheduling; explicitly reuses sealed data, quote and accounting stages."""
from datetime import time
from pathlib import Path
import json
import time as walltime
from app import forward_cash_policy as cash, forward_automation as base
from app.forward_automation import calendar_day, _stage, _intraday, _cancel_day, _signals, _runs, command, close
from app import forward_journal as j, forward_portfolio as p, forward_market_quotes as market, forward_corporate_audit as corporate
from app.file_lock import file_lock


def export(root=cash.ROOT,clock=j.now):
    root=Path(root);books={}
    if any(not (root/(role+'.sqlite3')).is_file() for role in ('strategy','benchmark')):raise ValueError('尚未初始化模擬帳本')
    for role in ('strategy','benchmark'):
        path=root/(role+'.sqlite3')
        cash.verify(path,role)
        s=p.summary(path,clock);s['rows']=[r for r in s['rows'] if r['kind'] in ('fill','close','simulation_review')];books[role]=s
    comparison=base.halts.compare(root/'strategy.sqlite3',root/'benchmark.sqlite3')
    if comparison['ready']:comparison['note']='固定規則推定的模擬成交，非使用者實際回報或券商成交'
    result=dict(allocation_policy=cash.RULES, classification=base.sim.RULES['classification'],live_qualified=False,observed_at=clock().isoformat(),books=books,
        comparison=comparison,recent_runs=_runs(root)[-30:])
    day=str(clock().astimezone(p.TZ).date())
    payload=json.dumps(result,ensure_ascii=False,indent=2)
    # The journals retain every observation. Archive one complete closing report,
    # rather than multiplying full cumulative ledgers on every intraday wakeup.
    if all(b['price_date']==day and b['nav'] is not None for b in books.values()):
        target=root/'reports'/day;target.mkdir(parents=True,exist_ok=True)
        closes=[next(r['hash'] for r in reversed(b['rows']) if r['kind']=='close') for b in books.values()]
        file=target/(j.digest(closes)+'.json')
        if not file.exists():file.write_text(payload)
    temp=root/'latest-report.tmp';temp.write_text(payload);temp.replace(root/'latest-report.json')
    return result


def run(root=cash.ROOT,observe_seconds=20,clock=j.now,sleeper=walltime.sleep,fetcher=market.fetch):
    if not 15<=observe_seconds<=30:raise ValueError('兩次觀察等待須15至30秒')
    root=Path(root)
    if not (root/'strategy.sqlite3').exists() or not (root/'benchmark.sqlite3').exists():raise ValueError('請先初始化獨立模擬帳本')
    with file_lock(root/'.run.lock',timeout=0):
        books={role:root/(role+'.sqlite3') for role in ('strategy','benchmark')}
        for role,path in books.items():
            if not path.exists():raise ValueError('請先初始化獨立模擬帳本')
            cash.verify(path,role)
        now=clock().astimezone(p.TZ);day=now.date();result=[]
        if not calendar_day(day):return dict(status='closed_market',date=str(day),stages=[])
        if time(8,45)<=now.time()<time(9):
            result.append(_stage(root,'preopen',lambda:dict(books={role:p.summary(path,clock)['orders'] for role,path in books.items()}),clock))
        elif time(9)<=now.time()<time(13,30):
            result.append(_stage(root,'observe',lambda:_intraday(root,books,clock,sleeper,observe_seconds,fetcher),clock,retry_seconds=60,once=False))
        elif time(13,30)<=now.time()<time(20):
            for role,path in books.items():result.append(_stage(root,'expire_'+role,lambda path=path:_cancel_day(path,clock),clock))
            if now.hour>=18:
                def pipeline():command(['make','pipeline'],root/'pipeline.log');return dict(log=str(root/'pipeline.log'))
                update=_stage(root,'pipeline',pipeline,clock);result.append(update)
                if update['status'] not in ('ok','already_done'):return dict(status='needs_attention',stages=result)
                result.append(_stage(root,'signals',lambda:_signals(root),clock))
                for role,path in books.items():
                    def settle(path=path):
                        corporate.refresh(path,request_budget=12)
                        return dict(hash=close(path,clock=clock)['hash'])
                    settled=_stage(root,'close_'+role,settle,clock);result.append(settled)
                    if settled['status'] in ('ok','already_done'):
                        result.append(_stage(root,'plan_'+role,lambda path=path,role=role:cash.save_plans(path,root/'signals.sqlite3',benchmark=role=='benchmark',clock=clock),clock))
        attention=any(x['status']=='blocked' or (isinstance(x.get('detail'),dict) and bool(x['detail'].get('entry_block')))  or (isinstance(x.get('detail'),list) and any(y.get('status')=='blocked' for y in x['detail'])) for x in result)
        if result:export(root,clock)
        return dict(status='needs_attention' if attention else 'ok' if result else 'outside_window',date=str(day),stages=result)
