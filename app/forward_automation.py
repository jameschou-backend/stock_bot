"""Bounded scheduled workflow. Missing evidence pauses a stage, never fabricates it."""
from datetime import datetime,time
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time as walltime
from app import forward_journal as j,forward_portfolio as p,forward_simulation as sim,forward_market_quotes as market,forward_corporate_audit as corporate,forward_halts as halts
from app.file_lock import file_lock


def calendar_day(day):
    from sqlalchemy import select
    from app.db import get_session
    from app.models import TradingCalendar
    with get_session() as db:value=db.scalar(select(TradingCalendar.is_open).where(TradingCalendar.trading_date==day))
    if value is None:raise ValueError('缺少當日官方交易日曆，不能以星期推測')
    return bool(value)


def markets(ids):
    from sqlalchemy import select
    from app.db import get_session
    from app.models import Stock
    with get_session() as db:rows=db.execute(select(Stock.stock_id,Stock.market,Stock.security_type).where(Stock.stock_id.in_(ids))).all()
    out={}
    for sid,exchange,kind in rows:
        if exchange not in ('TWSE','TPEX') or (kind!='stock' and sid!='0050'):raise ValueError('未支援的市場或交易單位：'+sid)
        out[sid]='tse' if exchange=='TWSE' else 'otc'
    if set(out)!=set(ids):raise ValueError('股票市場主檔缺漏')
    return out


def approval_signature(rows,evidence_path,clock):
    scope=corporate.scope(rows,clock);latest=corporate.latest(evidence_path)
    content=[]
    for query in corporate.specs(scope):
        e=latest.get(query)
        if not e or not corporate.fresh(e,clock) or e['body']['status']!='ok':raise ValueError('公司行動來源需先更新')
        content.append(dict(query=list(query),rows=e['body']['rows']))
    return j.digest(content)


def approve(path,reviewer,note,evidence_path=corporate.PATH,clock=j.now):
    if not reviewer.strip() or len(note.strip())<10:raise ValueError('需填核對人與至少10字核對說明')
    with j.connection(path) as con:
        rows=sim.verify(con);audit=corporate.inspect(path,evidence_path,clock,rows=rows)
        if audit['blocked']:raise ValueError('仍有公司行動衝突，不能以確認取代修復')
        body=dict(date=str(clock().astimezone(p.TZ).date()),reviewer=reviewer,note=note,source_fingerprint=approval_signature(rows,evidence_path,clock))
        return j.append(con,'simulation_review:'+j.digest(body),'simulation_review',body,clock)


def close(path,evidence_path=corporate.PATH,clock=j.now):
    rows=sim.read(path);s=p.state(rows);day=str(clock().astimezone(p.TZ).date())
    approved=not s['holdings']
    if s['holdings']:
        signature=approval_signature(rows,evidence_path,clock)
        approved=any(r['kind']=='simulation_review' and r['body']['date']==day and r['body']['source_fingerprint']==signature for r in rows)
    if not approved:raise ValueError('待人工核對今日公司行動；排程不可代勾已核對')
    return corporate.capture_close(path,actions_reviewed=approved,clock=clock,evidence_path=evidence_path)


def _log(root,stage,status,detail,clock):
    body=dict(date=str(clock().astimezone(p.TZ).date()),stage=stage,status=status,detail=detail)
    with j.connection(Path(root)/'runs.sqlite3') as con:return j.append(con,'run:'+j.digest(body)+':'+clock().isoformat(),'run',body,clock)


def _runs(root):
    path=Path(root)/'runs.sqlite3'
    if not path.exists():return []
    with j.connection(path) as con:return j.read_events(con)


def command(args,log_path,timeout=600):
    # Explicitly avoid unused per-stock Sponsor ingestion in the scheduled path.
    env=dict(os.environ,SPONSOR_INGEST='off')
    with Path(log_path).open('a') as output:
        proc=subprocess.Popen(args,cwd=j.ROOT,stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise ValueError('工作逾時，已停止整個子程序群；詳見本機日誌')
    if code:raise ValueError('工作失敗，結束碼 '+str(code)+'；詳見 '+str(log_path))


def _stage(root,name,fn,clock,retry_seconds=900,once=True):
    day=str(clock().astimezone(p.TZ).date());matches=[r for r in _runs(root) if r['body']['date']==day and r['body']['stage']==name]
    if once and any(r['body']['status']=='ok' for r in matches):return dict(stage=name,status='already_done')
    if matches and (clock()-j.timestamp(matches[-1]['recorded_at'])).total_seconds()<retry_seconds:return dict(stage=name,status='cooldown')
    _log(root,name,'running',{},clock)
    try:
        detail=fn();_log(root,name,'ok',detail,clock);return dict(stage=name,status='ok',detail=detail)
    except Exception as exc:
        # Error text may come from a provider; keep actionable class and bounded message only.
        from app.finmind import FinMindQuotaError
        message='FinMind 限流，保留進度等待下一次排程' if isinstance(exc,FinMindQuotaError) else str(exc)[:1000]
        _log(root,name,'blocked',dict(error=type(exc).__name__,message=message),clock)
        return dict(stage=name,status='blocked',message=message)


def _cancel_day(path,clock):
    day=str(clock().astimezone(p.TZ).date())
    with j.connection(path) as con:
        rows=sim.verify(con);s=p.state(rows);n=0
        for o in s['orders'].values():
            if not o['closed'] and o['filled']<o['qty'] and o['session']<=day:
                p.submit(con,dict(kind='cancel',id='sim-expire:'+o['order_id'],order_id=o['order_id'],reason='模擬觀察期間未達撮合條件；未觀察區間不補造成交'),clock);n+=1
        return dict(cancelled=n)


def _signals(root):
    command([sys.executable,'scripts/prepare_rolling_forward.py','--root',str(root)],Path(root)/'prepare.log')
    from app.forward_service import freeze_today
    return dict(hash=freeze_today(Path(root)/'signals.sqlite3',Path(root)/'signals'/str(datetime.now(p.TZ).date())/'signals.json')['hash'])


def _intraday(root,books,clock,sleeper,observe_seconds,fetcher):
    outputs=[];requests=[]
    for role,path in books.items():
        report=corporate.refresh(path,request_budget=12);audit=corporate.inspect(path)
        if audit['blocked']:outputs.append(dict(role=role,status='blocked',reason='公司行動資料缺漏或有待核對事件'));continue
        with j.connection(path) as con:rows=sim.verify(con)
        day=str(clock().astimezone(p.TZ).date());s=p.state(rows)
        current=[o for o in s['orders'].values() if o['session']==day and not o['closed'] and o['filled']<o['qty']]
        ids=sorted({o['stock_id'] for o in current});lookup=markets(ids)
        for o in current:
            key=(lookup[o['stock_id']],o['stock_id'],o['channel'])
            if key not in requests:requests.append(key)
        outputs.append(dict(role=role,status='collect',path=str(path)))
    if len(requests)>12:raise ValueError('本輪行情種類超過12組，停止避免非預期抓取')
    for iteration in range(2):
        for key in requests:
            observation=fetcher(*key)
            for entry in outputs:
                if entry['status']=='collect':
                    result=sim.match(Path(entry['path']),observation,clock)
                    entry.setdefault('observations',[]).append(dict(query=list(key),**{k:v for k,v in result.items() if k!='quote'}))
        if iteration==0 and requests:sleeper(observe_seconds)
    for role,path in books.items():
        try:
            halts.require_observed(path)
            from app.forward_portfolio_service import release_funded_intents
            release_funded_intents(path,clock)
        except ValueError as exc:outputs.append(dict(role=role,status='funding_wait',reason=str(exc)))
    return outputs


def run(root=sim.ROOT,observe_seconds=20,clock=j.now,sleeper=walltime.sleep,fetcher=market.fetch):
    if not 15<=observe_seconds<=30:raise ValueError('兩次觀察等待須15至30秒')
    root=Path(root)
    if not (root/'strategy.sqlite3').exists() or not (root/'benchmark.sqlite3').exists():raise ValueError('請先初始化獨立模擬帳本')
    with file_lock(root/'.run.lock',timeout=0):
        books={role:root/(role+'.sqlite3') for role in ('strategy','benchmark')}
        for path in books.values():
            if not path.exists():raise ValueError('請先初始化獨立模擬帳本')
            with j.connection(path) as con:sim.verify(con)
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
                        result.append(_stage(root,'plan_'+role,lambda path=path,role=role:halts.save_plans(path,root/'signals.sqlite3',benchmark=role=='benchmark',clock=clock),clock))
        attention=any(x['status']=='blocked' or (isinstance(x.get('detail'),list) and any(y.get('status')=='blocked' for y in x['detail'])) for x in result)
        if result:export(root,clock)
        return dict(status='needs_attention' if attention else 'ok' if result else 'outside_window',date=str(day),stages=result)


def export(root=sim.ROOT,clock=j.now):
    root=Path(root);books={}
    if any(not (root/(role+'.sqlite3')).is_file() for role in ('strategy','benchmark')):raise ValueError('尚未初始化模擬帳本')
    for role in ('strategy','benchmark'):
        path=root/(role+'.sqlite3')
        with j.connection(path) as con:sim.verify(con)
        s=p.summary(path,clock);s['rows']=[r for r in s['rows'] if r['kind'] in ('fill','close','simulation_review')];books[role]=s
    comparison=halts.compare(root/'strategy.sqlite3',root/'benchmark.sqlite3')
    if comparison['ready']:comparison['note']='固定規則推定的模擬成交，非使用者實際回報或券商成交'
    result=dict(classification=sim.RULES['classification'],live_qualified=False,observed_at=clock().isoformat(),books=books,
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
