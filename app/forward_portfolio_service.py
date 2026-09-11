"""DB-only portfolio proposals. No API request, historical replay, or broker call."""
from datetime import datetime
from decimal import Decimal
from sqlalchemy import select
from app import forward_journal as j, forward_portfolio as p


def capture_close(path=p.PATH, actions_reviewed=False, clock=j.now):
    from app.db import get_session
    from app.models import RawPrice, TradingCalendar
    from app.workbench_service import data_status
    status=data_status()
    if not status['data_ready']: raise ValueError('市場資料尚未完整：'+'；'.join(status['problems']))
    today=str(clock().astimezone(p.TZ).date())
    if status['price_date']!=today: raise ValueError('資料非今日，不能補寫前向結算')
    with j.connection(path) as con:
        p.initialize(con,clock)
        rows=j.read_events(con)
        existing=next((r for r in rows if r['kind']=='close' and r['body']['date']==today),None)
        if existing: return existing  # First observation is immutable, even if status/quota changes.
        state=p.state(rows)
        if state['holdings'] and not actions_reviewed: raise ValueError('持股需先核對公司行動，再封存收盤')
        ids=list(state['holdings'])+['0050']
        with get_session() as db:
            sessions=[str(d) for d in db.scalars(select(TradingCalendar.trading_date).where(
                TradingCalendar.is_open.is_(True)).order_by(TradingCalendar.trading_date))]
            prices={str(sid):str(price) for sid,price in db.execute(select(RawPrice.stock_id,RawPrice.close).where(
                RawPrice.stock_id.in_(ids),RawPrice.trading_date==today)) if price is not None and price>0}
        p.submit(con,dict(kind='calendar',id=today,sessions=sessions,source='DB/trading_calendar'),clock)
        body=dict(date=today,prices=prices,actions_reviewed=bool(actions_reviewed or not state['holdings']),
                  source='DB/raw_prices; '+j.digest(status))
        nav=p.valuation({**state,'mark':{'body':body}})
        body['nav']=str(nav) if nav is not None else None
        return p.submit(con,dict(kind='close',id=today,**body),clock)


def source_signal(path=j.DEFAULT_PATH, clock=j.now):
    with j.connection(path) as con:
        j.initialize(con,clock)  # Verify the sealed v1 code, not just the JSON hash chain.
        rows=j.read_events(con)
    today=str(clock().astimezone(p.TZ).date())
    return next((r for r in reversed(rows) if r['kind']=='signal' and r['body']['signal_date']==today),None)


def affordable(price,budget,step=1):
    lo,hi=0,int(max(Decimal(0),budget)/price)//step
    while lo<hi:
        middle=(lo+hi+1)//2
        value=price*middle*step
        if value+p.fee(value)<=budget: lo=middle
        else: hi=middle-1
    return lo*step


def proposals(path=p.PATH, signal_path=j.DEFAULT_PATH, clock=j.now):
    """Pure proposals, not stored orders. Candidate funding precedes idle ETF buys."""
    with j.connection(path) as con: rows=j.read_events(con)
    s=p.state(rows); mark=s['last_close']
    if not mark or mark['body']['date']!=str(clock().astimezone(p.TZ).date()) or mark['body']['nav'] is None:
        raise ValueError('請先完成今日收盤淨值與公司行動核對')
    day=mark['body']['date'];session=p.next_session(s,day)
    signal=source_signal(signal_path,clock)
    if signal is None: raise ValueError('今日原策略訊號尚未封存；不能把缺訊號當成買0050')
    nav=Decimal(mark['body']['nav']);reserved,sold=p.reserves(s,day)
    cash=s['cash']-reserved; out=[]
    # Preserve exit intent after a partial/unfilled exit, even if price recovers.
    for sid,pos in s['holdings'].items():
        if sid=='0050': continue
        close=Decimal(mark['body']['prices'][sid])
        held=sum(pos['entry_date']<=d<=day for d in s['calendar'])
        reason='stop12_close' if close<=pos['stop_basis']*Decimal('.88') else 'holding63' if held>=63 else None
        decision=s['decisions'].get(sid)
        if not reason and not decision: continue
        out.append(dict(kind='exit_proposal',stock_id=sid,session=session,qty=pos['qty']-sold.get(sid,0),
                        limit_price=str(close),reason=reason,prior_decision=decision['hash'] if decision else None))
    active={o['stock_id'] for o in s['orders'].values() if not o['closed'] and o['filled']<o['qty'] and o['stock_id']!='0050'}
    occupied={sid for sid in s['holdings'] if sid!='0050'}|active
    candidates=sorted(signal['body']['candidates'],key=lambda e:(-e['priority'],e['members'][0]))
    needs_funding=False
    for candidate in candidates:
        sid=candidate['members'][0]
        if sid in occupied or len(occupied)>=3: continue
        price=p.num(candidate['planning_reference_close'],True)
        liq=candidate['liquidity_before_entry']
        if not liq['complete_20_sessions'] or liq['mean_turnover20_twd']<50000000: continue
        budget=nav/3
        total=affordable(price,budget-Decimal(40))
        board=min(total//1000*1000,int(liq['adv20_shares']*.01)//1000*1000)
        odd=total%1000
        for channel,n in [('board',board),('odd',odd)]:
            if not n: continue
            value=price*n+p.fee(price*n)
            funded=value<=cash
            kind='order' if funded else 'funding_intent'
            c=dict(kind=kind,id=f'{session}:{sid}:{channel}',order_id=f'{session}:{sid}:{channel}',
                stock_id=sid,side='buy',channel=channel,qty=n,limit_price=str(price),session=session,
                reason='original_frozen_candidate',signal=candidate,signal_proof=signal,source_hash=signal['hash'])
            out.append(c)
            if funded: cash-=value
            else: needs_funding=True
        occupied.add(sid)
    if needs_funding:
        etf=s['holdings'].get('0050',{}).get('qty',0)-sold.get('0050',0)
        if etf:
            price=mark['body']['prices']['0050']
            for channel,n in [('board',etf//1000*1000),('odd',etf%1000)]:
                if n: out.append(dict(kind='order',id=f'{session}:fund:0050:{channel}',
                    order_id=f'{session}:fund:0050:{channel}',stock_id='0050',side='sell',channel=channel,
                    qty=n,limit_price=price,session=session,reason='fund_stock_entries'))
        out.append(dict(kind='funding_gap',stock_id='0050',available_shares=etf,
                        note='個股買單暫不預留不存在的現金；須先登錄0050賣出，才可釋放盤前資金需求。'))
    elif not any(c['kind']=='exit_proposal' for c in out) and cash>=5000 and not any(o['stock_id']=='0050' and o['side']=='buy' and o['session']==session for o in s['orders'].values()):
        price=p.num(mark['body']['prices']['0050'],True)
        total=affordable(price,cash-Decimal(40))
        for channel,n in [('board',total//1000*1000),('odd',total%1000)]:
            if n: out.append(dict(kind='order',id=f'{session}:0050:{channel}',order_id=f'{session}:0050:{channel}',
                stock_id='0050',side='buy',channel=channel,qty=n,limit_price=str(price),session=session,reason='idle_cash'))
    return out


def save_proposals(path=p.PATH, signal_path=j.DEFAULT_PATH, clock=j.now):
    plans=proposals(path,signal_path,clock)
    with j.connection(path) as con:
        saved=[]
        for c in plans:
            if c['kind'] in ('order','funding_intent'): saved.append(p.submit(con,c,clock))
            elif c['kind']=='exit_proposal' and c['qty']>0:
                day=str(clock().astimezone(p.TZ).date())
                reason=c['prior_decision']
                if not reason:
                    event=p.submit(con,dict(kind='decision',id=day+':'+c['stock_id'],
                        date=day,stock_id=c['stock_id'],reason=c['reason']),clock)
                    reason=event['hash']
                for channel,n in [('board',c['qty']//1000*1000),('odd',c['qty']%1000)]:
                    if n: saved.append(p.submit(con,dict(kind='order',id=f"exit:{c['session']}:{c['stock_id']}:{channel}",
                        order_id=f"exit:{c['session']}:{c['stock_id']}:{channel}",stock_id=c['stock_id'],side='sell',
                        qty=n,channel=channel,session=c['session'],limit_price=c['limit_price'],reason=reason),clock))
        return saved


def release_funded_intents(path=p.PATH, clock=j.now):
    """Explicit intraday release; frozen quantity/price never enlarged."""
    today=str(clock().astimezone(p.TZ).date())
    with j.connection(path) as con:
        rows=j.read_events(con); saved=[]
        for r in rows:
            if r['kind']!='funding_intent' or r['body']['session']!=today: continue
            c=dict(r['body'],kind='order',id=r['body']['order_id'],funding_intent=r['hash'])
            saved.append(p.submit(con,c,clock))
        return saved
