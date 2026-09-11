"""Additive prospective execution protocol; never rewrites the sealed v2 engine.

Both accounts use identical Decimal accounting. The benchmark purchases 0050
once, then reinvests only delivered cash dividends. No inferred fills.
"""
from datetime import datetime, time
from decimal import Decimal as D
import hashlib
from pathlib import Path
from app import forward_journal as j, forward_portfolio as p, forward_portfolio_service as legacy

BENCHMARK = j.ROOT / '.cache/forward-validation/benchmark-0050.sqlite3'
PROTOCOL = dict(version='comparison-and-exits-v1', initial_cash='1000000',
    benchmark='0050_buy_hold_reinvest_delivered_dividends',
    entry='same_initial_strategy_session_and_frozen_reference_price',
    unfilled_initial='cash_no_next_day_retry', fills='user_supplied_paper_reports_only',
    exits='persist_before_loading_new_signals', live_qualified=False)


def protocol(con, role, clock=j.now):
    p.initialize(con, clock)
    body=dict(protocol=PROTOCOL, role=role,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return j.append(con,'execution_protocol','execution_protocol',body,clock)


def seed_benchmark(strategy=p.PATH, benchmark=BENCHMARK, clock=j.now):
    """Adopt an already observed close before the first entry; never backdate it."""
    if Path(strategy).resolve()==Path(benchmark).resolve(): raise ValueError('比較帳本必須分開')
    with j.connection(strategy) as con:
        p.initialize(con,clock)
        original=j.read_events(con)
    seed=next((r for r in original if r['kind']=='close'),None)
    orders=[r for r in original if r['kind']=='order' and r['body']['side']=='buy']
    if not seed or not orders: raise ValueError('策略尚未封存初始收盤與進場計畫')
    session=min(r['body']['session'] for r in orders)
    source=dict(strategy_initial_close=seed['hash'], initial_date=seed['body']['date'],
                entry_session=session, initial_cash=p.RULES['initial_cash'])
    with j.connection(benchmark) as con:
        rows=j.read_events(con)
        if rows:
            protocol(con,'benchmark',clock)
            existing=next((r for r in rows if r['kind']=='comparison_anchor'),None)
            if not existing or existing['body']!=source: raise ValueError('比較起點不符，不能更換基準')
            return existing
        opening=datetime.fromisoformat(session).replace(hour=9,tzinfo=p.TZ)
        if clock()>=opening or j.timestamp(seed['recorded_at'])>clock():
            raise ValueError('必須在共同進場開盤前建立比較帳本，不可事後回填')
        if any(r['kind'] in ('fill','entitlement','delivery') for r in original):
            raise ValueError('策略已發生成交或公司行動，不能倒建共同起點')
        if seed['body']['nav']!=p.RULES['initial_cash'] or not seed['body']['actions_reviewed']:
            raise ValueError('初始本金或核對狀態不符')
        protocol(con,'benchmark',clock)
        anchor=j.append(con,'comparison_anchor','comparison_anchor',source,clock)
        s=p.state(original)
        if p.next_session(s,seed['body']['date'])!=session: raise ValueError('共同進場日期不符')
        p.submit(con,dict(kind='calendar',id='anchor',sessions=s['calendar'],source='verified_strategy_calendar'),clock)
        # This is explicitly inherited evidence, recorded NOW; it is not a new daily observation.
        price=seed['body']['prices']['0050']
        body=dict(date=seed['body']['date'], prices={'0050':price}, actions_reviewed=True,
            nav=p.RULES['initial_cash'],source='inherited_close:'+seed['hash'],
            source_recorded_at=seed['recorded_at'])
        j.append(con,'inherited_close','close',body,clock)
        total=legacy.affordable(p.num(price,True),D(p.RULES['initial_cash'])-40)
        for channel,n in [('board',total//1000*1000),('odd',total%1000)]:
            if n:
                p.submit(con,dict(kind='order',id='initial:'+channel,order_id='initial:'+channel,
                    stock_id='0050',side='buy',channel=channel,qty=n,limit_price=price,session=session,
                    reason='idle_cash',purpose='benchmark_initial'),clock)
        return anchor


def save_exits(path=p.PATH, clock=j.now):
    """Exit decisions commit independently of new-signal or entry-planning failures."""
    with j.connection(path) as con:
        protocol(con,'strategy',clock)
        s=p.state(j.read_events(con)); mark=s['mark']
        if not s['holdings']: return []
        day=str(clock().astimezone(p.TZ).date())
        if not mark or mark['body']['date']!=day or p.valuation(s) is None:
            raise ValueError('出場需要今日持倉價格與公司行動核對，不能以缺資料判定繼續持有')
        session=p.next_session(s,day); _,reserved=p.reserves(s,day)
        saved=[]
        for sid,pos in s['holdings'].items():
            if sid=='0050': continue
            n=pos['qty']-reserved.get(sid,0)
            if n<=0: continue
            price=D(mark['body']['prices'][sid])
            held=sum(pos['entry_date']<=d<=day for d in s['calendar'])
            reason='stop12_close' if price<=pos['stop_basis']*D('.88') else 'holding63' if held>=63 else None
            decision=s['decisions'].get(sid)
            if not reason and not decision: continue
            if not decision:
                decision=p.submit(con,dict(kind='decision',id=day+':'+sid,stock_id=sid,date=day,reason=reason),clock)
            for channel,size in [('board',n//1000*1000),('odd',n%1000)]:
                if size:
                    key=f'exit:{session}:{sid}:{channel}'
                    saved.append(p.submit(con,dict(kind='order',id=key,order_id=key,stock_id=sid,side='sell',
                        channel=channel,qty=size,limit_price=str(price),session=session,reason=decision['hash']),clock))
        return saved


def save_strategy_plans(path=p.PATH, signal_path=j.DEFAULT_PATH, clock=j.now):
    exits=save_exits(path,clock)
    try:
        new=legacy.save_proposals(path,signal_path,clock)
        return dict(exits=exits,new_plans=new,entry_block=None)
    except (ValueError,KeyError,OSError) as exc:
        # Record the gap, preserving already committed exits. No failed buy becomes a no-signal day.
        reason=f'{type(exc).__name__}: {exc}'
        with j.connection(path) as con:
            body=dict(date=str(clock().astimezone(p.TZ).date()),reason=reason)
            j.append(con,'entry_gap:'+j.digest(body),'entry_gap',body,clock)
        return dict(exits=exits,new_plans=[],entry_block=reason)


def record_benchmark(command, path=BENCHMARK, clock=j.now):
    if command['kind'] not in ('fill','cancel','entitlement','delivery'):
        raise ValueError('基準只接受成交、取消或公司行動，不接受任意選股與賣出')
    with j.connection(path) as con:
        if not any(r['kind']=='comparison_anchor' for r in j.read_events(con)):
            raise ValueError('尚未建立共同起點')
        protocol(con,'benchmark',clock)
        if command['kind']=='entitlement' and command['stock_id']!='0050': raise ValueError('基準只能持有0050')
        return p.submit(con,command,clock)


def capture_benchmark_close(path=BENCHMARK, actions_reviewed=False, clock=j.now):
    with j.connection(path) as con:
        if not any(r['kind']=='comparison_anchor' for r in j.read_events(con)): raise ValueError('基準尚未初始化')
        protocol(con,'benchmark',clock)
    return legacy.capture_close(path,actions_reviewed,clock)


def reinvest_dividends(path=BENCHMARK, clock=j.now):
    with j.connection(path) as con:
        protocol(con,'benchmark',clock)
        rows=j.read_events(con); s=p.state(rows); day=str(clock().astimezone(p.TZ).date())
        if not s['mark'] or s['mark']['body']['date']!=day or p.valuation(s) is None:
            raise ValueError('股息再投入前須完成今日結算')
        delivered=sum(D(r['amount']) for r in s['rights'].values() if r['paid'] and r['action_type']=='cash')
        spent=D(0);reserved=D(0)
        for r in rows:
            if r['kind']=='fill' and s['orders'][r['body']['order_id']].get('purpose')=='dividend_reinvestment':
                b=r['body'];spent+=D(b['price'])*b['qty']+D(b['fee'])+D(b['tax'])
        for o in s['orders'].values():
            left=o['qty']-o['filled']
            if o.get('purpose')=='dividend_reinvestment' and left and not o['closed']:
                value=D(o['limit_price'])*left;reserved+=value+p.fee(value)
        budget=delivered-spent-reserved
        price=p.num(s['mark']['body']['prices']['0050'],True)
        total=legacy.affordable(price,max(D(0),budget-40))
        session=p.next_session(s,day); saved=[]
        for channel,n in [('board',total//1000*1000),('odd',total%1000)]:
            if n:
                key='dividend:'+session+':'+channel
                if key in s['orders']: continue
                saved.append(p.submit(con,dict(kind='order',id=key,order_id=key,stock_id='0050',side='buy',channel=channel,
                    qty=n,limit_price=str(price),session=session,reason='idle_cash',purpose='dividend_reinvestment'),clock))
        return saved


def comparison(strategy=p.PATH, benchmark=BENCHMARK):
    reasons=[]; books={}; rows_by_role={}
    for role,path in [('strategy',strategy),('benchmark',benchmark)]:
        if not Path(path).exists():
            return dict(ready=False,reasons=['0050比較帳本尚未建立'],points=[],live_qualified=False)
        with j.connection(path) as con:
            p.initialize(con)
            rows=j.read_events(con)
        if role=='benchmark':
            expected=dict(protocol=PROTOCOL,role=role,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            if not any(r['kind']=='execution_protocol' and r['body']==expected for r in rows):
                raise ValueError('比較規則已變更，不可沿用舊基準')
        rows_by_role[role]=rows;books[role]=p.state(rows)
    anchor=next((r['body'] for r in rows_by_role['benchmark'] if r['kind']=='comparison_anchor'),None)
    first=next((r for r in rows_by_role['strategy'] if r['kind']=='close'),None)
    if not anchor or not first or anchor['strategy_initial_close']!=first['hash']:
        raise ValueError('兩帳本並非同一起點')
    for o in books['benchmark']['orders'].values():
        if o['stock_id']!='0050' or o['side']!='buy' or o.get('purpose') not in ('benchmark_initial','dividend_reinvestment'):
            raise ValueError('比較帳本含非基準規則的委託')
        if o.get('purpose')=='benchmark_initial' and o['session']!=anchor['entry_session']:
            raise ValueError('基準初始買進日期不符')
    dates=[]
    for role,s in books.items():
        mark=s['mark'];dates.append(mark['body']['date'] if mark else None)
        if mark is None or p.valuation(s) is None: reasons.append(role+' 尚未完成完整估值')
        if not any(r['kind']=='fill' for r in rows_by_role[role]): reasons.append(role+' 尚無成交回報')
    if dates[0]!=dates[1]: reasons.append('兩帳本最新估值日期不同')
    closes={role:{r['body']['date']:r for r in rows if r['kind']=='close'} for role,rows in rows_by_role.items()}
    latest=max(d for d in dates if d is not None) if any(dates) else anchor['initial_date']
    expected=[d for d in books['strategy']['calendar'] if anchor['entry_session']<=d<=latest]
    if expected!=[d for d in books['benchmark']['calendar'] if anchor['entry_session']<=d<=latest]:
        reasons.append('兩帳本交易日曆不一致')
    for role in books:
        if any(d not in closes[role] or closes[role][d]['body']['nav'] is None for d in expected):
            reasons.append(role+' 缺少共同期間的每日結算')
    points=[]
    for day in expected:
        if any(day not in closes[role] for role in books): continue
        a,b=closes['strategy'][day]['body'],closes['benchmark'][day]['body']
        if ('0050' not in a['prices'] or '0050' not in b['prices']
                or D(a['prices']['0050'])!=D(b['prices']['0050'])):
            reasons.append(day+' 0050估價不一致');continue
        if a['nav'] is not None and b['nav'] is not None:
            points.append(dict(date=day,strategy_nav=a['nav'],benchmark_nav=b['nav']))
    ready=not reasons and bool(points)
    result=dict(ready=ready,reasons=sorted(set(reasons)),points=points,live_qualified=False,
                entry_session=anchor['entry_session'],as_of=dates[0] if dates[0]==dates[1] else None)
    if ready:
        a,b=D(points[-1]['strategy_nav']),D(points[-1]['benchmark_nav']);initial=D(PROTOCOL['initial_cash'])
        result.update(strategy_return=float(a/initial-1),benchmark_return=float(b/initial-1),
                      excess_percentage_points=float((a-b)/initial*100),note='使用者提供的紙上成交與人工核對；不是實盤績效')
    return result
