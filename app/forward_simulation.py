"""Preregistered counterfactual fills; never writes the original evidence books."""
from datetime import datetime,time
from decimal import Decimal as D, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
from pathlib import Path
import shutil
import uuid
from app import forward_journal as j, forward_portfolio as p, forward_comparison as c, forward_halts as h
from app.file_lock import file_lock

ROOT=j.ROOT/'.cache/forward-simulation/v1'
CODE=('app/forward_simulation.py','app/forward_market_quotes.py','app/forward_automation.py','scripts/prepare_rolling_forward.py','app/forward_halts.py','app/forward_corporate_audit.py')
RULES=dict(version='forward-simulation-v1',classification='counterfactual_simulation_not_execution_evidence',
    initial_cash='1000000',min_delay_seconds=15,max_pair_gap_seconds=90,max_quote_age_seconds=15,
    visible_depth_fraction='0.10',interval_volume_fraction='0.10',adverse_slippage='0.001',
    commission='0.001425',minimum_fee='20',stock_sell_tax='0.003',etf0050_sell_tax='0.001',
    order_priority='sell_then_existing_order_sequence',comparison_capacity='independent_counterfactual_accounts',
    unobserved_intervals='no_fills_no_backfill',corporate_review='daily_human_attestation_required_for_close',live_qualified=False)


def read(path):
    if not Path(path).is_file():raise ValueError('來源帳本不存在')
    with j.connection(path) as con:p.initialize(con);return j.read_events(con)


def code_hashes():return {name:hashlib.sha256((j.ROOT/name).read_bytes()).hexdigest() for name in CODE}


def verify(con):
    p.initialize(con);rows=j.read_events(con)
    seal=next((r for r in rows if r['kind']=='simulation_protocol'),None)
    if not seal or seal['body']['rules']!=RULES or seal['body']['code_sha256']!=code_hashes():raise ValueError('模擬規則或程式已變更，禁止沿用舊模擬帳本；須另開版本')
    return rows


def initialize(root=ROOT,strategy=p.PATH,benchmark=c.BENCHMARK,clock=j.now,signals=j.DEFAULT_PATH):
    root=Path(root)
    with file_lock(root.parent/'.init.lock',timeout=0):
        if root.exists():
            for role in ('strategy','benchmark'):
                if not (root/(role+'.sqlite3')).is_file():raise ValueError('模擬目錄不完整，請核對初始化狀態')
                with j.connection(root/(role+'.sqlite3')) as con:verify(con)
            return root
        sources={role:read(path) for role,path in [('strategy',strategy),('benchmark',benchmark)]}
        anchor=next(r['body'] for r in sources['benchmark'] if r['kind']=='comparison_anchor')
        if clock()>=datetime.fromisoformat(anchor['entry_session']).replace(hour=9,tzinfo=p.TZ):raise ValueError('初始進場已開盤，不能事後建立模擬起點')
        first=next(r for r in sources['strategy'] if r['kind']=='close')
        if first['hash']!=anchor['strategy_initial_close']:raise ValueError('策略與基準起點不同')
        if any(r['kind'] in ('fill','entitlement','delivery','restatement_lineage','simulation_protocol') for rows in sources.values() for r in rows):raise ValueError('只能採用尚未成交的原始封存起點')
        temp=root.parent/('.sim-init-'+uuid.uuid4().hex);temp.mkdir(parents=True)
        try:
            for role,rows in sources.items():
                with j.connection(temp/(role+'.sqlite3')) as con:
                    for r in rows:
                        con.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)',(r['seq'],r['event_key'],r['kind'],r['recorded_at'],j.encode(r['body']),r['previous_hash'],r['hash']))
                    j.append(con,'simulation_protocol','simulation_protocol',dict(rules=RULES,code_sha256=code_hashes(),role=role,
                        inherited_source=str(Path(strategy if role=='strategy' else benchmark).resolve()),inherited_head=rows[-1]['hash'],
                        note='Earlier records are inherited evidence; all new fills are explicitly simulated.'),clock)
            with j.connection(signals) as source:
                j.initialize(source,clock);signal_rows=j.read_events(source)
            with j.connection(temp/'signals.sqlite3') as dest:
                for r in signal_rows:dest.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)',(r['seq'],r['event_key'],r['kind'],r['recorded_at'],j.encode(r['body']),r['previous_hash'],r['hash']))
            temp.rename(root)
        finally:
            if temp.exists():shutil.rmtree(temp)
    return root


def _tick(price,sid):
    # Only ordinary four-digit equities and the fixed 0050 ETF are supported.
    if sid=='0050':return D('.01') if price<50 else D('.05')
    return D('.01') if price<10 else D('.05') if price<50 else D('.1') if price<100 else D('.5') if price<500 else D('1') if price<1000 else D('5')


def proposal(order,previous,current,now):
    reasons=[];at=j.timestamp(current['quote_at']);before=j.timestamp(previous['quote_at']) if previous else None
    day=str(now.astimezone(p.TZ).date());start=time(9,10 if order['channel']=='odd' else 0)
    if current['stock_id']!=order['stock_id'] or current['channel']!=order['channel'] or current['quantity_unit']!='shares':raise ValueError('撮合股票或交易單位不符')
    if day!=order['session'] or str(at.astimezone(p.TZ).date())!=day:reasons.append('報價非當日委託')
    if not start<=now.astimezone(p.TZ).time()<time(13,30) or not start<=at.astimezone(p.TZ).time()<time(13,30):reasons.append('不在連續觀察時段；不推定開收盤撮合')
    if not 0<=(now-at).total_seconds()<=RULES['max_quote_age_seconds']:reasons.append('行情過期或在未來')
    if j.timestamp(current['retrieved_at'])>now:reasons.append('行情尚未取得')
    if (at-j.timestamp(order['recorded_at'])).total_seconds()<RULES['min_delay_seconds']:reasons.append('委託延遲不足')
    if not previous:reasons.append('需要兩次獨立觀察')
    elif any(previous[k]!=current[k] for k in ('stock_id','market','channel','quantity_unit')):raise ValueError('兩次行情不屬於同一市場／股票')
    elif not RULES['min_delay_seconds']<=(at-before).total_seconds()<=RULES['max_pair_gap_seconds']:reasons.append('觀察間隔不足或中斷，不能補推成交')
    elif j.timestamp(previous['retrieved_at'])>at:reasons.append('第一筆行情取得順序異常')
    elif not 0<=(j.timestamp(previous['retrieved_at'])-before).total_seconds()<=RULES['max_quote_age_seconds']:reasons.append('第一筆觀察取得時已過期')
    if reasons:return dict(qty=0,reasons=reasons)
    side='asks' if order['side']=='buy' else 'bids';limit=D(order['limit_price'])
    ladders=[]
    for quote in (previous,current):
        levels=[x for x in quote[side] if (D(x['price'])<=limit if order['side']=='buy' else D(x['price'])>=limit)]
        ladders.append(levels)
    if not all(ladders):return dict(qty=0,reasons=['兩次觀察未持續存在限價內對手盤'])
    delta=current['volume_shares']-previous['volume_shares']
    if delta<=0:return dict(qty=0,reasons=['累計成交量未增加或重置'])
    cap=min(int(min(sum(x['shares'] for x in levels) for levels in ladders)*D(RULES['visible_depth_fraction'])),int(delta*D(RULES['interval_volume_fraction'])))
    step=1000 if order['channel']=='board' else 1
    n=min(order['qty']-order['filled'],cap)//step*step
    if not n:return dict(qty=0,reasons=['深度／區間成交參與上限不足交易單位'])
    # Use the adverse end of both eligible ladders plus explicit slippage, never favorable last price.
    worst=(max if order['side']=='buy' else min)(D(x['price']) for levels in ladders for x in levels)
    price=worst*(1+D(RULES['adverse_slippage']) if order['side']=='buy' else 1-D(RULES['adverse_slippage']))
    tick=_tick(price,order['stock_id']);price=(price/tick).to_integral_value(rounding=ROUND_CEILING if order['side']=='buy' else ROUND_FLOOR)*tick
    if (order['side']=='buy' and price>limit) or (order['side']=='sell' and price<limit):return dict(qty=0,reasons=['加入滑價後超出限價，保持未成交'])
    value=price*n;tax=(value*D(RULES['etf0050_sell_tax'] if order['stock_id']=='0050' else RULES['stock_sell_tax'])).to_integral_value(rounding=ROUND_FLOOR) if order['side']=='sell' else D(0)
    return dict(qty=n,price=str(price),fee=str(p.fee(value)),tax=str(tax),capacity_shares=cap,volume_delta_shares=delta,reasons=[])


def match(path,observation,clock=j.now):
    if observation['body']['status']!='ok':return dict(fills=[],reasons=['行情取得失敗'])
    quote=observation['body']['quote'];key='simulation_observation:'+j.digest([quote[k] for k in ('market','stock_id','channel','quote_at')])
    with j.connection(path) as con:
        rows=verify(con);s=p.state(rows)
        existing=next((r for r in rows if r['event_key']==key),None)
        if existing:return existing['body']
        previous=next((r['body']['quote'] for r in reversed(rows) if r['kind']=='simulation_observation' and all(r['body']['quote'][k]==quote[k] for k in ('market','stock_id','channel'))),None)
        result=dict(quote_hash=observation['hash'],quote=quote,fills=[],decisions=[],classification=RULES['classification'])
        relevant=[o for o in s['orders'].values() if o['stock_id']==quote['stock_id'] and o['channel']==quote['channel'] and not o['closed'] and o['filled']<o['qty'] and o['session']==str(clock().astimezone(p.TZ).date())]
        if quote['stock_id'] in h.active(rows,str(clock().astimezone(p.TZ).date())):
            result['decisions'].append(dict(reason='有效停牌公告，暫停撮合'))
        else:
            used=0
            for o in sorted(relevant,key=lambda x:x['side']!='sell'):
                estimate=proposal(o,previous,quote,clock())
                if estimate['qty']:
                    step=1000 if o['channel']=='board' else 1
                    remaining=max(0,estimate['capacity_shares']-used)//step*step
                    if estimate['qty']>remaining:
                        # Recalculate cost using a bounded remaining order; never consume depth twice.
                        estimate=proposal(dict(o,qty=o['filled']+remaining),previous,quote,clock()) if remaining else dict(qty=0,reasons=['此觀察容量已使用'])
                result['decisions'].append(dict(order_id=o['order_id'],**estimate))
                if not estimate['qty']:continue
                fill_id='sim:'+j.digest([key,o['order_id']])[:32]
                command=dict(kind='fill',id=fill_id,order_id=o['order_id'],**{k:estimate[k] for k in ('qty','price','fee','tax')},
                    executed_at=clock().isoformat(),evidence=dict(source='paper_execution_report',report_id=fill_id,classification=RULES['classification'],quote_hash=observation['hash']))
                try:
                    event=p.submit(con,command,clock);used+=estimate['qty'];result['fills'].append(event['hash'])
                except ValueError as exc:result['decisions'][-1]['reasons'].append(str(exc));result['decisions'][-1]['not_posted']=True
        j.append(con,key,'simulation_observation',result,clock)
        return result
