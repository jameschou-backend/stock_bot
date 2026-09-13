"""Sealed prospective ranking experiment on existing cash/limit accounting."""
from datetime import datetime, time
from decimal import Decimal as D
from pathlib import Path
import hashlib
import shutil
import uuid

from app import forward_journal as j, forward_portfolio as p, forward_simulation as sim
from app import forward_cash_policy as cash, forward_comparison as comparison
from app import forward_halts as halts, forward_portfolio_service as service
from app.file_lock import file_lock

ROOT = j.ROOT / '.cache/forward-simulation/capacity-v1'
ROLES = ('strategy', 'control', 'benchmark')
RULES = dict(version='capacity-forward-v1', stock_slots=3, idle_asset='cash',
    ranking='signal_frozen_mean_turnover20_twd_descending', control='original_priority_descending',
    entry='next_session_signal_close_limit_ROD_no_chase', exit='inherited_stop12_and_holding63',
    max_drawdown='0.20', risk_setting='provisional_simulation_not_user_approved_live_limit',
    drawdown_action='latch_new_buys_until_explicit_review', broker_execution_verified=False,
    historical_return_transferable=False, live_qualified=False)
CODE = ('app/capacity_forward.py', 'app/capacity_forward_runner.py',
        'scripts/run_capacity_forward.py', 'scripts/prepare_capacity_signals.py', 'docs/prereg_capacity_forward_v1.md',
        'artifacts/forward_simulation/price_chronology_20260913.json')


def hashes():
    return {name: hashlib.sha256((j.ROOT/name).read_bytes()).hexdigest() for name in CODE}


def _verify_rows(rows, role=None):
    seal = next((r for r in rows if r['kind'] == 'capacity_protocol'), None)
    if (not seal or seal['body']['rules'] != RULES or seal['body']['code_sha256'] != hashes()
            or seal['body']['role'] not in ROLES or role and seal['body']['role'] != role):
        raise ValueError('成交金額前向版本或角色不符，請保留舊帳本另開版本')
    inherited = next((r for r in rows if r['kind'] == 'cash_allocation_protocol'), None)
    expected = 'benchmark' if seal['body']['role'] == 'benchmark' else 'strategy'
    if (not inherited or inherited['body']['role'] != expected or inherited['body']['rules'] != cash.RULES
            or inherited['body']['code_sha256'] != cash.hashes()):
        raise ValueError('繼承帳本角色或規則不符')
    if expected == 'strategy':
        state=p.state(rows)
        if '0050' in state['holdings'] or any(o['stock_id']=='0050' and not o['closed']
                and o['filled']<o['qty'] for o in state['orders'].values()):
            raise ValueError('現金策略不可持有0050或有效委託')
        if any(r['seq']>inherited['seq'] and r['kind'] in ('order','funding_intent')
               and r['body']['stock_id']=='0050' for r in rows):
            raise ValueError('現金策略出現0050買入紀錄')
    return rows


def verify(path, role=None):
    if not Path(path).is_file(): raise ValueError('排序版尚未初始化')
    with j.connection(path) as con:
        return _verify_rows(sim.verify(con),role)


def _copy_rows(source, target):
    with j.connection(source) as con:
        rows = sim.verify(con)
    with j.connection(target) as con:
        for r in rows:
            con.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)',
                (r['seq'], r['event_key'], r['kind'], r['recorded_at'], j.encode(r['body']), r['previous_hash'], r['hash']))


def initialize(root=ROOT, strategy=p.PATH, benchmark=comparison.BENCHMARK,
               signals=j.DEFAULT_PATH, clock=j.now):
    root = Path(root)
    with file_lock(root.parent/'.capacity-init.lock', timeout=0):
        if root.exists():
            for role in ROLES: verify(root/(role+'.sqlite3'), role)
            return root
        staging = root.parent/('.capacity-init-'+uuid.uuid4().hex)
        try:
            cash.initialize(staging, strategy, benchmark, signals, clock)
            _copy_rows(staging/'strategy.sqlite3', staging/'control.sqlite3')
            for role in ROLES:
                path = staging/(role+'.sqlite3')
                with j.connection(path) as con:
                    rows = sim.verify(con)
                    j.append(con, 'capacity_protocol', 'capacity_protocol',
                        dict(rules=RULES, code_sha256=hashes(), role=role, inherited_head=rows[-1]['hash']), clock)
                    if role != 'benchmark':
                        for o in p.state(rows)['orders'].values():
                            if not o['closed'] and o['filled'] < o['qty']:
                                p.submit(con, dict(kind='cancel', id='rank-init:'+o['order_id'], order_id=o['order_id'],
                                    reason='新版本首次開盤前重新依完整封存候選排序，舊帳本不變'), clock)
                if role != 'benchmark':
                    save_plans(path, staging/'signals.sqlite3', clock=clock)
                verify(path, role)
            staging.rename(root)
        finally:
            if staging.exists(): shutil.rmtree(staging)
    return root


def risk(rows):
    closes = [r for r in rows if r['kind'] == 'close']
    current = closes[-1]['body'].get('nav') if closes else None
    values = [D(r['body']['nav']) for r in closes if r['body'].get('nav') is not None]
    peak = max([D('1000000')] + values)
    drawdown = max(D(0), 1-D(current)/peak) if current is not None else None
    control = next((r for r in reversed(rows) if r['kind'] == 'capacity_pause'), None)
    paused = control['body']['paused'] if control else False
    reason = ('缺少完整收盤淨值' if drawdown is None else
              '整戶回撤達停止新增門檻' if drawdown >= D(RULES['max_drawdown']) else
              control['body']['reason'] if paused else None)
    return dict(blocked=reason is not None, reason=reason, drawdown=str(drawdown) if drawdown is not None else None,
                high_water_nav=str(peak), threshold=RULES['max_drawdown'], paused=paused)


def pause(path, paused, reason, clock=j.now):
    if type(paused) is not bool or not isinstance(reason, str) or len(reason.strip()) < 10:
        raise ValueError('需明確暫停狀態與至少10字核對說明')
    with j.connection(path) as con:
        rows=_verify_rows(sim.verify(con))
        if next(r['body']['role'] for r in rows if r['kind']=='capacity_protocol')=='benchmark':
            raise ValueError('0050獨立基準不套用策略暂停')
        r=risk(rows)
        if not paused and (r['drawdown'] is None or D(r['drawdown']) >= D(RULES['max_drawdown'])):
            raise ValueError('淨值缺漏或回撤仍達門檻，不可解除')
        body=dict(paused=paused, reason=reason.strip(), observed_at=clock().isoformat())
        event=j.append(con, 'capacity_pause:'+j.digest(body), 'capacity_pause', body, clock)
        if paused:
            for o in p.state(rows)['orders'].values():
                if o['side']=='buy' and not o['closed'] and o['filled']<o['qty']:
                    p.submit(con, dict(kind='cancel', id='pause:'+event['hash']+':'+o['order_id'],
                        order_id=o['order_id'], reason='暫停新增：'+reason), clock)
        return event


def _proof(signal_path, day, clock):
    if not Path(signal_path).is_file(): raise ValueError('今日候選來源尚未封存')
    with j.connection(signal_path) as con:
        j.initialize(con, clock)
        rows=j.read_events(con)
    result=next((r for r in rows if r['kind']=='signal' and r['body']['signal_date']==day), None)
    if result is None or not result['body']['eligible']:
        raise ValueError('候選來源缺漏或未通過資料檢查')
    if j.timestamp(result['recorded_at'])>clock(): raise ValueError('訊號尚未取得')
    return result


def plan(rows, proof, clock):
    """Both arms share sizing, risk, timing and fees. Only sort key differs."""
    role=next(r['body']['role'] for r in rows if r['kind']=='capacity_protocol')
    if role=='benchmark': raise ValueError('基準不產生個股計畫')
    s=p.state(rows);mark=s['last_close']
    if not mark or mark['body']['nav'] is None: raise ValueError('缺少完整上一收盤淨值')
    day=mark['body']['date'];session=p.next_session(s,day)
    if clock()>=datetime.fromisoformat(session).replace(hour=9,tzinfo=p.TZ):
        raise ValueError('已開盤，不得事後新增或重排買單')
    if proof['body']['signal_date']!=day or not proof['body']['eligible']:
        raise ValueError('訊號與上一收盤不符')
    candidates=proof['body']['candidates']
    def key(c):
        score=(p.num(c['liquidity_before_entry']['mean_turnover20_twd']) if role=='strategy'
               else D(str(c['priority'])))
        if not score.is_finite(): raise ValueError('排序分數必須有限')
        return (-score,c['members'][0],c.get('event_id',''))
    candidates=sorted(candidates,key=key)
    guard=risk(rows);nav=D(mark['body']['nav']);reserved,_=p.reserves(s,str(clock().astimezone(p.TZ).date()))
    available=s['cash']-reserved
    active={o['stock_id'] for o in s['orders'].values() if not o['closed'] and o['filled']<o['qty'] and o['side']=='buy'}
    occupied=set(s['holdings'])|active
    orders=[];decisions=[]
    for rank,c in enumerate(candidates,1):
        sid=c['members'][0];liq=c['liquidity_before_entry'];why=guard['reason']
        if not why and (sid in occupied or len(occupied)>=3): why='已持有／已預留或三個名額已滿'
        if not why and (not liq['complete_20_sessions'] or p.num(liq['mean_turnover20_twd'])<50000000): why='20日流動性不足'
        if not why and (c['signal_date']!=day or c['entry_date']!=session): why='候選交易時序不符'
        decision=dict(stock_id=sid,rank=rank,event_id=c.get('event_id'),mean_turnover20_twd=liq['mean_turnover20_twd'],
            priority=c['priority'],selected=False,reason=why,signal_date=day,entry_session=session)
        if not why:
            price=p.num(c['planning_reference_close'],True)
            total=service.affordable(price,max(D(0),min(nav/3,available)-40))
            board=min(total//1000*1000,int(p.num(liq['adv20_shares'])*D('.01'))//1000*1000)
            for channel,n in (('board',board),('odd',total%1000)):
                if not n: continue
                cost=price*n+p.fee(price*n)
                if cost>available: continue
                identity=f'capacity:{role}:{session}:{sid}:{channel}'
                orders.append(dict(kind='order',id=identity,order_id=identity,stock_id=sid,side='buy',channel=channel,
                    qty=n,limit_price=str(price),session=session,reason='frozen_capacity_candidate',
                    signal=c,signal_proof=proof,source_hash=proof['hash']))
                available-=cost;decision['selected']=True
            decision['reason']='依排序預留資金，成交待觀察' if decision['selected'] else '現金或可交易股數不足'
            if decision['selected']: occupied.add(sid)
        decisions.append(decision)
    return dict(signal_hash=proof['hash'],signal_date=day,entry_session=session,role=role,
                risk=guard,decisions=decisions,orders=orders,classification=sim.RULES['classification'])


def save_plans(path, signal_path, clock=j.now):
    rows=verify(path)
    role=next(r['body']['role'] for r in rows if r['kind']=='capacity_protocol')
    if role=='benchmark': return comparison.reinvest_dividends(path,clock)
    # Commit sell intent before any candidate, ranking or new-buy risk failure.
    s=p.state(rows);day=str(clock().astimezone(p.TZ).date())
    exits=comparison.save_exits(path,clock) if s['last_close'] and s['last_close']['body']['date']==day else []
    with j.connection(path) as con:
        rows=_verify_rows(sim.verify(con));s=p.state(rows);mark=s['last_close']
        if not mark: raise ValueError('尚無收盤起點')
        session=p.next_session(s,mark['body']['date'])
        event_key='capacity_plan:'+session
        previous=next((r for r in rows if r['event_key']==event_key),None)
        if previous: return dict(plan=previous,exits=exits,entry_block=risk(rows)['reason'])
        if halts.estimated(rows) or halts.active(rows,session): raise ValueError('停牌或估價未核實，暫停新增')
        guard=risk(rows)
        if guard['blocked'] and not guard['paused']:
            body=dict(paused=True,reason=guard['reason'],observed_at=clock().isoformat())
            j.append(con,'capacity_pause:'+j.digest(body),'capacity_pause',body,clock)
            rows=j.read_events(con)
        proof=_proof(signal_path,mark['body']['date'],clock)
        proposed=plan(rows,proof,clock)
        saved=[p.submit(con,c,clock)['hash'] for c in proposed.pop('orders')]
        record=j.append(con,event_key,'capacity_plan',dict(**proposed,order_hashes=saved),clock)
        return dict(plan=record,exits=exits,entry_block=proposed['risk']['reason'])
