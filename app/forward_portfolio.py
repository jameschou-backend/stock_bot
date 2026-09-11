"""Versioned prospective paper accounting. No quote-to-fill inference or brokerage.

Commands are serialized by the caller's journal transaction. All money uses
Decimal. Existing v1 evidence and strategy source are deliberately untouched.
"""
from datetime import date, datetime, time
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import re
from zoneinfo import ZoneInfo
from app import forward_journal as j

PATH = j.ROOT / '.cache/forward-validation/portfolio-v2.sqlite3'
TZ = ZoneInfo('Asia/Taipei')
D = Decimal
ZERO = D(0)
RULES = dict(version='paper-portfolio-v2', initial_cash='1000000', slots=3,
             allocation='last_complete_close_nav_divided_by_3', idle_asset='0050',
             stop='close_below_88pct_adjusted_entry_then_next_session_limit',
             holding_sessions=63, buy_retry=False, exit_retry=True,
             fill_source='user_supplied_paper_execution_report', live_qualified=False,
             cash_policy='confirmed_sale_proceeds_available_no_margin',
             corporate_policy='explicit_entitlement_then_payment_or_share_delivery',
             note='Separate prospective execution protocol; not a replay of historical returns')
CODE = ('app/forward_portfolio.py', 'app/forward_portfolio_service.py',
        'app/forward_journal.py')


def num(value, positive=False):
    try: result = D(str(value))
    except Exception: raise ValueError('數字格式錯誤') from None
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise ValueError('數字必須有限且不可為負')
    return result


def qty(value):
    if type(value) is not int or value <= 0:
        raise ValueError('股數必須為正整數')
    return value


def iso(value):
    if date.fromisoformat(value).isoformat() != value:
        raise ValueError('日期須為 YYYY-MM-DD')
    return value


def fee(value):
    return max(D(20), (value * D('.001425')).quantize(D(1), rounding=ROUND_HALF_UP))


def initialize(con, clock=j.now):
    body = dict(rules=RULES, code_sha256={p: hashlib.sha256((j.ROOT/p).read_bytes()).hexdigest() for p in CODE})
    rows = j.read_events(con)
    if rows and (rows[0]['kind'] != 'portfolio_rules' or rows[0]['body'] != body):
        raise ValueError('帳本版本或程式已變更；請保留舊帳本，另開新版')
    return j.append(con, 'portfolio_rules', 'portfolio_rules', body, clock)


def state(rows):
    """Replay reports in recorded order; no silent reordering of late executions."""
    cash, realized, expenses = D(RULES['initial_cash']), ZERO, ZERO
    holdings, orders, rights = {}, {}, {}
    mark, last_close, calendar, decisions = None, None, [], {}
    for event in rows:
        b, kind = event['body'], event['kind']
        if kind == 'calendar': calendar = b['sessions']
        elif kind == 'close': mark = last_close = event
        elif kind == 'decision': decisions[b['stock_id']] = event
        elif kind == 'order':
            orders[b['order_id']] = dict(b, hash=event['hash'], recorded_at=event['recorded_at'],
                                        filled=0, closed=False)
        elif kind == 'cancel': orders[b['order_id']]['closed'] = True
        elif kind == 'fill':
            o = orders[b['order_id']]
            n, price = b['qty'], D(b['price'])
            cost = D(b['fee']) + D(b['tax'])
            expenses += cost
            o['filled'] += n
            sid = o['stock_id']
            if o['side'] == 'buy':
                cash -= price*n + cost
                p = holdings.setdefault(sid, dict(qty=0, cost=ZERO, entry_gross=ZERO,
                    entry_date=str(j.timestamp(b['executed_at']).astimezone(TZ).date()), stop_basis=ZERO))
                p['qty'] += n
                p['cost'] += price*n + cost
                p['entry_gross'] += price*n
                p['stop_basis'] = p['entry_gross']/p['qty']
            else:
                p = holdings[sid]
                allocated = p['cost']*n/p['qty']
                cash += price*n-cost
                realized += price*n-cost-allocated
                p['cost'] -= allocated
                p['entry_gross'] -= p['stop_basis']*n
                p['qty'] -= n
                if not p['qty']:
                    del holdings[sid]
                    decisions.pop(sid, None)
            mark = None  # A close recorded before a fill cannot mark the new holdings.
        elif kind == 'entitlement':
            rights[b['action_id']] = dict(b, paid=False)
            p = holdings[b['stock_id']]
            if b['action_type'] == 'split':
                p['stop_basis'] /= D(b['ratio'])
            else:
                p['stop_basis'] -= D(b['cash_per_share'])
            p['entry_gross'] = p['stop_basis']*p['qty']
            mark = None
        elif kind == 'delivery':
            r = rights[b['action_id']]
            r['paid'] = True
            if r['action_type'] == 'cash':
                cash += D(r['amount'])
                realized += D(r['amount'])
            else:
                p = holdings[r['stock_id']]
                p['qty'] = r['result_qty']
                p['entry_gross'] = p['stop_basis']*p['qty']
            mark = None
    return dict(cash=cash, realized_pnl=realized, costs=expenses, holdings=holdings,
                orders=orders, rights=rights, mark=mark, last_close=last_close, calendar=calendar, decisions=decisions)


def reserves(s, day):
    cash, shares = ZERO, {}
    for o in s['orders'].values():
        left = o['qty']-o['filled']
        if o['closed'] or not left or o['session'] < day: continue
        if o['side'] == 'buy':
            value = D(o['limit_price'])*left
            cash += value+fee(value)
        else: shares[o['stock_id']] = shares.get(o['stock_id'], 0)+left
    return cash, shares


def valuation(s):
    if not s['holdings']: return s['cash'] + sum(D(r['amount']) for r in s['rights'].values() if not r['paid'] and r['action_type']=='cash')
    if not s['mark'] or not s['mark']['body']['actions_reviewed']: return None
    if any(not r['paid'] and r['action_type']=='split' for r in s['rights'].values()): return None
    prices = s['mark']['body']['prices']
    if any(sid not in prices for sid in s['holdings']): return None
    return (s['cash'] + sum(D(prices[sid])*p['qty'] for sid, p in s['holdings'].items())
            + sum(D(r['amount']) for r in s['rights'].values() if not r['paid'] and r['action_type']=='cash'))


def summary(path=PATH, clock=j.now):
    with j.connection(path) as con: rows = j.read_events(con)
    s = state(rows)
    day = str(clock().astimezone(TZ).date())
    reserved, _ = reserves(s, day)
    nav = valuation(s)
    result = dict(initialized=bool(rows), cash=str(s['cash']), reserved_cash=str(reserved),
        available_cash=str(s['cash']-reserved), nav=str(nav) if nav is not None else None,
        price_date=s['mark']['body']['date'] if s['mark'] else None,
        realized_pnl=str(s['realized_pnl']), fees_and_tax=str(s['costs']),
        holdings=[dict(stock_id=sid, qty=p['qty'], cost=str(p['cost']), average_cost=str(p['cost']/p['qty'])) for sid,p in s['holdings'].items()],
        orders=[dict(o, remaining=o['qty']-o['filled'], status=(
            '已成交' if o['filled']==o['qty'] else '已取消剩餘' if o['closed'] else
            '已到期，待核對回報' if o['session']<day else '部分成交' if o['filled'] else '等待成交證據')) for o in s['orders'].values()],
        fill_count=sum(r['kind']=='fill' for r in rows), live_qualified=False, rows=rows)
    return result


def next_session(s, day):
    if day not in s['calendar']: raise ValueError('當日不在封存交易日曆')
    days = [d for d in s['calendar'] if d>day]
    if not days: raise ValueError('缺少下一交易日，不能用平日推測')
    return days[0]


def submit(con, command, clock=j.now):
    """Append a validated command, idempotent by its stable id and exact payload."""
    initialize(con, clock)
    c = dict(command)
    kind, key = c.pop('kind'), c.pop('id')
    if not isinstance(key,str) or not key.strip(): raise ValueError('必須提供唯一紀錄編號')
    rows = j.read_events(con)
    existing = next((r for r in rows if r['event_key']==kind+':'+key), None)
    if existing: return j.append(con, kind+':'+key, kind, c, clock)
    s = state(rows)
    local = clock().astimezone(TZ)
    day = str(local.date())
    if kind == 'calendar':
        sessions = c['sessions']
        if not sessions or sessions != sorted(set(sessions)): raise ValueError('日曆需排序且不可重複')
        for d in sessions: iso(d)
        if not c['source']: raise ValueError('需提供交易日曆來源')
        if s['calendar'] and {d for d in sessions if d<=day}!={d for d in s['calendar'] if d<=day}:
            raise ValueError('不可覆寫已封存的過去交易日')
    elif kind == 'close':
        if iso(c['date'])!=day or local.hour<18 or day not in s['calendar']:
            raise ValueError('只能在交易日18點後封存當日收盤')
        if any(o['session']<=day and o['filled']<o['qty'] and not o['closed'] for o in s['orders'].values()):
            raise ValueError('請先核對當日未完成委託並取消剩餘，才可結算')
        if any(r['kind']=='close' and r['body']['date']==day for r in rows):
            raise ValueError('當日收盤已封存，不能覆寫')
        if not c['source'] or type(c['actions_reviewed']) is not bool:
            raise ValueError('需記錄來源與公司行動核對狀態')
        for sid,p in c['prices'].items():
            if not re.fullmatch(r'\d{4}',sid): raise ValueError('股票代號須四碼')
            num(p, True)
        projected=valuation({**s,'mark':{'body':c}})
        if c['nav'] != (str(projected) if projected is not None else None):
            raise ValueError('收盤淨值與成交帳本不符')
    elif kind in ('order', 'funding_intent'):
        if kind=='funding_intent' and (c['side']!='buy' or c['stock_id']=='0050' or local.date().isoformat()>=c['session']):
            raise ValueError('資金需求只供進場前一日封存個股買入')
        _order(c,s,rows,local, intent=kind=='funding_intent')
    elif kind == 'cancel':
        o = s['orders'].get(c['order_id'])
        if not o or o['closed'] or o['filled']==o['qty'] or not c['reason'].strip():
            raise ValueError('只能取消仍有剩餘數量的委託，並需填原因')
    elif kind == 'fill':
        _fill(c,s,rows,local)
    elif kind == 'decision':
        _decision(c,s,local)
    elif kind == 'entitlement':
        _entitlement(c,s,rows,day)
    elif kind == 'delivery':
        r = s['rights'].get(c['action_id'])
        if not r or r['paid'] or c['date']!=day or day<r['delivery_date'] or not c['evidence']:
            raise ValueError('權益未確認、重複交付或未到交付日')
    else: raise ValueError('不支援的帳本操作')
    event = j.append(con, kind+':'+key, kind, c, clock)
    return event


def _order(c,s,rows,local, intent=False):
    sid, side, channel = c['stock_id'],c['side'],c['channel']
    n, price = qty(c['qty']),num(c['limit_price'],True)
    session = iso(c['session']); day = str(local.date())
    if not re.fullmatch(r'\d{4}',sid) or side not in ('buy','sell') or channel not in ('board','odd'):
        raise ValueError('代號、買賣別或交易別錯誤')
    if channel=='board' and n%1000 or channel=='odd' and n>=1000: raise ValueError('整張為1000倍數，零股1至999股')
    if c['order_id'] in s['orders']: raise ValueError('委託編號已使用')
    if session not in s['calendar'] or session<day or (session==day and local.time()>=time(13,30)):
        raise ValueError('委託日期必須是尚未收盤的已知交易日')
    if any(not r['paid'] and r['stock_id']==sid and r['action_type']=='split' for r in s['rights'].values()):
        raise ValueError('股票權益尚未交付，暫停此股票委託')
    # A stale unconfirmed fill must be reconciled before releasing its slot/cash.
    if any(o['session']<day and not o['closed'] and o['filled']<o['qty'] for o in s['orders'].values()):
        raise ValueError('過期委託尚未核對，請先登錄回報或取消剩餘')
    reserved, sold = reserves(s, day)
    if side=='sell':
        if n>s['holdings'].get(sid,{}).get('qty',0)-sold.get(sid,0): raise ValueError('賣出超過未預留持股')
        if sid!='0050':
            decision=s['decisions'].get(sid)
            if not decision or c['reason']!=decision['hash'] or session<=decision['body']['date']:
                raise ValueError('個股賣出需事先封存出場決策')
        elif c['reason']!='fund_stock_entries': raise ValueError('0050僅可為個股進場釋放資金')
        return
    if not intent and price*n+fee(price*n)>s['cash']-reserved: raise ValueError('可用現金不足；0050尚未成交的賣出不算現金')
    mark=s['last_close']
    if not mark or mark['body']['nav'] is None or not mark['body']['actions_reviewed'] or next_session(s,mark['body']['date'])!=session:
        raise ValueError('買進需要上一交易日完整淨值及公司行動核對')
    if sid=='0050':
        if c['reason']!='idle_cash': raise ValueError('0050買入需標示閒置資金配置')
        if any(o['side']=='sell' and o['stock_id']=='0050' and not o['closed'] and o['filled']<o['qty'] for o in s['orders'].values()):
            raise ValueError('0050資金釋放委託仍未完成')
        return
    if local.date().isoformat()==session and local.time()>=time(9):
        # Intraday buys only release an already sealed pre-open intent after ETF sale.
        if not c.get('funding_intent'): raise ValueError('開盤後新增個股委託需盤前封存的資金需求')
        intent=next((r for r in rows if r['hash']==c['funding_intent'] and r['kind']=='funding_intent'),None)
        if not intent or intent['body']['stock_id']!=sid or intent['body']['session']!=session or c['channel']!=intent['body']['channel'] or c['order_id']!=intent['body']['order_id'] or n>intent['body']['qty'] or price!=D(intent['body']['limit_price']):
            raise ValueError('資金需求與委託不符')
    signal=c['signal']
    proof=c['signal_proof']
    if (j.digest({k:v for k,v in proof.items() if k!='hash'})!=proof['hash']
            or proof['kind']!='signal' or not proof['body']['eligible']
            or signal not in proof['body']['candidates']
            or c['source_hash']!=proof['hash']
            or j.timestamp(proof['recorded_at'])>=datetime.combine(date.fromisoformat(session),time(9),TZ)):
        raise ValueError('需提供盤前封存的原策略訊號證據')
    if signal['signal_date']!=mark['body']['date'] or signal['entry_date']!=session or signal['members']!=[sid] or not c['source_hash']:
        raise ValueError('訊號日期或候選不符')
    if price!=num(signal['planning_reference_close'],True): raise ValueError('限價必須等於事前訊號日收盤價')
    liq=signal['liquidity_before_entry']
    if not liq['complete_20_sessions'] or num(liq['mean_turnover20_twd'])<50000000:
        raise ValueError('缺少20日流動性證據或成交金額不足')
    active=[o for o in s['orders'].values() if o['side']=='buy' and not o['closed'] and o['filled']<o['qty'] and o['session']>=day]
    occupied={k for k in s['holdings'] if k!='0050'}|{o['stock_id'] for o in active if o['stock_id']!='0050'}|{sid}
    if len(occupied)>3: raise ValueError('最多持有或預留三檔個股')
    if sid in s['holdings'] and s['holdings'][sid]['entry_date']!=session:
        raise ValueError('本版不對既有強勢股加碼')
    # Use the frozen close NAV, never a new intraday mark or fixed initial capital.
    prior_nav=D(mark['body']['nav'])
    used=sum(D(o['limit_price'])*(o['qty']-o['filled'])+fee(D(o['limit_price'])*(o['qty']-o['filled'])) for o in active if o['stock_id']==sid)
    used+=s['holdings'].get(sid,{}).get('cost',ZERO)
    if used+price*n+fee(price*n)>prior_nav/3: raise ValueError('超過上一收盤淨值三分之一')
    board=sum(o['qty'] for o in s['orders'].values() if o['stock_id']==sid and o['session']==session and o['channel']=='board' and o['side']=='buy')
    if channel=='board' and board+n>num(liq['adv20_shares'])*D('.01'): raise ValueError('超過20日均量1%')


def _fill(c,s,rows,local):
    o=s['orders'].get(c['order_id'])
    if not o or o['closed']: raise ValueError('委託不存在或已取消，須另行核對遲到回報')
    n,price=qty(c['qty']),num(c['price'],True)
    commission,tax=num(c['fee']),num(c['tax'])
    at=j.timestamp(c['executed_at'])
    if at>local or at<j.timestamp(o['recorded_at']) or str(at.astimezone(TZ).date())!=o['session']:
        raise ValueError('成交時間超前或早於委託或不在委託交易日')
    trading_time=at.astimezone(TZ).time()
    if not time(9,10 if o['channel']=='odd' else 0)<=trading_time<=time(13,30):
        raise ValueError('本版僅接受盤中交易時段回報')
    last=[j.timestamp(r['body']['executed_at']) for r in rows if r['kind']=='fill']
    if last and at<max(last): raise ValueError('回報順序倒置，需另行對帳，不可回填改寫成本')
    if s['last_close'] and o['session']<=s['last_close']['body']['date']: raise ValueError('此日已結算，禁止補填成交改寫收盤淨值')
    if n>o['qty']-o['filled'] or (o['channel']=='board' and n%1000): raise ValueError('成交超過剩餘量或整張單位錯誤')
    if c['evidence']['source']!='paper_execution_report' or not c['evidence']['report_id']:
        raise ValueError('僅接受紙上成交回報；報價不能變成成交')
    if any(r['kind']=='fill' and r['body']['evidence']['report_id']==c['evidence']['report_id'] for r in rows):
        raise ValueError('成交回報編號重複')
    if (o['side']=='buy' and price>D(o['limit_price'])) or (o['side']=='sell' and price<D(o['limit_price'])):
        raise ValueError('成交違反限價')
    if o['side']=='buy':
        if tax: raise ValueError('買進不可收賣出交易稅')
        shadow={**s,'orders':{k:dict(v) for k,v in s['orders'].items()}}
        shadow['orders'][o['order_id']]['filled']+=n
        reserve,_=reserves(shadow,str(local.date()))
        if price*n+commission+reserve>s['cash']: raise ValueError('成交費用侵占其他委託預留資金')
    elif n>s['holdings'].get(o['stock_id'],{}).get('qty',0): raise ValueError('不能超賣持股')
    elif s['cash']+price*n-commission-tax<0: raise ValueError('費稅造成現金不足')


def _decision(c,s,local):
    mark=s['mark'];sid=c['stock_id']
    if not mark or mark['body']['date']!=c['date'] or c['date']!=str(local.date()) or valuation(s) is None:
        raise ValueError('出場決策需要當日完整收盤及公司行動核對')
    p=s['holdings'].get(sid)
    if not p or sid=='0050': raise ValueError('非個股持倉')
    held=sum(p['entry_date']<=d<=c['date'] for d in s['calendar'])
    stop=D(mark['body']['prices'][sid])<=p['stop_basis']*D('.88')
    if c['reason']!=('stop12_close' if stop else 'holding63' if held>=63 else None):
        raise ValueError('尚未達到12%收盤停損或63交易日')


def _entitlement(c,s,rows,day):
    sid=c['stock_id'];p=s['holdings'].get(sid)
    if c['action_id'] in s['rights'] or not p or c['ex_date']!=day or not c['evidence']:
        raise ValueError('權益重複、缺持倉證據或非今日除權息')
    if s['last_close'] and s['last_close']['body']['date']>=day: raise ValueError('當日已結算，不能回填公司行動')
    if any(r['stock_id']==sid and r['ex_date']==day and r['action_type']==c['action_type'] for r in s['rights'].values()): raise ValueError('同股票同日同類公司行動已登錄，請勿更換編號重複入帳')
    iso(c['delivery_date'])
    if c['delivery_date']<day: raise ValueError('交付日不可早於除權息日')
    if p['entry_date']>=day or c['eligible_qty']!=p['qty']:
        raise ValueError('除權息資格須為前日持股；不得使用除權日新買股數')
    if any(r['kind']=='fill' and str(j.timestamp(r['body']['executed_at']).astimezone(TZ).date())>=day for r in rows):
        raise ValueError('公司行動須在當日成交前核對')
    if any(o['stock_id']==sid and not o['closed'] and o['filled']<o['qty'] for o in s['orders'].values()):
        raise ValueError('公司行動前請先取消舊價格委託')
    if c['action_type']=='cash':
        per=num(c['cash_per_share'],True)
        if per>=p['stop_basis'] or num(c['amount'])!=per*p['qty']: raise ValueError('現金股息金額或停損調整異常')
    elif c['action_type']=='split':
        ratio=num(c['ratio'],True)
        if D(c['result_qty'])!=ratio*p['qty'] or type(c['result_qty']) is not int or c['result_qty']<=0:
            raise ValueError('碎股或未知股數須另外核對，不能默認可交易')
    else: raise ValueError('僅支援已核對現金股息及整數分割；減資等事件需另行對帳')
