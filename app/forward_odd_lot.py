"""Official MIS intraday odd-lot observations, not inferred executions.

Endpoint and field mapping verified against MIS category and DetailTableItem JS.
Prices a/b and SHARE quantities f/g form ask/bid ladders. No board-lot x1000.
"""
from datetime import datetime, time
import hashlib
from pathlib import Path
import re
import requests
from app import forward_journal as j, forward_portfolio as p

PATH=j.ROOT/'.cache/forward-validation/odd-lot-quotes.sqlite3'
URL='https://mis.twse.com.tw/stock/api/getOddInfo.jsp'
VERSION='mis-odd-evidence-v1'
TTL_SECONDS=15
MAX_AGE_SECONDS=15


def _levels(prices,quantities,ascending):
    if not isinstance(prices,str) or not isinstance(quantities,str):raise ValueError('零股五檔欄位不是字串')
    px=prices.rstrip('_').split('_');qty=quantities.rstrip('_').split('_')
    if len(px)!=len(qty) or len(px)>5:raise ValueError('零股價格與股數檔位不一致')
    out=[];ended=False
    for a,b in zip(px,qty):
        if a in ('','-') and b in ('','-','0'):ended=True;continue
        if ended:raise ValueError('零股五檔中間缺漏')
        # A market-price order (0) has no comparable limit price; reject explicitly.
        price=p.num(a,True)
        if not re.fullmatch(r'\d+',b):raise ValueError('零股掛單量須整數股，不能以張換算')
        n=int(b)
        if n<=0:raise ValueError('已揭示檔位股數須為正')
        if out and ((price<=p.num(out[-1]['price']) if ascending else price>=p.num(out[-1]['price']))):raise ValueError('零股五檔價格順序異常')
        out.append(dict(price=str(price),shares=n))
    return out


def parse(payload,market,sid,retrieved_at):
    if market not in ('tse','otc') or not re.fullmatch(r'\d{4}',sid):raise ValueError('需上市／上櫃及四碼代號')
    if payload.get('rtcode')!='0000' or not isinstance(payload.get('msgArray'),list):raise ValueError('證交所零股API未成功或格式改變')
    found=[x for x in payload['msgArray'] if x.get('c')==sid and x.get('ex')==market and x.get('ch')==sid+'.tw']
    if len(found)!=1:raise ValueError('未取得指定股票盤中零股行情；不改用整張行情')
    row=found[0]
    day=datetime.strptime(row['d'],'%Y%m%d').date()
    # tlong is the provider disclosure timestamp; do not substitute client receipt time.
    stamp=datetime.fromtimestamp(int(row['tlong'])/1000,p.TZ)
    if stamp.date()!=day or stamp>j.timestamp(retrieved_at):raise ValueError('零股揭示時間與交易日期不符或在未來')
    asks=_levels(row['a'],row['f'],True);bids=_levels(row['b'],row['g'],False)
    if asks and bids and p.num(bids[0]['price'])>=p.num(asks[0]['price']):raise ValueError('零股買賣盤交叉／鎖定，不能當作可用成交深度')
    return dict(stock_id=sid,market=market,channel='intraday_odd',quantity_unit='shares',
        quote_at=stamp.isoformat(),retrieved_at=retrieved_at,asks=asks,bids=bids,
        source=URL,raw_row_sha256=j.digest(row),daily_volume_shares=int(row['v']) if str(row.get('v','')).isdigit() else None)


def history(evidence_path=PATH):
    if not Path(evidence_path).exists():return []
    with j.connection(evidence_path) as con:return j.read_events(con)


def refresh(market,sid,evidence_path=PATH,clock=j.now,session_factory=requests.Session):
    if market not in ('tse','otc') or not isinstance(sid,str) or not re.fullmatch(r'\d{4}',sid):raise ValueError('市場或股票代號錯誤')
    with j.connection(evidence_path) as con:
        rows=j.read_events(con)
        last=next((r for r in reversed(rows) if r['body'].get('query')==[market,sid]),None)
        if last and 0<=(clock()-j.timestamp(last['recorded_at'])).total_seconds()<TTL_SECONDS:return last
        at=clock().isoformat()
        body=dict(version=VERSION,query=[market,sid],source=URL,requested_at=at,
            parser_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        try:
            with session_factory() as client:
                response=client.get(URL,params={'ex_ch':market+'_'+sid+'.tw','json':1,'delay':0},timeout=15)
                response.raise_for_status();raw=response.text
            body.update(raw=raw,raw_sha256=hashlib.sha256(raw.encode()).hexdigest())
            import json
            body.update(status='ok',quote=parse(json.loads(raw),market,sid,clock().isoformat()))
        except (requests.RequestException,ValueError,KeyError,TypeError,OverflowError) as exc:
            body.update(status='error',error=type(exc).__name__+': '+str(exc))
        return j.append(con,'odd_quote:'+j.digest(body),'odd_quote',body,clock)


def assess(quote,order,at):
    reasons=[];when=j.timestamp(at);observed=j.timestamp(quote['quote_at'])
    if quote['channel']!='intraday_odd' or quote['quantity_unit']!='shares' or order['channel']!='odd':raise ValueError('只能比對盤中零股，數量單位須為股')
    if quote['stock_id']!=order['stock_id']:raise ValueError('報價與委託股票不同')
    if str(observed.astimezone(p.TZ).date())!=order['session']:reasons.append('非委託交易日報價')
    age=(when-observed).total_seconds()
    if not 0<=age<=MAX_AGE_SECONDS:reasons.append('報價過期或晚於比對時間')
    local=when.astimezone(p.TZ)
    if not time(9,10)<=local.time()<=time(13,30):reasons.append('不在本版盤中零股時段')
    # Evidence cannot be retroactively treated as something the user observed beforehand.
    if j.timestamp(quote['retrieved_at'])>when:reasons.append('行情在比對時間之後才取得，僅供事後參考')
    price=p.num(order['limit_price'],True)
    levels=quote['asks'] if order['side']=='buy' else quote['bids']
    size=sum(x['shares'] for x in levels if (p.num(x['price'])<=price if order['side']=='buy' else p.num(x['price'])>=price))
    remaining=order['qty']-order.get('filled',0)
    if size<remaining:reasons.append('限價內可見股數少於未成交數量')
    return dict(visible_shares_at_limit=size,remaining_shares=remaining,quote_age_seconds=age,
        suitable_snapshot=not reasons,reasons=reasons,fill_guaranteed=False,
        note='五檔為瞬間揭示；排隊、撤單及撮合會變動，不能據此自動建立成交')


def attach(path,fill_hash,quote_hash,proof,evidence_path=PATH,clock=j.now):
    from app.forward_restatement import evidence
    verified=evidence(proof)
    quote=next((r for r in history(evidence_path) if r['hash']==quote_hash and r['body']['status']=='ok'),None)
    if not quote:raise ValueError('找不到成功的零股來源證據')
    with j.connection(path) as con:
        p.initialize(con,clock);rows=j.read_events(con)
        fill=next((r for r in rows if r['hash']==fill_hash and r['kind']=='fill'),None)
        if not fill:raise ValueError('需先有獨立的成交回報，不能由報價建立成交')
        idx=rows.index(fill);order=p.state(rows[:idx])['orders'][fill['body']['order_id']]
        result=assess(quote['body']['quote'],order,fill['body']['executed_at'])
        # Cumulative fills attached to one snapshot cannot each claim the full depth.
        linked=[r for r in rows if r['kind']=='execution_evidence' and r['body']['quote_hash']==quote_hash]
        duplicate=next((r for r in linked if r['body']['fill_hash']==fill_hash),None)
        total=fill['body']['qty']+sum(r['body']['fill_qty'] for r in linked if r['body']['fill_hash']!=fill_hash)
        body=dict(version=VERSION,fill_hash=fill_hash,fill_qty=fill['body']['qty'],quote_hash=quote_hash,
            quote=quote['body']['quote'],evidence=verified,assessment=result,
            cumulative_attached_shares=total,classification='user_report_with_quote_context_not_broker_verified')
        if duplicate:
            if duplicate['body']['evidence']!=verified:raise ValueError('同一成交證據已存在，不能改寫')
            return duplicate
        if total>result['visible_shares_at_limit']:
            body['assessment']['reasons'].append('同一快照對應累計成交股數超過當時可見深度；需核對其他撮合證據')
            body['assessment']['suitable_snapshot']=False
        return j.append(con,'execution_evidence:'+fill_hash,'execution_evidence',body,clock)


def review(path):
    with j.connection(path) as con:p.initialize(con);rows=j.read_events(con)
    s=p.state(rows);out=[]
    attachments={r['body']['fill_hash']:r['body'] for r in rows if r['kind']=='execution_evidence'}
    for row in rows:
        if row['kind']!='fill':continue
        b=row['body'];o=s['orders'][b['order_id']]
        if o['channel']!='odd':continue
        link=attachments.get(row['hash']);limit=p.num(o['limit_price']);price=p.num(b['price'])
        out.append(dict(fill_hash=row['hash'],order_id=o['order_id'],stock_id=o['stock_id'],qty=b['qty'],price=b['price'],
            adverse_slippage_bps=float((price/limit-1)*10000*(1 if o['side']=='buy' else -1)),
            quote_status='已附比對證據' if link else '缺少零股行情／成交憑證連結',
            issues=link['assessment']['reasons'] if link else ['成交為使用者回報，未驗證實際可成交深度']))
    return dict(fills=out,live_qualified=False)
