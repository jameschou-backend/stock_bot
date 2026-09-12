"""Separate regular/odd MIS snapshots with explicit share-unit normalization."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import requests
from app import forward_journal as j, forward_portfolio as p, forward_odd_lot as odd

PATH=j.ROOT/'.cache/forward-simulation/market-quotes.sqlite3'
BASE='https://mis.twse.com.tw/stock/api/'


def parse(payload,market,sid,channel,retrieved_at):
    if market not in ('tse','otc') or not re.fullmatch(r'\d{4}',sid) or channel not in ('odd','board'):raise ValueError('市場、四碼股票或整零股類別錯誤')
    if payload.get('rtcode')!='0000':raise ValueError('MIS 回應失敗')
    matches=[r for r in payload.get('msgArray',[]) if r.get('c')==sid and r.get('ex')==market and r.get('ch')==sid+'.tw']
    if len(matches)!=1:raise ValueError('MIS 缺少指定股票／市場行情')
    row=matches[0];at=datetime.fromtimestamp(int(row['tlong'])/1000,p.TZ)
    if at.strftime('%Y%m%d')!=row['d'] or at>j.timestamp(retrieved_at):raise ValueError('揭示日期／時間不符或在未來')
    asks=odd._levels(row['a'],row['f'],True);bids=odd._levels(row['b'],row['g'],False)
    if asks and bids and p.num(bids[0]['price'])>=p.num(asks[0]['price']):raise ValueError('交叉／鎖定買賣盤不能用於撮合')
    if not str(row.get('v','')).isdigit():raise ValueError('缺少累計成交量，不推定區間成交')
    multiplier=1000 if channel=='board' else 1
    for level in asks+bids:level['shares']*=multiplier
    return dict(stock_id=sid,market=market,channel=channel,quantity_unit='shares',provider_quantity_unit='lots_1000' if channel=='board' else 'shares',
        quote_at=at.isoformat(),retrieved_at=retrieved_at,asks=asks,bids=bids,volume_shares=int(row['v'])*multiplier,raw_row_sha256=j.digest(row))


def fetch(market,sid,channel,path=PATH,clock=j.now,session_factory=requests.Session):
    if market not in ('tse','otc') or not isinstance(sid,str) or not re.fullmatch(r'\d{4}',sid) or channel not in ('odd','board'):raise ValueError('行情查詢參數錯誤')
    endpoint=BASE+('getOddInfo.jsp' if channel=='odd' else 'getStockInfo.jsp')
    query=[market,sid,channel]
    with j.connection(path) as con:
        rows=j.read_events(con);old=next((r for r in reversed(rows) if r['body'].get('query')==query),None)
        if old and 0<=(clock()-j.timestamp(old['recorded_at'])).total_seconds()<15:return old
        body=dict(query=query,url=endpoint,parser_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        try:
            with session_factory() as client:
                response=client.get(endpoint,params={'ex_ch':market+'_'+sid+'.tw','json':1,'delay':0},timeout=8)
                response.raise_for_status();raw=response.text
            body.update(raw=raw,raw_sha256=hashlib.sha256(raw.encode()).hexdigest())
            body.update(status='ok',quote=parse(json.loads(raw),market,sid,channel,clock().isoformat()))
        except (requests.RequestException,ValueError,KeyError,TypeError,OverflowError) as exc:body.update(status='error',error=type(exc).__name__+': '+str(exc))
        return j.append(con,'quote:'+j.digest(body)+':'+str(int(clock().timestamp())),'quote',body,clock)
