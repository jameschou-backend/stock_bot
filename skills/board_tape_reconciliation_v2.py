"""Full TWSE daily-total decomposition, including explicit basket constituents.

This extends source coverage only; matching daily totals is still not proof of a
complete sequence, queue position or one's own executable order.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import re

from skills.board_tape_reconciliation import (
    _payload, _rows, _indexed, number, parse_twse, digest,
)


def basket_requests(payload,day):
    """Extract server-provided constituent identities; never guess absent holdings."""
    _payload(payload,day)
    if payload.get('selectType')!='M' or not payload.get('title','').endswith('鉅額交易日成交資訊-股票組合'):
        raise ValueError('Expected complete basket-block list')
    result=[];totals=[];seen=set()
    for row in _rows(payload):
        if str(row['序號']).strip() in ('總計','合計'):
            totals.append((number(row['成交總股數']),number(row['成交總金額'],cents=True)));continue
        identity=row['資料內容']
        if not isinstance(identity,dict) or set(identity)!={'sub','stockType','buyNo','date'}:
            raise ValueError('Basket source query identity incomplete')
        if identity['date']!=day.replace('-',''):
            raise ValueError('Basket identity date mismatch')
        if (not str(identity['sub']).isdigit() or not str(identity['stockType']).isdigit()
                or not re.fullmatch(r'[A-Za-z0-9]+',str(identity['buyNo']))):
            raise ValueError('Unexpected basket identity values')
        key=json.dumps(identity,sort_keys=True)
        if key in seen:raise ValueError('Duplicate basket query identity')
        seen.add(key)
        result.append(dict(identity=identity,security_count=number(row['股票種數']),
                           shares=number(row['成交總股數']),amount_cents=number(row['成交總金額'],cents=True)))
    if result:
        if totals!=[(sum(x['shares'] for x in result),sum(x['amount_cents'] for x in result))]:
            raise ValueError('Basket list totals do not reconcile')
    elif totals and totals!=[(0,0)]:raise ValueError('Nonempty basket totals without constituent identities')
    return result


def basket_key(item):
    return sha256(json.dumps(item['identity'],sort_keys=True,separators=(',',':')).encode()).hexdigest()


def parse_basket_detail(payload,request,day):
    """Require exact total/count agreement with the official parent basket row."""
    _payload(payload,day)
    if any(str(payload.get(key,''))!=str(expected) for key,expected in request['identity'].items()):
        raise ValueError('Basket detail identity does not equal parent query')
    if payload.get('selectType')=='M':raise ValueError('Parent basket list returned instead of constituent detail')
    rows=_rows(payload)
    result={};totals=[]
    for row in rows:
        sid=str(row['證券代號']).strip()
        if sid in ('總計','合計',''):
            totals.append((number(row['成交股數']),number(row['成交金額'],cents=True)));continue
        if not re.fullmatch(r'[A-Z0-9]{4,10}',sid):raise ValueError('Unknown basket security identity')
        if sid in result:raise ValueError('Duplicate security within basket')
        result[sid]=dict(shares=number(row['成交股數']),amount_cents=number(row['成交金額'],cents=True))
    actual=(sum(r['shares'] for r in result.values()),sum(r['amount_cents'] for r in result.values()))
    if (len(result)!=request['security_count'] or actual!=(request['shares'],request['amount_cents'])
            or (totals and totals!=[actual])):
        raise ValueError('Basket constituent count/amount/quantity does not equal parent')
    return result


def parse_twse_complete(parts,details,day):
    """Subtract every independently retrieved basket; missing details block the day."""
    requests=basket_requests(parts['block_basket'],day)
    expected={basket_key(r) for r in requests}
    if set(details)!=expected:raise ValueError('Required basket constituent reports are missing or extra')
    baskets={}
    for request in requests:
        parsed=parse_basket_detail(details[basket_key(request)],request,day)
        for sid,r in parsed.items():
            previous=baskets.setdefault(sid,dict(shares=0,amount_cents=0))
            previous['shares']+=r['shares'];previous['amount_cents']+=r['amount_cents']
    # Reuse the sealed ordinary/odd/fixed/single-block unit checks. This copy only
    # defers basket subtraction to the explicit, validated constituent totals.
    without_baskets=deepcopy(parts)
    without_baskets['block_basket']['data']=[]
    for key in ('total','totalCount'):
        if key in without_baskets['block_basket']:without_baskets['block_basket'][key]=0
    result=parse_twse(without_baskets,day)
    for sid,row in result.items():
        basket=baskets.get(sid,dict(shares=0,amount_cents=0))
        row['shares']-=basket['shares'];row['amount_cents']-=basket['amount_cents']
        if row['shares']<0 or row['amount_cents']<0:raise ValueError('Negative residual after basket subtraction')
        if basket['shares']:row['transaction_count']=None
        row['components']['block_basket']=basket
        row['scope']='twse_total_minus_all_other_sessions_including_verified_baskets'
    return result


def verify_report(path,root):
    path,root=Path(path).resolve(),Path(root).resolve()
    if path.with_suffix('.sha256').read_text().strip()!=digest(path):raise ValueError('Full board audit hash mismatch')
    result=json.loads(path.read_text())
    if result.get('schema')!='board_tape_reconciliation_v2':raise ValueError('Unknown full board audit schema')
    for name,expected in {**result['input_sha256'],**result['code_sha256']}.items():
        source=(root/name).resolve()
        if not source.is_relative_to(root) or digest(source)!=expected:raise ValueError('Full board input changed: '+name)
    if any(result.get(k) is not False for k in ('strict_data_ready','live_qualified','own_order_fill_proven')):
        raise ValueError('Daily totals cannot qualify actual trading')
    return result
