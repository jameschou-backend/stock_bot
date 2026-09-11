"""Evidence-based restatements in NEW books; original observations are immutable.

Historical validation uses virtual execution dates. Materialized events are
recorded now and explicitly marked as reconstructed, never as original evidence.
"""
from copy import deepcopy
from datetime import datetime
from decimal import Decimal as D
import hashlib
from pathlib import Path
import re
import uuid

from app import forward_journal as j, forward_portfolio as p

ROOT = j.ROOT / '.cache/forward-validation/restatements'
VERSION = 'paper-restatement-v1'
EDITABLE = {'fill', 'entitlement', 'delivery', 'close', 'cancel'}


def read(path):
    if not Path(path).is_file():
        raise ValueError('原始帳本不存在')
    with j.connection(path) as con:
        p.initialize(con)
        return j.read_events(con)


def evidence(value):
    required = {'reference','reviewer','reason','text'}
    if not isinstance(value,dict) or set(value)!=required or any(not isinstance(v,str) or not v.strip() for v in value.values()):
        raise ValueError('更正需回報／文件編號、核對人、原因及原始證據文字')
    if not 20<=len(value['text'])<=20000:
        raise ValueError('更正證據文字須20至20000字')
    return dict(value,text_sha256=hashlib.sha256(value['text'].encode()).hexdigest(),verified_by='user_report_not_broker_authenticated')


def _project(rows, operations):
    if not isinstance(operations,list) or not 1<=len(operations)<=100:
        raise ValueError('每次更正需1至100項明確操作')
    by_hash={r['hash']:r for r in rows}
    edits={};inserts={};first_close=next((r['hash'] for r in rows if r['kind']=='close'),None)
    for index,op in enumerate(operations):
        if not isinstance(op,dict):raise ValueError('更正操作格式錯誤')
        mode=op.get('op')
        if mode in ('replace','void'):
            target=op.get('target');original=by_hash.get(target)
            if not original or original['kind'] not in EDITABLE or target in edits or target==first_close:
                raise ValueError('只能更正未重複指定的成交、權益、交付、收盤或取消；期初收盤不能改')
            if set(op)!=({'op','target','body'} if mode=='replace' else {'op','target'}):
                raise ValueError('更正欄位不符')
            if mode=='replace':
                body=deepcopy(op['body'])
                if not isinstance(body,dict):raise ValueError('更正內容須完整JSON物件')
                # Identity changes would leave ambiguous references in later rows.
                for field in ('order_id','action_id'):
                    if field in original['body'] and body.get(field)!=original['body'][field]:
                        raise ValueError('不能更換原始委託或權益識別碼')
                edits[target]=dict(original,body=body)
            else:edits[target]=None
        elif mode=='insert':
            if set(op)!={'op','before','kind','body'} or op['kind'] not in EDITABLE:
                raise ValueError('補登只接受成交、權益、交付、收盤或取消，且必須指定插入位置')
            target=op['before']
            if target not in by_hash and target!='$end':raise ValueError('補登位置不存在')
            if target==rows[0]['hash'] or target==first_close:raise ValueError('不能在期初帳本之前補登')
            item=dict(kind=op['kind'],body=deepcopy(op['body']),hash='insert:'+str(index),
                      event_key='restated_insert:'+str(index),recorded_at=None)
            inserts.setdefault(target,[]).append(item)
        else:raise ValueError('更正類型須replace、void或insert')
    result=[]
    for row in rows:
        result.extend(inserts.get(row['hash'],[]))
        item=edits.get(row['hash'],row)
        if item is not None:result.append(item)
    result.extend(inserts.get('$end',[]))
    return result


def _effective_time(row,clock):
    b=row['body'];kind=row['kind']
    if kind=='fill':return j.timestamp(b['executed_at'])
    if kind=='entitlement':return datetime.fromisoformat(p.iso(b['ex_date'])).replace(hour=8,tzinfo=p.TZ)
    if kind=='delivery':return datetime.fromisoformat(p.iso(b['date'])).replace(hour=8,tzinfo=p.TZ)
    if kind=='close':return datetime.fromisoformat(p.iso(b['date'])).replace(hour=18,tzinfo=p.TZ)
    if kind=='cancel' and not row.get('recorded_at'):raise ValueError('取消補登需改用更正原有取消紀錄；無法推定取消的實際時間')
    return j.timestamp(row['recorded_at']) if row.get('recorded_at') else clock()


def replay(projected,clock=j.now):
    out=[];deviations=[];seen_keys=set();closes=set()
    for original in projected:
        row=deepcopy(original);kind=row['kind'];b=row['body'];s=p.state(out)
        if not isinstance(b,dict):raise ValueError('帳本內容格式錯誤')
        if row['event_key'] in seen_keys:raise ValueError('更正後紀錄編號重複')
        seen_keys.add(row['event_key'])
        when=_effective_time(row,clock)
        if when>clock():raise ValueError('更正不能包含未來發生的成交、權益或收盤')
        row['recorded_at']=row.get('recorded_at') or when.isoformat()
        try:
            if kind=='fill':
                p._fill(b,s,out,when)
            elif kind=='entitlement':
                p._entitlement(b,s,out,str(when.astimezone(p.TZ).date()))
            elif kind=='delivery':
                right=s['rights'].get(b['action_id'])
                if not right or right['paid'] or p.iso(b['date'])<right['delivery_date'] or not b['evidence']:
                    raise ValueError('交付缺少權益、重複、早於應交付日或沒有證據')
            elif kind=='close':
                day=p.iso(b['date'])
                if day in closes or day not in s['calendar'] or (s['last_close'] and day<=s['last_close']['body']['date']):
                    raise ValueError('更正後收盤日期重複、逆序或不是已確認交易日')
                if any(o['session']<=day and o['filled']<o['qty'] and not o['closed'] for o in s['orders'].values()):raise ValueError('更正後收盤仍有未核對剩餘委託，需一併更正取消或成交')
                if type(b['actions_reviewed']) is not bool or not b['source']:raise ValueError('收盤缺核對狀態或來源')
                for sid,price in b['prices'].items():
                    if not re.fullmatch(r'\d{4}',sid):raise ValueError('估價代號須四碼')
                    p.num(price,True)
                if b.get('estimated_prices'):
                    from app.forward_halts import valuation_prices
                    observed={sid:price for sid,price in b['prices'].items() if sid not in b['estimated_prices']}
                    b['prices'],b['estimated_prices']=valuation_prices(out,day,observed)
                nav=p.valuation({**s,'mark':{'body':b}})
                b['nav']=str(nav) if nav is not None else None
                b['restated']=True
                closes.add(day)
            elif kind in ('order','funding_intent'):
                # Historical decisions are not re-optimized after observing corrections.
                if b['order_id'] in s['orders']:raise ValueError('委託識別碼重複')
                if not re.fullmatch(r'\d{4}',b['stock_id']) or b['side'] not in ('buy','sell') or b['channel'] not in ('odd','board'):
                    raise ValueError('委託欄位錯誤')
                n=p.qty(b['qty']);p.num(b['limit_price'],True);p.iso(b['session'])
                if (b['channel']=='board' and n%1000) or (b['channel']=='odd' and n>=1000):raise ValueError('整零股委託數量不符')
            elif kind=='cancel':
                o=s['orders'].get(b['order_id'])
                if not o or o['closed'] or o['filled']>=o['qty'] or not b['reason']:raise ValueError('取消紀錄失去對應的未完成委託')
            elif kind=='decision':
                try:p._decision(b,s,when.astimezone(p.TZ))
                except (ValueError,KeyError):deviations.append(dict(event=row['hash'],note='更正後不符合原策略出場條件；保留原決策及實際成交，不重選歷史策略'))
            if kind in ('entitlement','fill','delivery') and s['last_close']:
                day=str(when.astimezone(p.TZ).date())
                if day<=s['last_close']['body']['date']:raise ValueError('補登位置晚於該日收盤，請放到相應收盤之前')
            out.append(row)
            state=p.state(out)
            if state['cash']<0 or any(v['qty']<0 or v['cost']<0 for v in state['holdings'].values()):
                raise ValueError('更正後現金、股數或持股成本為負')
        except (KeyError,TypeError,ArithmeticError) as exc:
            raise ValueError('更正後相依紀錄或數值不完整：'+kind) from exc
    s=p.state(out)
    reserve,_=p.reserves(s,str(clock().astimezone(p.TZ).date()))
    if reserve>s['cash']:raise ValueError('更正後預留資金不足；請核對尚未完成的委託，不能假造現金')
    return out,deviations


def _summary(rows):
    s=p.state(rows);nav=p.valuation(s)
    return dict(cash=str(s['cash']),nav=str(nav) if nav is not None else None,
        costs=str(s['costs']),realized_pnl=str(s['realized_pnl']),
        holdings={sid:{k:str(v) if isinstance(v,D) else v for k,v in pos.items()} for sid,pos in s['holdings'].items()},
        curve=[dict(date=r['body']['date'],nav=r['body']['nav']) for r in rows if r['kind']=='close'])


def preview(path,operations,proof,clock=j.now):
    rows=read(path)
    if any(r['kind']=='restatement_lineage' for r in rows):raise ValueError('請從原始帳本合併全部更正重新建立版本，不能對更正版本疊加重建')
    changed,deviations=replay(_project(rows,operations),clock)
    body=dict(version=VERSION,source_path=str(Path(path).resolve()),source_head=rows[-1]['hash'],
              operations=operations,evidence=evidence(proof),
              projection_sha256=j.digest(changed),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    command=dict(id=j.digest(body)[:32],**body)
    return dict(command=command,before=_summary(rows),after=_summary(changed),deviations=deviations,
                classification='restated_not_original_forward_evidence',live_qualified=False)


def _remap(value,mapping):
    if isinstance(value,dict):return {k:_remap(v,mapping) for k,v in value.items()}
    if isinstance(value,list):return [_remap(v,mapping) for v in value]
    return mapping.get(value,value) if isinstance(value,str) else value


def materialize(path,command,directory=ROOT,clock=j.now):
    """New book only, serialized with its source. Never auto-activate a correction."""
    if not isinstance(command,dict) or not re.fullmatch(r'[0-9a-f]{32}',str(command.get('id',''))):raise ValueError('更正識別碼格式錯誤')
    if str(Path(path).resolve())!=command.get('source_path'):raise ValueError('更正預覽不屬於此帳本')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    target=directory/(command['id']+'.sqlite3')
    if target.resolve()==Path(path).resolve():raise ValueError('更正版本必須與原始帳本分開')
    with j.connection(path) as source:
        p.initialize(source,clock);rows=j.read_events(source)
        if rows[-1]['hash']!=command['source_head']:raise ValueError('原帳本已更新，請重新預覽更正')
        projected,deviations=replay(_project(rows,command['operations']),clock)
        expected=dict(version=VERSION,source_path=str(Path(path).resolve()),source_head=rows[-1]['hash'],
            operations=command['operations'],evidence=evidence({k:v for k,v in command['evidence'].items() if k in {'reference','reviewer','reason','text'}}),
            projection_sha256=j.digest(projected),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if {k:v for k,v in command.items() if k!='id'}!=expected or command['id']!=j.digest(expected)[:32]:
            raise ValueError('更正內容、程式或預覽結果已變更')
        if target.exists():
            existing=read(target)
            lineage=next((r for r in existing if r['kind']=='restatement_lineage'),None)
            if not lineage or lineage['body']['command']!=command:raise ValueError('更正版本識別碼衝突')
            return target
        temp=directory/(command['id']+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with j.connection(temp) as con:
                first=p.initialize(con,clock);mapping={rows[0]['hash']:first['hash']}
                lineage=j.append(con,'restatement_lineage','restatement_lineage',dict(command=command,
                    reconstructed_at=clock().isoformat(),original_event_times={r['hash']:r['recorded_at'] for r in rows},
                    classification='restated_not_original_forward_evidence',live_qualified=False,deviations=deviations),clock)
                for row in projected:
                    if row['kind']=='portfolio_rules':continue
                    b=deepcopy(row['body'])
                    # Only executable references are rebased; historical proofs remain exact.
                    if row['kind'] in ('order','funding_intent'):
                        for field in ('reason','funding_intent'):
                            if field in b:b[field]=_remap(b[field],mapping)
                    saved=j.append(con,row['event_key'],row['kind'],b,clock)
                    mapping[row['hash']]=saved['hash']
                # Values must be identical to preview even though new record timestamps are NOW.
                if _summary(j.read_events(con))!=_summary(projected):raise ValueError('更正版本會計結果與預覽不符')
            temp.rename(target)
        finally:
            if temp.exists():temp.unlink()
    return target


def versions(path,directory=ROOT):
    result=[]
    for candidate in sorted(Path(directory).glob('*.sqlite3')):
        rows=read(candidate)
        lineage=next((r for r in rows if r['kind']=='restatement_lineage'),None)
        if lineage and lineage['body']['command']['source_path']==str(Path(path).resolve()):
            result.append(dict(path=str(candidate),created_at=lineage['recorded_at'],source_head=lineage['body']['command']['source_head']))
    return result
