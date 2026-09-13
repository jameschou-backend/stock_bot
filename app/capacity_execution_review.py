"""Compare user-supplied execution reports without changing simulation accounts."""
from decimal import Decimal as D
import re
from app import forward_journal as j, forward_portfolio as p

FIELDS={'execution_id','order_id','stock_id','side','channel','qty','price','fee','tax','executed_at'}


def compare(rows,executions,clock=j.now):
    orders=p.state(rows)['orders'];seen=set();groups={}
    for raw in executions:
        if set(raw)!=FIELDS:raise ValueError('成交CSV欄位須為：'+','.join(sorted(FIELDS)))
        e=dict(raw);identity=e['execution_id']
        if not isinstance(identity,str) or not identity.strip() or identity in seen:
            raise ValueError('成交回報編號缺漏或重複')
        seen.add(identity)
        order=orders.get(e['order_id'])
        if not order:raise ValueError('回報沒有對應的事前委託：'+str(e['order_id']))
        if any(e[k]!=order[k] for k in ('stock_id','side','channel')):
            raise ValueError('回報股票、買賣方向或交易別不符')
        if not re.fullmatch(r'[1-9]\d*',str(e['qty'])):raise ValueError('成交股數需正整數，單位為股')
        e['qty']=int(e['qty'])
        if order['channel']=='board' and e['qty']%1000:raise ValueError('整股成交須1000股倍數')
        at=j.timestamp(e['executed_at'])
        if at>clock() or at<j.timestamp(order['recorded_at']) or str(at.astimezone(p.TZ).date())!=order['session']:
            raise ValueError('成交時間超前、早於委託或不在委託交易日')
        for key in ('price','fee','tax'):e[key]=p.num(e[key],key=='price')
        if e['side']=='buy' and e['tax']!=0:raise ValueError('買進回報不可有賣出稅')
        group=groups.setdefault(e['order_id'],[]);group.append(e)
        if sum(x['qty'] for x in group)>order['qty']:raise ValueError('成交總股數超過原委託')
    result=[]
    for identity,order in orders.items():
        supplied=groups.get(identity,[])
        actual_qty=sum(x['qty'] for x in supplied)
        actual_value=sum((x['price']*x['qty'] for x in supplied),D(0))
        simulated=[r['body'] for r in rows if r['kind']=='fill' and r['body']['order_id']==identity]
        simulated_qty=sum(x['qty'] for x in simulated)
        actual_price=actual_value/actual_qty if actual_qty else None
        simulated_value=sum((D(x['price'])*x['qty'] for x in simulated),D(0))
        simulated_price=simulated_value/simulated_qty if simulated_qty else None
        breach=any((x['price']>D(order['limit_price']) if order['side']=='buy' else x['price']<D(order['limit_price'])) for x in supplied)
        result.append(dict(order_id=identity,stock_id=order['stock_id'],side=order['side'],channel=order['channel'],
            planned_qty=order['qty'],simulated_qty=simulated_qty,reported_qty=actual_qty if supplied else None,
            reported_average_price=str(actual_price) if actual_price is not None else None,
            simulated_average_price=str(simulated_price) if simulated_price is not None else None,
            reported_fees_and_tax=str(sum((x['fee']+x['tax'] for x in supplied),D(0))) if supplied else None,
            adverse_vs_simulation_bps=float((actual_price/simulated_price-1)*10000*(1 if order['side']=='buy' else -1))
                if actual_price is not None and simulated_price is not None else None,
            limit_breach=breach,status='限價不符，需核對委託或回報' if breach else '已提供回報，尚未驗證來源真實性' if supplied else '尚未提供回報，不能當作未成交'))
    return dict(rows=result,report_count=len(seen),classification='user_supplied_execution_comparison',
                broker_transport_verified=False,live_qualified=False,
                note='僅比對提供的回報，不寫入模擬帳本；未提供回報不是零成交，也不是完整真實帳戶損益。')
