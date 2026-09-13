"""Cross-day session variant; retain sealed one-day studies unchanged."""
from collections import defaultdict
from datetime import date
from decimal import Decimal
import heapq
from skills.contingent_replay import SettlementGate, cumulative_bill
from skills.contingent_execution import Plan, OPEN, CUTOFF, integer
from skills.intraday_limit_replay import limit_price


class RightsGate(SettlementGate):
    def __init__(self,*args,reserved_members=(),**kwargs):
        super().__init__(*args,**kwargs)
        self.reserved_members=frozenset(reserved_members)
        if any(not isinstance(s,str) or len(s)!=4 or not s.isdigit() for s in self.reserved_members):
            raise ValueError('Invalid pending share member')
        if len(set(self.holdings)|self.reserved_members)>self.slots:
            raise ValueError('Pending stock rights exceed portfolio slots')

    def submit_ready(self,at):
        self._time(at)
        if at<OPEN:raise ValueError('Order before start time')
        self.clock=at;sent=[]
        ordered=[p for p in self.plans if p.side=='sell']+[p for p in self.plans if p.side=='buy']
        for plan in ordered:
            if plan.order_id in self.orders:continue
            if plan.side=='buy':
                members=set(self.holdings)|self.reserved_members|{o['plan'].stock_id for o in self.orders.values()
                    if o['plan'].side=='buy' and o['remaining']}
                if (plan.stock_id not in members and len(members)>=self.slots or self.available<plan.budget_cents):break
                self.available-=plan.budget_cents
            self.orders[plan.order_id]=dict(plan=plan,sent_at=at,remaining=plan.qty,reserved=plan.budget_cents)
            sent.append(plan.order_id)
        self.events.append(dict(kind='submit',at=at,order_ids=sent,available=self.available))
        return sent


def replay_session(spec, tapes):
    day=spec['date'];decision=spec['decision_date'];calendar=spec['calendar']
    if (calendar!=sorted(set(calendar)) or day not in calendar or calendar.index(day)==0
            or calendar[calendar.index(day)-1]!=decision or spec['prior_data_date']!=decision):
        raise ValueError('Decision must be previous audited trading session')
    for d in calendar:date.fromisoformat(d)
    plans=[Plan(**p) for p in spec['plans']]
    keys=[(p.stock_id,p.channel) for p in plans]
    if len(keys)!=len(set(keys)):raise ValueError('Only one order per stock/channel; no volume reuse')
    if any(p.signal_date>decision for p in plans):raise ValueError('Plan uses future signal')
    delay=spec['credit_delay_us']
    if delay is not None:integer(delay)
    slip=spec['slippage_bps'];integer(slip)
    if slip>1000:raise ValueError('Slippage outside supported range')
    for p in plans:
        p.validate(day)
        legal=Decimal(str(limit_price(p.limit_cents/100,p.stock_id,p.side)))*100
        if legal!=p.limit_cents:raise ValueError('Order limit violates price tick size')
        if p.side=='buy':
            total=p.qty*p.limit_cents
            if p.budget_cents<total+cumulative_bill(total,'buy',p.stock_id,slip)['total']:
                raise ValueError('Buy budget omits full-order costs')
        if p.channel=='board':integer(spec['prior_avg_volume_shares'][p.stock_id],1)
    for tape in tapes.values():tape.validate(day)
    for key,tape in tapes.items():
        if key!=(tape.stock_id,tape.channel):raise ValueError('Tape mapping identity mismatch')
    missing=[dict(stock_id=sid,channel=channel) for sid,channel in keys if (sid,channel) not in tapes]
    base=dict(date=day,scope='single_day_contingent_execution',total_return=None,live_qualified=False,
        credit_policy='no_same_day_reuse' if delay is None else 'assumed_delay',credit_delay_us=delay,
        slippage_bps=slip,decision_date=decision)
    if missing:return dict(base,completed=False,missing=missing,events=[],fills=[])
    selected={k:tapes[k] for k in keys}
    gate=RightsGate(day,plans,spec['holdings'],spec['available_cents'],spec['slots'], reserved_members=spec.get('reserved_members',[]))
    gate.submit_ready(OPEN)
    # Equal timestamps cannot finance fills of newly released buy orders.
    queue=[];seq=0
    for key in sorted(selected):
        for index,row in enumerate(selected[key].rows):
            at,price,volume,actual=row
            if OPEN<at<CUTOFF and actual:
                heapq.heappush(queue,(at,1,seq,('tick',key,index,price,volume)));seq+=1
    eligible=defaultdict(int);filled=defaultdict(int);fills=[];blocked=[]
    by_key={(p.stock_id,p.channel):p for p in plans}
    while queue:
        at=queue[0][0]
        if at>=CUTOFF:break
        batch=[]
        while queue and queue[0][0]==at:batch.append(heapq.heappop(queue))
        changed=False
        for _,_,_,event in batch:
            if event[0]=='credit':
                _,fid,net=event
                gate.confirm_available('credit:'+fid,fid,at,net);changed=True
                continue
            _,key,index,price,volume=event;plan=by_key[key]
            order=gate.orders.get(plan.order_id)
            if not order or not order['remaining'] or order['sent_at']>=at:continue
            if not (price<plan.limit_cents if plan.side=='buy' else price>plan.limit_cents):continue
            eligible[key]+=volume
            capacity=eligible[key]//(100 if plan.channel=='board' else 20)
            step=1000 if plan.channel=='board' else 1
            if plan.channel=='board':capacity=min(capacity,spec['prior_avg_volume_shares'][plan.stock_id]//100)
            qty=min(order['remaining'],max(0,capacity//step*step-filled[key]))
            if not qty:continue
            gross=qty*plan.limit_cents;prior_gross=filled[key]*plan.limit_cents
            bill=cumulative_bill(prior_gross+gross,plan.side,plan.stock_id,slip)
            before=cumulative_bill(prior_gross,plan.side,plan.stock_id,slip)
            delta={k:bill[k]-before[k] for k in bill}
            net=gross-delta['total'] if plan.side=='sell' else gross+delta['total']
            if plan.side=='sell' and net<0 and -net>gate.available:
                blocked.append(dict(order_id=plan.order_id,at=at,reason='residual_sale_cost_unfunded'));continue
            fid=f'simulated:{plan.order_id}:{index}'
            gate.confirm_fill(fid,plan.order_id,at,qty,plan.limit_cents,net)
            filled[key]+=qty;changed=True
            fills.append(dict(fill_id=fid,order_id=plan.order_id,stock_id=plan.stock_id,
                side=plan.side,channel=plan.channel,at=at,qty=qty,price_cents=plan.limit_cents,
                cash_cents=net,costs_cents=delta,eligible_shares=eligible[key],capacity_shares=capacity,
                sent_at=order['sent_at'],source_tick_index=index,source_sha256=selected[key].sha256))
            if plan.side=='sell' and net>0 and delay is not None:
                heapq.heappush(queue,(at+delay,0,seq,('credit',fid,net)));seq+=1
        if changed:gate.submit_ready(at)
    gate.close()
    cash=spec['available_cents']+sum(f['cash_cents']*(1 if f['side']=='sell' else -1) for f in fills)
    if cash!=gate.available+sum(gate.receivables.values()):raise ValueError('Cash conservation failed')
    holdings=dict(spec['holdings'])
    for f in fills:holdings[f['stock_id']]=holdings.get(f['stock_id'],0)+f['qty']*(1 if f['side']=='buy' else -1)
    if {s:q for s,q in holdings.items() if q}!=gate.holdings:raise ValueError('Holdings conservation failed')
    return dict(base,completed=True,synthetic=any(t.synthetic for t in selected.values()),
        source_sha256={':'.join(k):t.sha256 for k,t in selected.items()},fills=fills,events=gate.events,
        available_cents=gate.available,receivable_cents=sum(gate.receivables.values()),holdings=gate.holdings,
        unfilled=[dict(order_id=p.order_id,qty=p.qty-filled[(p.stock_id,p.channel)],
            submitted=p.order_id in gate.orders) for p in plans if p.qty>filled[(p.stock_id,p.channel)]],
        execution_blocks=blocked,audit=dict(cash_conserved=True,holdings_conserved=True))
