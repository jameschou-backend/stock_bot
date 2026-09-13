"""Chronological single-session research replay with separate board/odd tapes."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR, ROUND_CEILING
import heapq

from skills.contingent_execution import Plan, ConfirmationGate, OPEN, CUTOFF, integer
from skills.intraday_limit_replay import limit_price


@dataclass(frozen=True)
class Tape:
    stock_id: str
    channel: str
    market: str
    day: str
    # Each row: microseconds since midnight, integer price cents, shares, actual trade flag.
    rows: tuple
    source: str
    sha256: str
    synthetic: bool

    def validate(self, day):
        if (self.day!=day or self.market not in ('TWSE','TPEX') or self.channel not in ('board','odd')
                or not self.source or len(self.sha256)!=64 or type(self.synthetic) is not bool):
            raise ValueError('Invalid tape identity or provenance')
        if any(c not in '0123456789abcdef' for c in self.sha256):
            raise ValueError('Invalid source hash')
        previous=-1
        for at,price,shares,actual in self.rows:
            integer(at);integer(price,1);integer(shares)
            if at<previous or at>=86400*1_000_000 or type(actual) is not bool:
                raise ValueError('Invalid or nonmonotonic tape')
            if self.channel=='board' and shares%1000:
                raise ValueError('Board volume must be normalized from lots to shares')
            previous=at


class SettlementGate(ConfirmationGate):
    """Extend the sealed gate only for zero/negative net residual sales."""
    def confirm_fill(self, fill_id, order_id, at, qty, price_cents, cash_cents):
        if type(cash_cents) is not int:raise ValueError('Integer cash required')
        if cash_cents>0:
            return super().confirm_fill(fill_id,order_id,at,qty,price_cents,cash_cents)
        self._time(at);integer(qty,1);integer(price_cents,1)
        if not fill_id or fill_id in self.seen or order_id not in self.orders:
            raise ValueError('Invalid sale confirmation identity')
        order=self.orders[order_id];plan=order['plan']
        if (plan.side!='sell' or at<=order['sent_at'] or qty>order['remaining']
                or price_cents<plan.limit_cents or qty>self.holdings.get(plan.stock_id,0)
                or plan.channel=='board' and qty%1000 or -cash_cents>self.available):
            raise ValueError('Invalid or unfunded residual sale')
        self.clock=at;self.seen.add(fill_id);order['remaining']-=qty
        self.holdings[plan.stock_id]-=qty;self.available+=cash_cents
        if not self.holdings[plan.stock_id]:del self.holdings[plan.stock_id]
        self.events.append(dict(kind='fill',fill_id=fill_id,order_id=order_id,at=at,
            qty=qty,price_cents=price_cents,cash_cents=cash_cents))


def cumulative_bill(gross_cents, side, sid, slippage_bps):
    if not gross_cents:return dict(commission=0,tax=0,slippage=0,total=0)
    gross=Decimal(gross_cents)/100
    fee=max(Decimal(20),(gross*Decimal('.001425')).quantize(Decimal(1),rounding=ROUND_HALF_UP))
    tax=(gross*Decimal('.001' if sid=='0050' else '.003')).quantize(Decimal(1),rounding=ROUND_FLOOR) if side=='sell' else Decimal(0)
    slip=(gross*Decimal(slippage_bps)/10000).quantize(Decimal(1),rounding=ROUND_CEILING)
    return dict(commission=int(fee*100),tax=int(tax*100),slippage=int(slip*100),total=int((fee+tax+slip)*100))


def replay_day(spec, tapes):
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
    gate=SettlementGate(day,plans,spec['holdings'],spec['available_cents'],spec['slots'])
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
