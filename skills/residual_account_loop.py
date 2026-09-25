"""Sealed integer-account loop with one explicit slot-membership seam.

Copied from million_replay.Replay.run; the only loop change replaces its inline
holding-count admission check. Accounting and execution order remain identical.
A neutral-policy full-account comparison detects accidental differences.
"""
from collections import defaultdict
import math

from skills.million_replay import (Replay, MIN_AMOUNT20, MIN_FEE, COMMISSION,
                                  SLIPPAGE, PARTICIPATION, ODD_PARTICIPATION)


class ResidualAccountLoop(Replay):
    def run(self):
        peak = self.initial_cash
        for day in self.days[(self.days>=self.start)&(self.days<=self.end)]:
            self.used = defaultdict(int)
            self.day_cost = self.day_basis = 0.
            opening_nav = self.previous_nav
            old_assets = sum(h['qty']*self.marks[sid]['price'] for sid,h in self.holdings.items() if h['qty'])+self.receivable_value()
            income = self.corporate_day(day)
            # Marks are valuation only; missing current quotes remain visibly stale.
            for sid in {s for s,h in self.holdings.items() if h['qty']}|{r['stock_id'] for r in self.receivables}:
                price = self.raw(day,sid)
                if price:
                    self.marks[sid] = dict(price=price,date=str(day.date()))
                elif sid not in self.marks:
                    raise ValueError('No valuation price for held asset '+sid)
            marked_assets = sum(h['qty']*self.marks[sid]['price'] for sid,h in self.holdings.items() if h['qty'])+self.receivable_value()
            # Paid dividends migrate receivable to cash and are not market P&L.
            paid_today = sum(r['cash_change'] for r in self.cash_ledger if r['date']==str(day.date()) and r['kind'] in ('dividend_payment','fractional_share_payment'))
            market_pnl = marked_assets-old_assets-income+paid_today
            if day == self.start:
                self.buy_etf(day,'initial_allocation')
            for sid,holding in list(self.holdings.items()):
                if sid!='0050' and holding['qty'] and self.positions[day] >= holding['due_index']:
                    self.order(day,sid,'sell',holding['qty'],'scheduled_exit',holding['event_id'])
            # Zero positions can close only after outstanding stock rights deliver.
            for sid,h in list(self.holdings.items()):
                if sid!='0050' and h['qty']==0 and not any(r.get('qty',0)>0 and r['event_id']==h['event_id'] for r in self.receivables):
                    cohort = next(c for c in self.cohorts if c['event_id']==h['event_id'])
                    cohort['exit_date'] = str(day.date())
                    del self.holdings[sid]
            for event in self.events.get(day,[]):
                sid = event['members'][0]
                why = None
                if sid in self.holdings:
                    why = 'overlapping_member'
                elif len(self.entry_slot_members()) >= self.slots:
                    why = 'slots_full'
                elif not math.isfinite(float(self.amount20.at[day,sid])) or self.amount20.at[day,sid]<MIN_AMOUNT20:
                    why = 'prior_liquidity_below_50m_or_missing'
                previous_price = self.prior(day,sid)
                if why or not previous_price:
                    self.orders.append(dict(date=str(day.date()),stock_id=sid,name=self.names.get(sid,sid),
                        event_id=event['event_id'],signal_date=event['signal_date'],side='buy',channel='event',
                        requested_qty=0,filled_qty=0,reason='leader_entry',failure=why or 'no_prior_price'))
                    continue
                budget = min(opening_nav/self.slots, self.cash+self.holdings.get('0050',{}).get('qty',0)*(self.prior(day,'0050') or 0))
                qty = max(0,math.floor((budget-2*MIN_FEE)/(previous_price*(1+COMMISSION+SLIPPAGE))))
                if not qty:
                    continue
                self.corporate.prepare(sid)
                if self.cash<budget and self.holdings.get('0050',{}).get('qty',0):
                    etf_prior = self.prior(day,'0050')
                    funding_qty = min(self.holdings['0050']['qty'],math.ceil((budget-self.cash+2*MIN_FEE)/(etf_prior*(1-COMMISSION-SLIPPAGE-.001))))
                    self.order(day,'0050','sell',int(funding_qty),'fund_stock',event['event_id'],event['signal_date'])
                due = self.positions[day]+self.horizon
                self.holdings[sid] = dict(qty=0,event_id=event['event_id'],due_index=due)
                if self.raw(day,sid):
                    self.marks[sid] = dict(price=self.raw(day,sid),date=str(day.date()))
                filled = self.order(day,sid,'buy',qty,'leader_entry',event['event_id'],event['signal_date'])
                if filled:
                    self.cohorts.append(dict(event,stock_id=sid,name=self.names.get(sid,sid),bought_qty=filled,
                        due_index=due,due_date=str(self.days[due].date()) if due<len(self.days) else None,exit_date=None))
                else:
                    del self.holdings[sid]
            self.buy_etf(day,'idle_cash')
            assets = 0.
            for sid,h in self.holdings.items():
                if not h['qty']:
                    continue
                if sid not in self.marks:
                    self.marks[sid] = dict(price=self.raw(day,sid),date=str(day.date()))
                mark = self.marks[sid]
                value = h['qty']*mark['price']; assets+=value
                self.holding_rows.append(dict(date=str(day.date()),stock_id=sid,name=self.names.get(sid,sid),
                    qty=h['qty'],price=mark['price'],market_value=value,mark_date=mark['date'],
                    event_id=h['event_id'],stale=mark['date']!=str(day.date())))
            receivable = self.receivable_value()
            nav = self.cash+assets+receivable
            expected = opening_nav+market_pnl+income+self.day_basis-self.day_cost
            if not math.isclose(nav,expected,abs_tol=.06,rel_tol=1e-10):
                raise ValueError(f'Daily account does not reconcile {day.date()}: {nav} vs {expected}')
            peak = max(peak,nav)
            self.daily.append(dict(date=str(day.date()),opening_nav=opening_nav,market_pnl=market_pnl,
                dividend_entitlement=income,execution_basis_pnl=self.day_basis,cost=self.day_cost,
                cash=self.cash,market_value=assets,receivable=receivable,nav=nav,
                daily_return=nav/opening_nav-1,total_return=nav/self.initial_cash-1,
                drawdown=nav/peak-1,holdings=len([s for s,h in self.holdings.items() if s!='0050' and h['qty']]),
                stale_holdings=sum(r['stale'] for r in self.holding_rows if r['date']==str(day.date()))))
            self.previous_nav = nav
        return dict(daily=self.daily,trades=self.trades,orders=self.orders,corporate_actions=self.actions,
            cash_ledger=self.cash_ledger,holdings=self.holding_rows,cohorts=self.cohorts,
            receivables=self.receivables,settings=dict(initial_cash=self.initial_cash,slots=self.slots,
                horizon=self.horizon,slippage=SLIPPAGE,commission=COMMISSION,minimum_fee=MIN_FEE,
                participation=PARTICIPATION,odd_participation=ODD_PARTICIPATION,benchmark=self.benchmark))
