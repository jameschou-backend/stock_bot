"""Causal intraday sell intents and a separate, conservative tape-fill proxy.

Input groups contain ALL prints at one timestamp (shares, not lots). The caller
must certify a complete post-entry stream and supply same-basis prices. This
module never sends broker orders. A trigger is not a fill or a live qualification.
"""
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
import re
from zoneinfo import ZoneInfo

from skills.execution_stress import StressOrder
from skills.peak_stop_research import positive


def instant(value):
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError('Timezone-aware timestamps required')
    return result.astimezone(ZoneInfo('Asia/Taipei'))


class IntradayPeakExit(StressOrder):
    """One position/venue; latched exit survives rebounds and session changes."""
    def __init__(self, *, stock_id, qty, venue, peak, asof, basis, source,
                 participation=.01):
        if not re.fullmatch(r'[1-9]\d{3}', stock_id):
            raise ValueError('Individual four-digit stock required')
        if venue not in ('board', 'odd') or type(qty) is not int or qty <= 0:
            raise ValueError('Positive share quantity and explicit venue required')
        self.step = 1000 if venue == 'board' else 1
        if qty % self.step or (venue == 'odd' and qty >= 1000) or not basis or not source:
            raise ValueError('Invalid lot size or missing price-basis/ownership evidence')
        self.participation = positive(participation)
        if self.participation > 1:
            raise ValueError('Participation exceeds one')
        self.stock_id, self.venue, self.basis = stock_id, venue, basis
        self.initial_qty = self.remaining = qty
        self.peak, self.last = positive(peak), instant(asof)
        self.source = source
        self.trigger = None
        self.session = None
        self.orders, self.fills = [], []
        self._setup_stress('control')

    def start_session(self, *, date, known_at, lower, upper, prior_adv_shares,
                      basis, source):
        known = instant(known_at)
        low, high, adv = map(positive, (lower, upper, prior_adv_shares))
        if (basis != self.basis or not source or low >= high or known < self.last
                or known.date().isoformat() != date
                or (self.session and date <= self.session['date'])):
            raise ValueError('Invalid session evidence, chronology or corporate-action basis')
        self.session = dict(date=date, known_at=known_at, lower=low, upper=high,
                            adv=adv, source=source, eligible=0, filled=0)
        self.last = known
        if self.trigger and self.remaining:
            self._order(known_at)

    def _order(self, at):
        row = dict(order_id=f'{self.stock_id}:{self.venue}:{self.session["date"]}',
                   stock_id=self.stock_id, venue=self.venue, side='sell',
                   created_at=at, qty=self.remaining, limit_price=self.session['lower'],
                   reason='intraday_peak_stop15', trigger_at=self.trigger['at'],
                   state='research_intent_only', broker_submitted=False)
        self.orders.append(row)
        return deepcopy(row)

    def observe(self, *, at, stock_id, venue, basis, prints):
        """Emit intent immediately; only STRICTLY later groups may supply fills.

        Tied timestamps are a group, not an invented trade order. If a group
        could first set a peak then cross its stop, its order is ambiguous and
        rejected without mutation. Invalid feeds must pause the caller.
        """
        now = instant(at)
        if (not self.session or now <= self.last or now.date().isoformat() != self.session['date']
                or not '09:00:00' <= now.strftime('%H:%M:%S') < '13:25:00'
                or (stock_id, venue, basis) != (self.stock_id, self.venue, self.basis)):
            raise ValueError('Unordered, missing-session, identity or basis mismatch')
        if not prints:
            raise ValueError('Empty timestamp group')
        rows = []
        for row in prints:
            price, shares = positive(row['price']), row['shares']
            if (type(shares) is not int or shares <= 0 or shares % self.step
                    or not self.session['lower'] <= price <= self.session['upper']):
                raise ValueError('Invalid shares or price outside official session limits')
            rows.append((price, shares))
        low, high = min(p for p, _ in rows), max(p for p, _ in rows)
        new_peak = max(self.peak, high)
        if not self.trigger and low > self.peak*.85 + 1e-10 and low <= new_peak*.85 + 1e-10:
            raise ValueError('Ambiguous within-timestamp peak/stop order; provider sequence required')
        self.last = now
        if not self.remaining:
            return dict(intent=None, fill=None)
        if not self.trigger:
            if low <= self.peak*.85 + 1e-10:
                self.trigger = dict(at=at, peak=self.peak, threshold=self.peak*.85,
                                    observed_low=low, fill_price=None)
                return dict(intent=self._order(at), fill=None)
            self.peak = new_peak
            return dict(intent=None, fill=None)
        # Same-limit queue/depth is unknown: locked-down prints earn no capacity.
        eligible = [(p, q) for p, q in rows if p > self.session['lower']]
        self.session['eligible'] += sum(q for _, q in eligible)
        volume = min(self.session['eligible'], self.session['adv'])
        cap = int((Decimal(str(volume))*Decimal(str(self.participation))).to_integral_value(
            rounding=ROUND_FLOOR)) // self.step * self.step
        qty = min(self.remaining, max(0, cap-self.session['filled']))
        if not qty or not eligible:
            return dict(intent=None, fill=None)
        # Conservative group price, never the pre-trigger daily high/low midpoint.
        price = min(p for p, _ in eligible)
        paid = self._costs(price, qty, 'sell', self.stock_id)
        if paid['cash_change'] <= 0:
            return dict(intent=None, fill=None)
        fill = dict(at=at, stock_id=self.stock_id, venue=self.venue, qty=qty,
                    price=price, order_id=self.orders[-1]['order_id'], **paid)
        self.remaining -= qty
        self.session['filled'] += qty
        self.fills.append(fill)
        return dict(intent=None, fill=deepcopy(fill))

    def report(self):
        return deepcopy(dict(trigger=self.trigger, orders=self.orders, fills=self.fills,
            initial_qty=self.initial_qty, remaining_qty=self.remaining, peak=self.peak,
            status='closed_proxy' if not self.remaining else ('pending_exit' if self.trigger else 'watching'),
            net_sell_proceeds=sum(f['cash_change'] for f in self.fills),
            live_qualified=False, actual_fill_verified=False,
            fill_model='post_trigger_tape_participation_proxy', broker_submitted=False))


def replay(payload):
    """Deterministic adapter used by CLI; complete history is the checkpoint.

    Replaying a saved input is idempotent because no external orders are sent.
    Append new sessions/groups to that history to retain a pending stop.
    """
    engine = IntradayPeakExit(**payload['position'])
    for event in payload['events']:
        args = {k: v for k, v in event.items() if k != 'kind'}
        if event['kind'] == 'session':
            engine.start_session(**args)
        elif event['kind'] == 'prints':
            engine.observe(**args)
        else:
            raise ValueError('Unknown intraday event type')
    return engine.report()
