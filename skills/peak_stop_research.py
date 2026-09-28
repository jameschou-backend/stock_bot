"""Unarmed 15% peak stops; separate trigger evidence from execution claims."""
from dataclasses import dataclass
import math

from skills.midpoint_replay import MidpointStock


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Price must be finite and positive')
    return float(value)


@dataclass
class PeakStop:
    """Ordered observations after a known fill; no fills inferred from triggers.

    Times are strictly increasing integer event ordinals, supplied by a caller
    with an audited ordering. An identical timestamp needs provider sequence
    evidence; sorting tied timestamps arbitrarily is not valid input.
    Prices must share one split/dividend-adjusted basis. This class does not
    infer corporate actions or claim that tape volume is executable depth.
    """
    peak: float
    last_sequence: int
    trigger: dict | None = None

    def __post_init__(self):
        self.peak = positive(self.peak)
        if type(self.last_sequence) is not int:
            raise ValueError('Known entry event sequence required')
        if self.trigger is not None:
            raise ValueError('New position must start without a trigger')

    def observe(self, sequence, price):
        price = positive(price)
        if type(sequence) is not int or sequence <= self.last_sequence:
            raise ValueError('Observations must follow the fill and be strictly ordered')
        self.last_sequence = sequence
        if self.trigger is not None:
            return dict(self.trigger)
        self.peak = max(self.peak, price)
        threshold = self.peak * .85
        if price <= threshold + 1e-10:
            self.trigger = dict(sequence=sequence, peak=self.peak, threshold=threshold,
                                observed_price=price, fill_price=None)
            return dict(self.trigger)
        return None


def bar_status(peak, opening, high, low, close):
    """Classify a FULL held-session bar without guessing its high/low order.

    Any price <= prior peak threshold is certain. If only today's new high
    raises the threshold above the low, close <= that threshold proves a later
    crossing; otherwise both hit and no-hit paths can share this OHLC bar.
    Never call this on an entry session without its post-fill price path.
    """
    peak, opening, high, low, close = map(positive, (peak, opening, high, low, close))
    if not low <= min(opening, close) <= max(opening, close) <= high:
        raise ValueError('Inconsistent OHLC')
    initial = max(peak, opening)
    raised = max(initial, high)
    if low <= initial * .85 + 1e-10 or close <= raised * .85 + 1e-10:
        status = 'certain_trigger'
    elif low <= raised * .85 + 1e-10:
        status = 'intraday_order_ambiguous'
    else:
        status = 'no_trigger'
    return dict(status=status, next_peak=raised, threshold=raised*.85)


class ClosePeakStopReplay(MidpointStock):
    """Explicit daily-close proxy, not the requested intraday execution model."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = 'fixed63'  # Replace loss12, retain the unchanged expiry.
        self.peak_decisions = []

    def corporate_day(self, day):
        income = super().corporate_day(day)
        i = self.positions[day]
        for sid, holding in self.holdings.items():
            state = self.exit_states.get(holding['event_id'])
            if not holding['qty'] or not state or state['trigger_reason']:
                continue
            context = self.exit_signals.context(i, sid, state)
            drawdown = context['peak_drawdown']
            reason = 'close_peak_stop15' if drawdown is not None and drawdown <= -.15+1e-12 else None
            signal = str(self.days[i-1].date())
            self.peak_decisions.append(dict(date=str(day.date()), signal_date=signal,
                event_id=holding['event_id'], stock_id=sid, peak=state['peak_price'],
                close=self.exit_signals.price(i-1, sid), reason=reason))
            if reason:
                state.update(trigger_reason=reason, signal_date=signal,
                    target_date=str(day.date()), target_index=i)
                holding['due_index'] = i
                self._plan(day, sid, 'sell', holding['event_id'], signal,
                    holding['qty']//1000*1000, 0., self.opening_limit, None)
        return income

    def run(self):
        result = super().run()
        result['settings'].update(exit_mechanism='close_peak_stop15_or_time63',
            intraday_stop=False, stop_profit_arm=None, stop_drawdown=.15,
            stop_signal='prior_close_vs_highest_close_since_entry')
        return result
