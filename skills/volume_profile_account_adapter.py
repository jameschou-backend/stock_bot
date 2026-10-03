"""Candidate-only hook before the sealed account reserves opening resources."""
from copy import deepcopy
import math

from skills.five_axis_replay import FiveAxisReplay
from skills.million_replay import money

QUALITY_REASONS = frozenset({
    'corporate_action_or_nonconstant_price_scale', 'pre_signal_daily_path_invalid',
    'official_identity_missing_or_ambiguous', 'ordinary_tape_conflict',
})


class ReservationPlanner:
    """Pure copy of TickPlanning's attempted-order reservation decision.

    It reads only prior price/liquidity and the opening account. The board/odd
    quantity calculation and actual execution remain in the sealed engine.
    """
    def __init__(self, engine, day):
        self.engine, self.day = engine, day
        self.cash = min(engine.opening_limit, engine.cash)
        self.occupied = set(engine.opening_members) - engine.released_residuals
        self.held = set(engine.holdings)

    def snapshot(self):
        return dict(cash=self.cash, occupied=sorted(self.occupied),
                    held=sorted(self.held), slots=self.engine.slots,
                    opening_cash=self.engine.opening_limit,
                    previous_nav=self.engine.previous_nav,
                    residual_budget=self.engine.residual_budget,
                    residual_block=self.engine.residual_block)

    def can_continue(self):
        return (len(self.occupied) < self.engine.slots
                and not self.engine.residual_block and self.cash > 40)

    def probe(self, event):
        sid = event['members'][0]
        price = self.engine.prior(self.day, sid)
        amount = float(self.engine.amount20.at[self.day, sid])
        allocation = min(self.cash, self.engine.previous_nav/self.engine.slots)
        budget = min(allocation, self.engine.residual_budget)
        failure = None
        if sid in self.held or sid in self.occupied or len(self.occupied) >= self.engine.slots:
            failure = 'opening_slots_locked'
        elif self.engine.residual_block:
            failure = 'residual_exposure_cap'
        elif not math.isfinite(amount) or amount < 50e6:
            failure = 'prior_liquidity_below_50m_or_missing'
        raw_qty = max(0, math.floor((allocation-40)/(price*(1+.001425+.0045)))) if price else 0
        return dict(attempted=failure is None and raw_qty > 0,
                    failure=failure, allocation=allocation, budget=budget,
                    raw_qty=raw_qty, stock_id=sid, event_id=event['event_id'])

    def consider(self, event):
        result = self.probe(event)
        if result['attempted']:
            self.cash = money(self.cash-result['allocation'])
            self.occupied.add(result['stock_id'])
        return result


class AccountCandidateHook(FiveAxisReplay):
    """MRO seam: after FiveAxis ordering and before TickPlanning reservation.

    The baseline never queries profiles or mutates its original event sequence.
    A caller can supply a frozen selector/provider for separately preregistered
    POC arms. This class does not fetch data or alter execution assumptions.
    """
    def __init__(self, *args, selection_arm='original', profile_provider=None,
                 candidate_selector=None, **kwargs):
        self.selection_arm = selection_arm
        self.profile_provider = profile_provider
        self.candidate_selector = candidate_selector
        self.selection_decisions = []
        if selection_arm != 'original' and (profile_provider is None or candidate_selector is None):
            raise ValueError('POC account arms require explicit provider and selector')
        super().__init__(*args, **kwargs)

    def corporate_day(self, day):
        income = super().corporate_day(day)
        events = self.events.get(day, [])
        if not events:
            return income
        planner = ReservationPlanner(self, day)
        context = dict(date=str(day.date()), arm=self.selection_arm,
                       original_event_ids=[e['event_id'] for e in events],
                       candidate_count=len(events), initial_state=planner.snapshot(),
                       opening_can_reserve=planner.can_continue())
        self.selection_decisions.append(context)
        if self.selection_arm == 'original':
            # Observation only; no reservation or profile computation occurs.
            context['selected_event_ids'] = list(context['original_event_ids'])
            return income
        from skills.volume_profile_selection import UnresolvedProfile
        core_arm = self.selection_arm.removesuffix('_available')
        try:
            result = self.candidate_selector(events, arm=core_arm,
                provider=self.profile_provider, planner=planner)
        except UnresolvedProfile as exc:
            context['certificate'] = deepcopy(exc.certificate)
            allowed = (self.selection_arm.endswith('_available')
                       and exc.recoverable is False and exc.reason in QUALITY_REASONS)
            context['fallback_to_original'] = allowed
            if not allowed:
                raise
            context['fallback_reason'] = exc.reason
            context['discarded_reservation_event_ids'] = list(exc.certificate['reserved_event_ids'])
            context['reset_state'] = ReservationPlanner(self, day).snapshot()
            if context['reset_state'] != context['initial_state']:
                raise ValueError('POC failure changed the live opening account')
            # The selector changed only its private planner. Restore the entire
            # original list; TickPlanning now starts its original reservations.
            self.events[day] = events
            context['selected_event_ids'] = list(context['original_event_ids'])
            return income
        context['certificate'] = deepcopy(result['certificate'])
        self.events[day] = result['events']
        context['selected_event_ids'] = [e['event_id'] for e in result['events']]
        return income

    def run(self):
        result = super().run()
        result['selection_decisions'] = deepcopy(self.selection_decisions)
        return result


class AccountReservationAudit:
    """Compare the lazy certificate to actual precommitted plans, before fills."""
    def corporate_day(self, day):
        income = super().corporate_day(day)
        contexts = getattr(self, 'selection_decisions', [])
        context = contexts[-1] if contexts and contexts[-1]['date'] == str(day.date()) else None
        if context is None or context['arm'] == 'original' or context.get('fallback_to_original'):
            return income
        expected = context['certificate']['reserved_event_ids']
        plans = [p for p in self.day_plans.values() if p['side'] == 'buy' and p['reserved_cash'] > 0]
        if [p['event_id'] for p in plans] != expected:
            raise ValueError('Lazy POC reservations differ from actual account plans')
        decisions = {d['event_id']: d for d in context['certificate']['decisions']}
        for plan in plans:
            allocation = decisions[plan['event_id']]['reservation']['allocation']
            if abs(plan['reserved_cash'] - allocation) > .005:
                raise ValueError('Lazy POC budget differs from actual account plan')
        context['actual_reservations_verified'] = True
        return income
