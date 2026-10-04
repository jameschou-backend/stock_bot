"""Disclosed whole-order exclusions when the strict matcher lacks usable data.

This is a post-hoc missing-data scenario, not evidence of executable fills.
The committed order, cash reservation and attempted slot survive an exclusion.
An unexecuted sale retains its actual shares and its original exit signal.
"""
from copy import deepcopy

from skills.poc_executable_replay import ExecutableOrders
from skills.replay_market_feeds import ReplayDataUnavailable


GAP_POLICY = 'user_authorized_whole_order_data_gap_exclusion_v1'
GAP_REASON = 'user_authorized_data_gap_exclusion'

# Only execution-side state is copied. Source caches, quote panels and the
# accumulated research journals are neither copied nor rewound. Each journal
# below is append-only inside ExecutableOrders._execute_order/cash_move.
_SCALARS = ('cash', 'day_cost', 'day_basis', 'active_budget', 'opening_remaining',
            'residual_spend_left', 'volatility_spend_left')
_MAPS = ('holdings', 'marks', 'used', 'markets', 'reservations')
_SETS = ('tick_attempts',)
_JOURNALS = ('orders', 'trades', 'cash_ledger')


def gap_key(row):
    return row.get('date'), row.get('stock_id'), row.get('side'), row.get('event_id')


def _snapshot(engine):
    return dict(values={name: deepcopy(getattr(engine, name))
                        for name in (*_SCALARS, *_MAPS, *_SETS) if hasattr(engine, name)},
                lengths={name: len(getattr(engine, name))
                         for name in _JOURNALS if hasattr(engine, name)})


def _restore_map(target, saved):
    # Keep nested holding/reservation identities: outer callers can still hold
    # references to these dictionaries while the matching method is executing.
    for key in list(target):
        if key not in saved:
            del target[key]
    for key, value in saved.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _restore_map(target[key], value)
        else:
            target[key] = deepcopy(value)


def _restore(engine, saved):
    for name, value in saved['values'].items():
        if name in _MAPS:
            _restore_map(getattr(engine, name), value)
        elif name in _SETS:
            getattr(engine, name).clear()
            getattr(engine, name).update(value)
        else:
            setattr(engine, name, value)
    for name, length in saved['lengths'].items():
        del getattr(engine, name)[length:]


def _failure_stage(error):
    """Use the exact strict matcher frame, without replacing its data adapters."""
    trace = error.__traceback__
    while trace:
        if trace.tb_frame.f_code is ExecutableOrders._execute_order.__code__:
            channel = trace.tb_frame.f_locals.get('channel')
            return channel if channel in ('board', 'odd') else 'execution_context'
        trace = trace.tb_next
    return 'execution_context'


def exclusion_order(record):
    """One zero event explicitly accounts for all positive child instructions."""
    return dict(date=record['date'], stock_id=record['stock_id'], side=record['side'],
        event_id=record['event_id'], signal_date=record['signal_date'], reason=record['reason'],
        channel='event', requested_qty=record['original_plan']['planned_qty'], filled_qty=0,
        failure=GAP_REASON, exclusion_policy=GAP_POLICY,
        failure_reason=record['failure_reason'], failure_stage=record['failure_stage'],
        excluded_children=deepcopy(record['excluded_children']),
        execution_evidence='not_executed_whole_order_data_exclusion',
        actual_fill_verified=False, live_qualified=False)


class DataGapOrders(ExecutableOrders):
    def __init__(self, *args, **kwargs):
        self.data_gap_exclusions = []
        super().__init__(*args, **kwargs)

    def _execute_order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        plan = self.day_plans[(event_id, side)]
        stamp = str(day.date())
        resolved_reason, resolved_signal = reason, signal_date
        if side == 'sell' and reason == 'scheduled_exit' and hasattr(self, 'exit_states'):
            state = self.exit_states.get(event_id)
            if not state or not state.get('trigger_reason'):
                raise ValueError('Exit needs a previously latched signal')
            resolved_reason, resolved_signal = state['trigger_reason'], state['signal_date']
        # A data exception must not conceal malformed plans, duplicate attempts,
        # same-day signals or changed resource quantities.
        key = stamp, sid, side, event_id
        quantities = [plan.get(k) for k in ('planned_qty', 'board_qty', 'odd_qty')]
        if (side not in ('buy', 'sell') or gap_key(plan) != key
                or any(type(n) is not int or n < 0 for n in quantities)
                or plan['planned_qty'] != plan['board_qty']+plan['odd_qty']
                or plan['board_qty'] % 1000 or plan['odd_qty'] >= 1000
                or type(qty) is not int or qty < plan['planned_qty']
                or not isinstance(resolved_signal, str) or resolved_signal >= stamp
                or plan['signal_date'] != resolved_signal or sid in self.tick_attempts
                or [p for p in self.tick_plans if gap_key(p) == key] != [plan]):
            raise ValueError('Data-gap execution differs from its unique committed plan')
        before = _snapshot(self)
        exit_before = deepcopy(getattr(self, 'exit_states', {}).get(event_id)) if side == 'sell' else None
        try:
            return super()._execute_order(day, sid, side, qty, reason, event_id, signal_date)
        except ReplayDataUnavailable as error:
            if not plan['planned_qty']:
                # No positive order exists to exclude; let a broken caller fail.
                raise
            failure_stage = _failure_stage(error)
            market = self.markets.get(sid)
            _restore(self, before)
            self.tick_attempts.add(sid)
            children = [dict(channel=channel, requested_qty=plan[channel+'_qty'])
                        for channel in ('board', 'odd') if plan[channel+'_qty']]
            record = dict(policy=GAP_POLICY, date=stamp, stock_id=sid, side=side,
                event_id=event_id, signal_date=resolved_signal, reason=resolved_reason,
                original_plan=deepcopy(plan), excluded_channels=[c['channel'] for c in children],
                excluded_children=children, failure_class=type(error).__name__,
                failure_reason=str(error), failure_stage=failure_stage, market=market,
                filled_qty=0, cash_before=before['values']['cash'], cash_after=self.cash,
                holding_qty_before=before['values']['holdings'].get(sid, {}).get('qty', 0),
                holding_qty_after=self.holdings.get(sid, {}).get('qty', 0),
                cash_ledger_length_before=before['lengths'].get('cash_ledger', 0),
                trade_count_before=before['lengths']['trades'],
                order_count_before=before['lengths']['orders'],
                exit_state_before=exit_before,
                exit_state_after=deepcopy(getattr(self, 'exit_states', {}).get(event_id)) if side == 'sell' else None,
                retry_sell=side == 'sell', posthoc_data_exclusion=True,
                source_verification_required_by_caller=True,
                actual_fill_verified=False, live_qualified=False)
            self.orders.append(exclusion_order(record))
            self.data_gap_exclusions.append(record)
            # Outer wrappers record an attempted zero fill and lock the full
            # unused daily budget. A failed sale's original due_index is intact.
            return 0

    def run(self):
        account = super().run()
        exclusions = deepcopy(self.data_gap_exclusions)
        children = [c for row in exclusions for c in row['excluded_children']]
        board = sum(c['channel'] == 'board' for c in children)
        odd = sum(c['channel'] == 'odd' for c in children)
        account['data_gap_exclusions'] = exclusions
        account['settings'].update(data_gap_policy=GAP_POLICY,
            # The strict after-hours matcher uses 1%; replace the inactive 5%
            # field inherited from the older mixed-odd daily-price simulator.
            odd_participation=.01,
            missing_ordinary_policy='exclude_whole_order_keep_reservation_and_retry_sales',
            missing_odd_policy='exclude_whole_order_keep_reservation_and_retry_sales',
            posthoc_data_exclusion=True, excluded_event_count=len(exclusions),
            excluded_board_children=board, excluded_odd_children=odd,
            actual_fill_verified=False, live_qualified=False, unseen_validation=False)
        volume = account['ordinary_volume_evidence']
        volume.update(excluded_positive_board_children=board,
            original_requested_board_children=volume['requested_board_children']+board,
            all_original_requested_board_capacity_observed=(not board and
                volume['all_requested_board_capacity_observed']))
        return account
