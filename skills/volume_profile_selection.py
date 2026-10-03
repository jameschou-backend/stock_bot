"""Causal, lazy POC ordering before a frozen account reserves its buy orders.

The caller supplies a private deterministic reservation planner, never the live
account. Its inputs must be known before execution. ``probe`` is read-only;
``consider`` applies the exact same reservation and never consults fills. Cash
and free slots may only decrease, so a failed probe cannot become admissible
later in this planning pass. ``can_continue == False`` must certify that no
remaining candidate can reserve resources. A filled quantity is not that proof.

For priority, true profiles are visited in original order. False profiles wait
until the entire possibly admissible suffix has been resolved. Once reservations
exhaust resources, every unqueried suffix completion produces the same buy plans.
This is an execution-plan prefix certificate, not a claim that every profile or
the complete ranking was observed. Unknown necessary profiles always raise.
"""
from copy import deepcopy
from datetime import date
import math
from collections.abc import Mapping


ARMS = ('original', 'poc_priority', 'poc_filter')


class UnresolvedProfile(ValueError):
    """A necessary flag is unknown; a caller may not publish a complete account."""

    def __init__(self, event_id, reason, recoverable, certificate):
        self.event_id = event_id
        self.reason = reason
        self.recoverable = recoverable
        self.certificate = deepcopy(certificate)
        super().__init__(f'Unresolved POC profile for {event_id}: {reason}')


def _events(events):
    rows = deepcopy(list(events))
    ids, signal_dates = set(), set()
    for row in rows:
        eid, score = row['event_id'], row['priority']
        if not isinstance(eid, str) or not eid or eid in ids:
            raise ValueError('Unique nonempty candidate event identities required')
        ids.add(eid)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError('Every candidate needs a finite original priority')
        if len(row['members']) != 1 or not isinstance(row['members'][0], str):
            raise ValueError('A candidate must identify one stock')
        signal = date.fromisoformat(row['signal_date'])
        if row.get('entry_date') and date.fromisoformat(row['entry_date']) <= signal:
            raise ValueError('Profile ordering must precede the entry session')
        signal_dates.add(signal)
    if len(signal_dates) > 1:
        raise ValueError('One signal session per planning pass required')
    ordered = sorted(rows, key=lambda r: (-r['priority'], r['event_id']))
    if [r['event_id'] for r in rows] != [r['event_id'] for r in ordered]:
        raise ValueError('Input candidates must retain original priority/event-id order')
    return rows


def select_candidates(events, *, arm, provider, planner):
    """Return ``events`` admitted to planning and a complete diagnostic certificate.

    Planner: snapshot()->mapping, can_continue()->bool, probe(event)->mapping,
    consider(event)->same mapping. Probe/consider include ``attempted: bool`` and
    may include allocation, budget, raw_qty, failure, and other audit fields.
    Provider: available bool, poc_up bool when known, reason/recoverable when not.
    No profile check is needed for an original arm, an exhausted resource state,
    or a candidate proven permanently unreservable by the causal planner.
    """
    if arm not in ARMS:
        raise ValueError('Unregistered POC selection arm')
    rows = _events(events)
    initial = deepcopy(planner.snapshot())
    decisions = [dict(event_id=r['event_id'], stock_id=r['members'][0],
        signal_date=r['signal_date'], original_rank=i + 1, priority=r['priority'],
        profile_status='not_requested', selection_status='not_processed', reason=None)
        for i, r in enumerate(rows)]
    cert = dict(schema='volume_profile_selection_v1', arm=arm, complete=False,
        original_ordered_ids=[r['event_id'] for r in rows], initial_state=initial,
        final_state=deepcopy(initial), decisions=decisions, requested_profiles=[],
        reserved_event_ids=[], stop_reason=None, failure_classification=None,
        profile_ranking_complete=False, execution_prices_used_for_selection=False)
    if arm == 'original':
        for d in decisions:
            d.update(profile_status='not_needed_original', selection_status='original_unchanged')
        cert.update(complete=True, stop_reason='original_unchanged')
        return dict(events=rows, certificate=cert)

    selected, deferred = [], []

    def snapshot():
        value = planner.snapshot()
        if not isinstance(value, Mapping):
            raise ValueError('Planner snapshot must be a mapping')
        return deepcopy(dict(value))

    def can_continue():
        before = snapshot()
        value = planner.can_continue()
        if type(value) is not bool or snapshot() != before:
            raise ValueError('Resource exhaustion check must be a read-only bool')
        return value

    def probe(event):
        before = snapshot()
        result = planner.probe(deepcopy(event))
        if (not isinstance(result, Mapping) or type(result.get('attempted')) is not bool
                or snapshot() != before):
            raise ValueError('Reservation probe must be read-only with boolean attempted')
        return deepcopy(dict(result))

    def reserve(event, decision, expected):
        before = snapshot()
        result = planner.consider(deepcopy(event))
        if result != expected or not expected['attempted']:
            raise ValueError('Applied reservation differs from its causal probe')
        after = snapshot()
        if before == after:
            raise ValueError('An attempted reservation must consume resources')
        decision.update(selection_status='reserved', reservation=deepcopy(result),
                        state_before=before, state_after=after)
        cert['reserved_event_ids'].append(event['event_id'])
        selected.append(event)

    stopped = False
    for row, decision in zip(rows, decisions):
        if not can_continue():
            stopped = True
            break
        result = probe(row)
        decision['probe'] = result
        if not result['attempted']:
            decision.update(profile_status='not_needed_unreservable',
                selection_status='not_reserved', reason=result.get('failure') or 'zero_raw_quantity')
            continue
        before = snapshot()
        profile = provider(deepcopy(row))
        if snapshot() != before:
            raise ValueError('Profile lookup changed the reservation planner')
        if not isinstance(profile, Mapping) or type(profile.get('available')) is not bool:
            raise ValueError('Profile provider must return an explicit availability bool')
        if profile.get('signal_date', row['signal_date']) != row['signal_date']:
            raise ValueError('Profile signal date differs from requested candidate')
        if profile.get('stock_id', row['members'][0]) != row['members'][0]:
            raise ValueError('Profile stock differs from requested candidate')
        if profile.get('event_id', row['event_id']) != row['event_id']:
            raise ValueError('Profile event differs from requested candidate')
        known = profile['available'] and type(profile.get('poc_up')) is bool
        request = dict(event_id=row['event_id'], stock_id=row['members'][0],
            signal_date=row['signal_date'], original_rank=decision['original_rank'],
            available=bool(known), poc_up=profile.get('poc_up') if known else None,
            reason=profile.get('reason'), recoverable=profile.get('recoverable'))
        cert['requested_profiles'].append(request)
        if not known:
            reason = profile.get('reason') or 'profile_flag_unavailable'
            decision.update(profile_status='unknown', selection_status='unresolved', reason=reason)
            cert.update(final_state=snapshot(), stop_reason='necessary_profile_unknown',
                failure_classification=dict(event_id=row['event_id'], reason=reason,
                                            recoverable=profile.get('recoverable')))
            raise UnresolvedProfile(row['event_id'], reason, profile.get('recoverable'), cert)
        if profile['poc_up']:
            decision['profile_status'] = 'known_true'
            reserve(row, decision, result)
        else:
            decision['profile_status'] = 'known_false'
            decision['selection_status'] = 'hard_filter_rejected' if arm == 'poc_filter' else 'deferred_false'
            deferred.append((row, decision))

    # Only a completely scanned suffix proves that no unqueried true candidate
    # can outrank these false candidates. No false is admitted before this point.
    if not stopped and arm == 'poc_priority':
        for row, decision in deferred:
            if not can_continue():
                stopped = True
                break
            result = probe(row)
            decision['fallback_probe'] = result
            if result['attempted']:
                reserve(row, decision, result)
            else:
                decision.update(selection_status='not_reserved',
                                reason=result.get('failure') or 'zero_raw_quantity')
    for decision in decisions:
        if decision['selection_status'] == 'not_processed':
            decision.update(profile_status='not_needed_resource_exhausted',
                            selection_status='not_needed_resource_exhausted')
        elif decision['selection_status'] == 'deferred_false':
            decision['selection_status'] = 'not_needed_resource_exhausted'
    cert.update(complete=True, final_state=snapshot(),
        stop_reason='resource_exhausted' if stopped else 'candidates_exhausted',
        profile_ranking_complete=all(d['profile_status'] in ('known_true', 'known_false') for d in decisions))
    return dict(events=selected, certificate=cert)
