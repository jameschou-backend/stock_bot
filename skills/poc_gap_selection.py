"""Explicit candidate exclusions for otherwise unresolved POC data.

This is a different research policy, not an inference that an unknown POC is
false. The sealed selector and its permanent-quality fallback remain intact.
Each pass gets a fresh private reservation planner. A profile is requested at
most once per event in this invocation; rebuilding never repeats network work.
"""
from collections.abc import Mapping
from copy import deepcopy

from skills.volume_profile_account_adapter import QUALITY_REASONS, ReservationPlanner
from skills.volume_profile_selection import ARMS, UnresolvedProfile, _events, select_candidates


POLICY = 'exclude_unresolved_profile_candidate_preserve_quality_fallback_v1'


def _snapshot(planner):
    value = planner.snapshot()
    if not isinstance(value, Mapping):
        raise ValueError('POC gap planner snapshot must be a mapping')
    return deepcopy(dict(value))


def select_with_profile_gaps(events, *, arm, provider, planner, planner_factory=None):
    """Return the normal events/certificate with explicit data exclusions.

    The optional test seam ``planner_factory(source_planner)`` must return a
    NEW planner in the identical opening state. Production uses exactly
    ``ReservationPlanner(planner.engine, planner.day)``. The caller's planner
    is never considered/reserved. Only structured ``UnresolvedProfile`` from
    the sealed selector is handled; arbitrary provider/program errors still
    propagate. Existing nonrecoverable QUALITY_REASONS are re-raised unchanged
    so AccountCandidateHook's ``_available`` fallback continues to work.
    """
    if arm not in ARMS:
        raise ValueError('Unregistered POC gap selection arm')
    rows = _events(events)
    initial = _snapshot(planner)
    ranks = {r['event_id']:i+1 for i,r in enumerate(rows)}
    by_id = {r['event_id']:r for r in rows}
    factory = planner_factory or (lambda source: ReservationPlanner(source.engine, source.day))
    remaining, exclusions, excluded_decisions, cached_profiles = rows, [], {}, {}
    attempted_planners = []

    def pristine():
        if _snapshot(planner) != initial:
            raise ValueError('POC gap selection changed the caller reservation planner')

    def fresh():
        pristine()
        candidate = factory(planner)
        if candidate is planner or any(candidate is old for old in attempted_planners):
            raise ValueError('POC gap retries require a fresh reservation planner')
        if _snapshot(candidate) != initial:
            raise ValueError('POC gap retry opening reservation state differs')
        attempted_planners.append(candidate)
        return candidate

    def profile(event):
        eid = event['event_id']
        if eid not in by_id:
            raise ValueError('POC gap provider asked for an unregistered candidate')
        if eid not in cached_profiles:
            cached_profiles[eid] = deepcopy(provider(deepcopy(event)))
        return deepcopy(cached_profiles[eid])

    def annotate_error(exc):
        # A permanent-quality fallback restores the entire original list in
        # AccountCandidateHook. Earlier provisional exclusions then do NOT
        # describe the final day's entries; retain that distinction explicitly.
        exc.certificate.update(profile_gap_policy=POLICY,
            profile_data_exclusions=[dict(r,applied=False) for r in exclusions],
            profile_data_exclusions_applied=False,
            gap_original_ordered_ids=[r['event_id'] for r in rows],
            profile_provider_event_ids=list(cached_profiles),
            selection_passes=len(attempted_planners), source_planner_unchanged=True)

    if arm == 'original':
        result = select_candidates(rows,arm=arm,provider=provider,planner=planner)
        pristine()
        result['certificate'].update(profile_gap_policy=POLICY,profile_data_exclusions=[],
            profile_data_exclusions_applied=False,profile_provider_event_ids=[],
            selection_passes=1,source_planner_unchanged=True)
        return result

    while True:
        active = fresh()
        try:
            result = select_candidates(remaining,arm=arm,provider=profile,planner=active)
        except UnresolvedProfile as exc:
            pristine()
            if exc.recoverable is False and exc.reason in QUALITY_REASONS:
                annotate_error(exc)
                raise  # Preserve the existing hook's precise fallback contract.
            if (exc.event_id not in {r['event_id'] for r in remaining}
                    or exc.event_id not in cached_profiles):
                raise  # A provider-raised exception is not a validated missing flag.
            decision = next((d for d in exc.certificate.get('decisions', [])
                             if d.get('event_id') == exc.event_id),None)
            if (decision is None or decision.get('profile_status') != 'unknown'
                    or decision.get('selection_status') != 'unresolved'
                    or exc.certificate.get('initial_state') != initial):
                raise ValueError('POC gap unresolved exception lacks a valid selector certificate') from exc
            row = by_id[exc.event_id]
            exclusions.append(dict(event_id=exc.event_id,stock_id=row['members'][0],
                signal_date=row['signal_date'],original_rank=ranks[exc.event_id],
                reason=exc.reason,recoverable=exc.recoverable,available=False,poc_up=None,
                policy=POLICY,applied=True))
            excluded_decisions[exc.event_id] = dict(deepcopy(decision),
                original_rank=ranks[exc.event_id],selection_status='profile_data_excluded',
                available=False,poc_up=None,exclusion_policy=POLICY)
            remaining = [r for r in remaining if r['event_id'] != exc.event_id]
            continue
        pristine()
        # Confirm provider/planner work did not alter the live opening inputs.
        check = factory(planner)
        if check is planner or any(check is old for old in attempted_planners) or _snapshot(check) != initial:
            raise ValueError('POC gap selection changed the live opening reservation state')
        cert = result['certificate']
        retained = list(cert['original_ordered_ids'])
        decisions = {d['event_id']:dict(deepcopy(d),original_rank=ranks[d['event_id']])
                     for d in cert['decisions']}
        decisions.update(excluded_decisions)
        cert['decisions'] = [decisions[r['event_id']] for r in rows]
        requests = []
        for eid, value in cached_profiles.items():
            row = by_id[eid]
            known = value['available'] and type(value.get('poc_up')) is bool
            requests.append(dict(event_id=eid,stock_id=row['members'][0],signal_date=row['signal_date'],
                original_rank=ranks[eid],available=bool(known),poc_up=value.get('poc_up') if known else None,
                reason=value.get('reason'),recoverable=value.get('recoverable')))
        cert.update(original_ordered_ids=[r['event_id'] for r in rows],
            retained_ordered_ids=retained,requested_profiles=requests,
            profile_gap_policy=POLICY,profile_data_exclusions=deepcopy(exclusions),
            profile_data_exclusions_applied=bool(exclusions),profile_provider_event_ids=list(cached_profiles),
            selection_passes=len(attempted_planners),source_planner_unchanged=True,
            profile_ranking_complete=cert['profile_ranking_complete'] and not exclusions)
        return result
