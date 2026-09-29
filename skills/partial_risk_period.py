"""Period contracts for the frozen stock selector; never relabel a shorter run."""
from datetime import date


def validate_period(start, end, *, selector_start, selector_end, identity_start,
                    identity_end, sessions):
    for value in (start, end, selector_start, selector_end, identity_start, identity_end):
        date.fromisoformat(value)
    if start > end:
        raise ValueError('Start follows end')
    if start < max(selector_start, identity_start) or end > min(selector_end, identity_end):
        raise ValueError('Requested period exceeds verified selector/identity coverage')
    available = set(sessions)
    if start not in available or end not in available:
        raise ValueError('Requested endpoints are not audited market sessions')
    return [d for d in sessions if start <= d <= end]


def reconcile_runs(left, right):
    if left['start'] != right['start'] or left['end'] != right['end']:
        raise ValueError('Comparison periods differ')
    if left['source_sha256'] != right['source_sha256']:
        raise ValueError('Comparison sources differ')
    for report in (left, right):
        if report['preparation'] or not report['all_completed']:
            raise ValueError('Two complete offline runs required')
        if set(report['cases']) != {'original', 'cap40', 'benchmark'}:
            raise ValueError('Missing fixed comparison arm')
