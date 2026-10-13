"""Causal candidate admission experiment; the frozen cash engine stays intact."""
from copy import deepcopy


def cooldown_candidates(entries, calendar, sessions=5):
    if type(sessions) is not int or sessions != 5:
        raise ValueError('Only preregistered five-session cooldown is supported')
    if len(set(calendar)) != len(calendar) or list(calendar) != sorted(calendar):
        raise ValueError('Cooldown requires a unique ordered market calendar')
    index = {day: i for i, day in enumerate(calendar)}
    identities = [row['event_id'] for row in entries]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate candidate identity')
    retained, last, decisions = set(), {}, []
    # Stable order breaks same-day ties without altering the original ranking.
    for row in sorted(entries, key=lambda r: r['signal_date']):
        if len(row['members']) != 1:
            raise ValueError('Cooldown expects single-stock candidates')
        sid, signal, entry = row['members'][0], row['signal_date'], row['entry_date']
        if signal not in index or entry not in index or index[entry] != index[signal] + 1:
            raise ValueError('Candidate needs a known next-session entry date')
        previous = last.get(sid)
        gap = index[signal] - index[previous] if previous else None
        allowed = previous is None or gap > sessions
        decisions.append(dict(event_id=row['event_id'], stock_id=sid, signal_date=signal,
            previous_retained_signal=previous, gap_sessions=gap, allowed=allowed))
        if allowed:
            last[sid] = signal
            retained.add(row['event_id'])
    return [deepcopy(r) for r in entries if r['event_id'] in retained], decisions


def audit_cooldown(entries, selected, decisions, calendar):
    """Independent prefix oracle, checking skipped signals do not reset the clock."""
    positions = {d: i for i, d in enumerate(calendar)}
    source = {r['event_id']: r for r in entries}
    rows = {r['event_id']: r for r in decisions}
    if len(source) != len(entries) or set(rows) != set(source) or len(rows) != len(decisions):
        raise ValueError('Cooldown decision coverage differs')
    approved = []
    for candidate in sorted(entries, key=lambda r: r['signal_date']):
        sid, signal = candidate['members'][0], candidate['signal_date']
        prior = [r for r in approved if r['members'][0] == sid]
        previous = prior[-1]['signal_date'] if prior else None
        gap = positions[signal] - positions[previous] if previous else None
        expected = dict(event_id=candidate['event_id'], stock_id=sid, signal_date=signal,
            previous_retained_signal=previous, gap_sessions=gap, allowed=gap is None or gap > 5)
        if rows[candidate['event_id']] != expected:
            raise ValueError('Cooldown decision differs from causal prefix oracle')
        if expected['allowed']:
            approved.append(candidate)
    ids = {r['event_id'] for r in approved}
    if selected != [r for r in entries if r['event_id'] in ids]:
        raise ValueError('Cooldown changed retained candidate identity, ranking or order')
    return dict(input_candidates=len(entries), retained_candidates=len(selected),
        suppressed_candidates=len(entries)-len(selected), independent_prefix_checked=True)


def install_candidates(runner, calendar, metadata):
    original_select, original_write = runner.select_arm_entries, runner.write
    evidence = {}

    def select(arm, *args, **kwargs):
        entries, gates, first = original_select(arm, *args, **kwargs)
        selected, decisions = cooldown_candidates(entries, calendar)
        audit = audit_cooldown(entries, selected, decisions, calendar)
        evidence[arm] = dict(decisions=decisions, audit=audit,
            approved={r['event_id'] for r in selected})
        return selected, gates, first

    def write(path, value):
        if 'family_rules' in value:
            arm = path.stem
            item = evidence[arm]
            if value.get('completed'):
                funded = {c['event_id'] for c in value['account']['cohorts']}
                if not funded <= item['approved']:
                    raise ValueError('Account funded a suppressed candidate')
            value['optimization'] = dict(deepcopy(metadata), **item['audit'])
            value['cooldown_decisions'] = item['decisions']
        elif path.name == 'report.json':
            value['optimization'] = deepcopy(metadata)
        return original_write(path, value)
    runner.select_arm_entries, runner.write = select, write
