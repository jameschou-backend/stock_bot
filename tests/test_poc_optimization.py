from copy import deepcopy
import pytest

from skills.poc_optimization import cooldown_candidates, audit_cooldown

CALENDAR = ['2026-10-%02d' % n for n in (1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15)]


def event(index, sid='1101', priority=3):
    return dict(event_id=f'{sid}-{index}', signal_date=CALENDAR[index],
        entry_date=CALENDAR[index+1], members=[sid], priority=priority)


def test_calendar_cooldown_skips_weekends_and_rejections_do_not_restart_clock():
    entries = [event(i) for i in range(9)]
    selected, decisions = cooldown_candidates(entries, CALENDAR)
    assert [r['event_id'] for r in selected] == ['1101-0', '1101-6']
    assert decisions[5]['gap_sessions'] == 5 and not decisions[5]['allowed']
    assert decisions[6]['previous_retained_signal'] == CALENDAR[0]
    assert audit_cooldown(entries, selected, decisions, CALENDAR)['suppressed_candidates'] == 7


def test_prefix_causal_independent_stocks_and_preserved_ranking():
    entries = [event(0, '1101', 7), event(0, '2330', 9), event(4, '1101'),
               event(6, '1101'), event(8, '2330')]
    original = deepcopy(entries)
    full, _ = cooldown_candidates(entries, CALENDAR)
    for stop in range(1, len(entries)+1):
        selected, decisions = cooldown_candidates(entries[:stop], CALENDAR)
        assert selected == [r for r in full if r in entries[:stop]]
        audit_cooldown(entries[:stop], selected, decisions, CALENDAR)
    assert entries == original
    assert [r['priority'] for r in full[:2]] == [7, 9]


def test_independent_oracle_rejects_mutated_decisions_and_priority():
    entries = [event(0), event(2), event(6)]
    selected, decisions = cooldown_candidates(entries, CALENDAR)
    decisions[1]['allowed'] = True
    with pytest.raises(ValueError, match='prefix oracle'):
        audit_cooldown(entries, selected, decisions, CALENDAR)
    selected, decisions = cooldown_candidates(entries, CALENDAR)
    selected[0]['priority'] += 1
    with pytest.raises(ValueError, match='ranking'):
        audit_cooldown(entries, selected, decisions, CALENDAR)


@pytest.mark.parametrize('mutation', ['calendar', 'duplicate', 'same_day', 'unknown_day'])
def test_bad_timing_is_not_a_silent_fallback(mutation):
    entries, calendar = [event(0)], CALENDAR.copy()
    if mutation == 'calendar': calendar.reverse()
    if mutation == 'duplicate': entries += deepcopy(entries)
    if mutation == 'same_day': entries[0]['entry_date'] = entries[0]['signal_date']
    if mutation == 'unknown_day': entries[0]['signal_date'] = '2026-09-30'
    with pytest.raises(ValueError): cooldown_candidates(entries, calendar)
