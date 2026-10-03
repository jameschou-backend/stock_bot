from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts.research_staged_entry_20261003 import (
    comparison, observe_second_fill, scope_comparisons, staged_decision, staged_row,
    verify_fixed_support,
)
from skills.independent_signals import net_unit_return
from skills.independent_three_black import ThreeBlackPath, observe


ENTRY = 62


def path(n=150):
    c = np.full(n, 100.)
    return ThreeBlackPath(pd.bdate_range('2024-01-01', periods=n), c.copy(), c.copy(),
                          np.ones(n, bool), c.copy(), c + 2, c - 2, c * 1000, c.copy())


def set_bar(p, index, close, opened=None, high=None, low=None, factor=1.):
    opened = close if opened is None else opened
    high = max(close, opened) + 2 if high is None else high
    low = min(close, opened) - 2 if low is None else low
    p.close[index] = p.other[index] = close
    p.raw_close[index], p.opened[index] = close / factor, opened / factor
    p.high[index], p.low[index] = high / factor, low / factor


def original(p):
    return dict(observe(p, ENTRY), signal_id='e', stock_id='1234', name='公司',
                signal_date=str(p.days[ENTRY - 1].date()), entry_date=str(p.days[ENTRY].date()))


def confirmation_path(n=150):
    p = path(n)
    for index, value in ((ENTRY, 101), (ENTRY + 1, 102), (ENTRY + 2, 104), (ENTRY + 3, 106)):
        if index < n:
            set_bar(p, index, value)
    return p


def truncate(p, length):
    return replace(p, days=p.days[:length], **{
        field: getattr(p, field)[:length].copy()
        for field in ('close', 'other', 'eligible', 'raw_close', 'high', 'low', 'volume', 'opened')})


def test_confirmation_uses_third_holding_close_and_adds_fourth_day_only():
    p = confirmation_path()
    decision = staged_decision(p, ENTRY)
    assert decision['add_decision'] == 'add'
    assert decision['add_decision_known_at'] == str(p.days[ENTRY + 2].date())
    assert decision['fixed_support'] == 100
    fill = observe_second_fill(p, ENTRY, decision)
    assert fill['second_entry_date'] == str(p.days[ENTRY + 3].date())
    assert fill['second_entry_adjusted_hl2'] == 106


def test_future_mutations_and_truncation_do_not_change_confirmation():
    p = confirmation_path()
    decision = staged_decision(p, ENTRY)
    assert staged_decision(truncate(p, ENTRY + 3), ENTRY) == decision
    for field in ('close', 'other', 'raw_close', 'high', 'low', 'volume', 'opened'):
        getattr(p, field)[ENTRY + 3:] = np.nan
    p.eligible[ENTRY + 3:] = False
    assert staged_decision(p, ENTRY) == decision
    assert observe_second_fill(p, ENTRY, decision)['add_fill_status'] == 'unresolved'


def test_confirmation_boundary_includes_support_but_strictly_exceeds_first_fill():
    p = path()
    set_bar(p, ENTRY + 2, 101)
    decision = staged_decision(p, ENTRY)
    assert decision['three_closes_hold_support'] is True
    assert decision['add_decision'] == 'add'
    set_bar(p, ENTRY + 2, 100)
    decision = staged_decision(p, ENTRY)
    assert decision['above_first_entry'] is False
    assert decision['add_decision'] == 'no_add'
    set_bar(p, ENTRY + 2, 105)
    set_bar(p, ENTRY + 1, 99.99)
    assert staged_decision(p, ENTRY)['three_closes_hold_support'] is False


def test_support_excludes_signal_day_high_and_never_becomes_trailing_high():
    p = confirmation_path()
    set_bar(p, ENTRY - 1, 110)
    assert staged_decision(p, ENTRY)['fixed_support'] == 100
    p.close[ENTRY + 4:] = 1000
    assert staged_decision(p, ENTRY)['fixed_support'] == 100


def test_original_three_black_exit_wins_even_if_price_conditions_confirm():
    p = path()
    set_bar(p, ENTRY - 1, 110)
    set_bar(p, ENTRY, 108, opened=109, high=110, low=90)  # Entry HL2 is 100.
    set_bar(p, ENTRY + 1, 107, opened=108)
    set_bar(p, ENTRY + 2, 106, opened=107)
    decision = staged_decision(p, ENTRY)
    assert decision['add_decision'] == 'no_add'
    assert decision['original_exit_before_add_reason'] == 'three_black'
    assert decision['original_exit_before_add_date'] == str(p.days[ENTRY + 2].date())
    row = staged_row(original(p), p, ENTRY)
    assert row['exit_date'] == str(p.days[ENTRY + 3].date())
    assert row['add_fill_status'] == 'not_requested'


def test_loss12_wins_before_third_day_and_does_not_read_unused_confirmation():
    p = path()
    set_bar(p, ENTRY + 1, 88)
    decision = staged_decision(p, ENTRY)
    assert decision['original_exit_before_add_reason'] == 'loss12'
    assert decision['add_decision_known_at'] == str(p.days[ENTRY + 1].date())
    p.close[ENTRY + 2] = np.nan
    p.close[:ENTRY - 1] = np.nan
    p.close[ENTRY - 1] = 100
    assert staged_decision(p, ENTRY) == decision


def test_failed_confirmation_keeps_half_cash_and_never_retries_later_strength():
    p = path()
    for i in range(ENTRY + 4, len(p.days)):
        set_bar(p, i, 110)
    base = original(p)
    row = staged_row(base, p, ENTRY)
    assert row['status'] == 'closed'
    assert row['add_decision'] == 'no_add'
    assert row['second_entry_date'] is None
    assert row['idle_cash_weight'] == .5
    assert row['net_return'] == pytest.approx(base['net_return'] / 2)
    assert row['second_leg_return_component'] == 0


def test_two_fee_inclusive_halves_and_original_exit_anchor_are_preserved():
    p = confirmation_path()
    base = original(p)
    row = staged_row(base, p, ENTRY)
    end = base['adjusted_end_price']
    assert row['net_return'] == pytest.approx(.5 * net_unit_return(end / 101) + .5 * net_unit_return(end / 106))
    for field in ('entry_date', 'exit_date', 'exit_trigger_date', 'exit_reason',
                  'stop_anchor_adjusted_close', 'holding_days', 'holding_days_inclusive'):
        assert row[field] == base[field]
    assert row['exit_date'] == str(p.days[ENTRY + 63].date())
    assert row['idle_cash_weight'] == 0
    assert row['invested_budget_weight'] == 1
    assert row['mfe'] is None and row['original_mfe'] == base['mfe']


def test_second_fill_does_not_recheck_strength_using_same_day_low_or_close():
    p = confirmation_path()
    set_bar(p, ENTRY + 3, 94, opened=94, high=96, low=92)
    decision = staged_decision(p, ENTRY)
    fill = observe_second_fill(p, ENTRY, decision)
    assert decision['add_decision'] == 'add'
    assert fill['add_fill_status'] == 'filled_hl2_proxy'
    assert fill['second_entry_adjusted_hl2'] == 94


def test_original_exit_on_add_day_close_is_following_day_exit_not_add_cancellation():
    p = confirmation_path()
    set_bar(p, ENTRY + 3, 88)
    base = original(p)
    row = staged_row(base, p, ENTRY)
    assert row['add_fill_status'] == 'filled_hl2_proxy'
    assert row['exit_reason'] == 'loss12'
    assert row['exit_trigger_date'] == row['second_entry_date']
    assert row['exit_date'] == str(p.days[ENTRY + 4].date())


def test_second_leg_uses_adjusted_scale_across_split_without_resetting_anchor():
    p = confirmation_path()
    for i in range(ENTRY + 3, len(p.days)):
        for field in ('raw_close', 'opened', 'high', 'low'):
            getattr(p, field)[i] /= 2
    base = original(p)
    row = staged_row(base, p, ENTRY)
    assert row['second_entry_raw_hl2'] == 53
    assert row['second_entry_adjusted_hl2'] == 106
    assert row['stop_anchor_adjusted_close'] == 101
    assert row['net_return'] == pytest.approx(.5 * net_unit_return(100 / 101) + .5 * net_unit_return(100 / 106))


@pytest.mark.parametrize('kind', ['price', 'eligibility', 'zero_volume', 'dual_source_conflict'])
def test_missing_or_invalid_second_entry_never_becomes_free_cash_no_add(kind):
    p = confirmation_path()
    if kind == 'price':
        p.high[ENTRY + 3] = np.nan
    elif kind == 'eligibility':
        p.eligible[ENTRY + 3] = False
    elif kind == 'zero_volume':
        p.volume[ENTRY + 3] = 0
    else:
        p.other[ENTRY + 3] = 101
    row = staged_row(original(p), p, ENTRY)
    assert row['add_decision'] == 'add'
    assert row['add_fill_status'] == 'unresolved'
    assert row['status'] == 'unresolved' and row['net_return'] is None
    assert row['invested_budget_weight'] is None and row['idle_cash_weight'] is None
    assert row['requested_second_budget_weight'] == .5


def test_pending_confirmation_and_pending_add_remain_visible_unrealized_rows():
    for end, decision, fill in ((ENTRY + 2, 'pending_confirmation', 'not_requested'),
                               (ENTRY + 3, 'add', 'pending_add')):
        p = confirmation_path(end)
        base = original(p)
        row = staged_row(base, p, ENTRY)
        assert row['status'] == 'open' and row['net_return'] is None
        assert row['add_decision'] == decision and row['add_fill_status'] == fill
        assert row['unrealized_net_return'] == pytest.approx(base['unrealized_net_return'] / 2)


def test_known_no_add_is_not_overridden_by_future_path_unknown():
    p = path()
    p.high[ENTRY + 4] = np.nan
    row = staged_row(original(p), p, ENTRY)
    assert row['add_decision'] == 'no_add'
    assert row['status'] == 'unresolved'
    assert row['net_return'] is None


def test_last_day_original_exit_stays_pending_and_disables_addition():
    p = path(ENTRY + 3)
    set_bar(p, ENTRY - 1, 110)
    for index, value in ((ENTRY, 108), (ENTRY + 1, 107), (ENTRY + 2, 106)):
        set_bar(p, index, value, opened=value + 1)
    base = original(p)
    row = staged_row(base, p, ENTRY)
    assert row['status'] == 'pending_exit'
    assert row['exit_reason'] == 'three_black' and row['exit_date'] is None
    assert row['add_decision'] == 'no_add'
    assert row['net_return'] is None
    assert row['unrealized_net_return'] == pytest.approx(base['unrealized_net_return'] / 2)


def test_loss12_priority_over_simultaneous_three_black_is_unchanged():
    p = path()
    set_bar(p, ENTRY - 1, 102)
    for index, value in ((ENTRY, 100), (ENTRY + 1, 94), (ENTRY + 2, 88)):
        set_bar(p, index, value, opened=value + 1)
    decision = staged_decision(p, ENTRY)
    assert decision['original_exit_before_add_reason'] == 'loss12'
    assert staged_row(original(p), p, ENTRY)['exit_reason'] == 'loss12'


def test_missing_support_is_unknown_not_negative_confirmation():
    p = confirmation_path()
    p.close[10] = np.nan
    decision = staged_decision(p, ENTRY)
    assert decision['add_decision'] == 'unresolved'
    assert decision['add_decision_issue'] == 'missing_signal_prior60_support'
    assert staged_row(original(p), p, ENTRY)['status'] == 'unresolved'


def test_all_opportunity_comparison_keeps_half_cash_and_reports_unknowns():
    base = [dict(signal_id=str(i), stock_id='1234', signal_date='2024-01-03', status='closed',
                 net_return=v, holding_days=4, exit_reason='three_black')
            for i, v in enumerate((.6, -.2, .4))]
    staged = deepcopy(base)
    for r in staged:
        r.update(add_decision='no_add', add_fill_status='not_requested')
        r['net_return'] /= 2
    report = comparison(base, staged)
    assert report['equal_unit_mean_staged_including_idle_cash'] == pytest.approx(.4 / 3)
    assert report['winner_retention']['original_return30']['remains_return30'] == 1
    assert report['winner_retention']['original_return30']['full_budget_added'] == 0
    staged[2].update(status='unresolved', net_return=None)
    report = comparison(base, staged)
    assert report['common_closed_opportunities'] == 2
    assert report['original_closed_to_unknown_or_unfinished'] == 1
    assert report['full_original_denominator_mean_staged'] is None
    assert report['equal_unit_mean_staged_including_idle_cash'] == pytest.approx(.1)
    assert report['winner_retention']['original_return30']['staged_unknown_or_unfinished'] == 1
    scopes = scope_comparisons(base, staged)
    assert len(scopes) == 12 and scopes['2023_2024']['common_closed_opportunities'] == 2


def test_not_entered_signal_is_retained_and_wrong_timing_rejected():
    row = dict(signal_id='last', status='not_entered', entry_date=None)
    assert staged_row(row, None, None)['idle_cash_weight'] == 1
    p = path()
    base = original(p)
    base['entry_date'] = str(p.days[ENTRY + 1].date())
    with pytest.raises(ValueError, match='T then T'):
        staged_row(base, p, ENTRY)


def test_cached_confirmation_support_is_bound_to_the_original_signal_feature():
    decision = staged_decision(confirmation_path(), ENTRY)
    verify_fixed_support(decision, {'previous60_high_adjusted': 100.})
    with pytest.raises(ValueError, match='sealed original signal feature'):
        verify_fixed_support(decision, {'previous60_high_adjusted': 101.})
    # An original exit before confirmation does not need this unused feature.
    verify_fixed_support({'fixed_support': None}, {'previous60_high_adjusted': np.nan})
