"""Synthetic causal/source regressions; never load historical research outcomes."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts.research_volume_profile import audit_day, constant_scale, early_exit, paired
from skills.independent_three_black import ThreeBlackPath, observe


DAY = '2026-09-22'
ENTRY = 25


def ticks():
    return pd.DataFrame(dict(date=[DAY] * 4, stock_id=['2330'] * 4,
        deal_price=[100, 101, 99, 98], volume=[1, 2, 3, 4],
        Time=['09:00:00', '10:00:00', '13:33:59', '14:30:00'], TickType=[1] * 4))


def official(**changes):
    return dict(dict(market='TWSE', volume=6000, volume_scope='ordinary_session',
        source_id='synthetic', open=100., high=101., low=99., close=99.), **changes)


def exact(**changes):
    return dict(dict(stock_id='2330', date=DAY, market='TWSE', shares=6000,
        amount_cents=59900000, open_cents=10000, high_cents=10100,
        low_cents=9900, close_cents=9900), **changes)


def path():
    c = np.full(100, 100.)
    return ThreeBlackPath(pd.bdate_range('2024-01-01', periods=len(c)), c.copy(),
        c.copy(), np.ones(len(c), bool), c.copy(), c + 1, c - 1,
        np.full(len(c), 1000000.), c.copy())


def set_bar(p, i, close, high=None, low=None):
    p.close[i] = p.other[i] = p.raw_close[i] = p.opened[i] = close
    p.high[i] = close + 1 if high is None else high
    p.low[i] = close - 1 if low is None else low


def original(p):
    return dict(observe(p, ENTRY), signal_id='synthetic', stock_id='2330',
        signal_date=str(p.days[ENTRY - 1].date()), entry_date=str(p.days[ENTRY].date()))


def feature(vah=99.5):
    return dict(status='known', above_vah=True, vah_adjusted=vah)


def test_ordinary_volume_match_does_not_claim_amount_or_tick_sequence():
    checked = audit_day(ticks(), '2330', DAY, official())
    assert checked['status'] == 'usable_ordinary_daily_matched'
    assert checked['ordinary_volume_matched'] is True
    assert checked['ordinary_amount_matched'] is False
    assert checked['tick_sequence_complete'] is False
    assert checked['tape']['shares'] == 6000
    assert checked['tape']['fixed_price_shares'] == 4000
    assert audit_day(ticks(), '2330', DAY, official(volume=5999))['status'] == 'official_ordinary_volume_conflict'


def test_exact_ordinary_reference_checks_identity_amount_and_volume():
    checked = audit_day(ticks(), '2330', DAY, official(), exact())
    assert checked['ordinary_volume_matched'] and checked['ordinary_amount_matched']
    assert checked['tick_sequence_complete'] is False
    for change in (dict(amount_cents=59899999), dict(shares=5999)):
        bad = audit_day(ticks(), '2330', DAY, official(), exact(**change))
        assert bad['status'] == 'official_ordinary_aggregate_conflict'
        assert not bad['ordinary_amount_matched']
    with pytest.raises(ValueError, match='identity mismatch'):
        audit_day(ticks(), '2330', DAY, official(), exact(date='2026-09-23'))


def test_all_session_total_is_only_upper_bound_and_repeated_prints_are_preserved():
    all_session = official(volume=10000, volume_scope='all_daily_sessions')
    checked = audit_day(ticks(), '2330', DAY, all_session)
    assert checked['status'] == 'usable_provider_diagnostic'
    assert not checked['ordinary_volume_matched']
    assert not checked['ordinary_amount_matched']
    assert audit_day(ticks(), '2330', DAY, all_session | {'volume': 9999})['status'] == 'exceeds_official_all_session_volume'
    duplicate = pd.concat([ticks().iloc[:1], ticks()], ignore_index=True)
    checked = audit_day(duplicate, '2330', DAY, official(volume=7000))
    assert checked['tape']['shares'] == 7000
    assert checked['status'] == 'usable_ordinary_daily_matched'


def test_constant_scale_accepts_rounding_but_rejects_corporate_action_or_missing():
    assert constant_scale([80., 80.8, 81.6], [100., 101., 102.])
    assert constant_scale([80.001, 80.802, 81.597], [100., 101., 102.])
    for adjusted, raw in (([100., 50.], [100., 100.]), ([100., np.nan], [100., 100.]),
                          ([100., 0.], [100., 100.]), ([100.], [100., 100.]), ([], [])):
        assert not constant_scale(adjusted, raw)


def test_support_close_trigger_uses_next_session_hl2_and_both_sided_costs():
    p = path()
    set_bar(p, ENTRY + 3, 99.)
    set_bar(p, ENTRY + 4, 90., high=92., low=88.)
    base = original(p)
    assert base['status'] == 'closed' and base['exit_reason'] == 'time63'
    changed = early_exit(base, p, feature())
    assert changed['exit_trigger_date'] == str(p.days[ENTRY + 3].date())
    assert changed['exit_date'] == str(p.days[ENTRY + 4].date())
    assert changed['exit_price'] == 90.
    expected = .9 * (1 - .001425 - .0045 - .003) / (1 + .001425 + .0045) - 1
    assert changed['net_return'] == pytest.approx(expected)
    assert changed['holding_days'] == 4
    assert changed['mfe'] is None and changed['confirmed_high_return'] is None
    for name in ('close', 'other', 'raw_close', 'high', 'low', 'opened', 'volume'):
        getattr(p, name)[ENTRY + 5:] = np.nan
    assert early_exit(base, p, feature()) == changed


def test_original_stop_wins_same_close_support_break():
    p = path()
    set_bar(p, ENTRY + 2, 88.)
    base = original(p)
    assert base['exit_reason'] == 'loss12'
    assert early_exit(base, p, feature(95.)) == base


def test_unfilled_original_stop_is_not_replaced_by_a_later_successful_exit():
    p = path()
    set_bar(p, ENTRY + 1, 88.)
    set_bar(p, ENTRY + 2, 88.)
    p.volume[ENTRY + 2] = 0
    set_bar(p, ENTRY + 3, 79.)
    set_bar(p, ENTRY + 4, 80.)
    base = original(p)
    assert base['status'] == 'unresolved' and base['exit_trigger_date'] is None
    assert early_exit(base, p, feature(80.)) == base


def test_recovered_later_source_conflict_cannot_hide_unknown_decision_prefix():
    p = path()
    # Each adjacent discrepancy stays below 0.5%, but the sixth prefix exceeds
    # 2%. Recovery before the original exit must not certify that earlier close.
    p.other[ENTRY + 1:ENTRY + 13] = [99.6, 99.2, 98.8, 98.4, 98., 97.6,
                                             98., 98.4, 98.8, 99.2, 99.6, 100.]
    base = original(p)
    assert base['status'] == 'closed'
    changed = early_exit(base, p, feature(90.))
    assert changed['status'] == 'unresolved'
    assert changed['data_issue'] == 'volume_profile_decision_path_invalid'
    for key in ('net_return', 'gross_return', 'exit_date', 'exit_trigger_date', 'mfe'):
        assert changed[key] is None


def test_unknown_profile_or_unfillable_next_session_is_not_zero_return():
    p = path()
    base = original(p)
    unavailable = early_exit(base, p, dict(status='unknown'))
    assert unavailable['status'] == 'unresolved'
    assert unavailable['net_return'] is None and unavailable['gross_return'] is None
    assert unavailable['exit_date'] is None and unavailable['exit_trigger_date'] is None
    set_bar(p, ENTRY + 3, 99.)
    p.volume[ENTRY + 4] = 0
    changed = early_exit(base, p, feature())
    assert changed['status'] == 'unresolved'
    assert changed['net_return'] is None
    assert changed['data_issue'] == 'volume_profile_exit_path_invalid'


def test_unknown_path_is_excluded_explicitly_from_paired_return_not_given_zero():
    p = path()
    base = original(p)
    changed = early_exit(base, p, dict(status='unknown'))
    stats = paired([base], [changed])
    assert stats['original_count'] == 1 and stats['common_closed'] == 0
    assert stats['unpaired_statuses'] == {'unresolved': 1}
    assert stats['mean_difference'] is None
