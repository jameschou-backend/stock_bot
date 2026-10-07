"""Timing, unknown-data and adjustment boundaries for fixed exit diagnostics."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from skills.rally_checkpoint import build_checkpoints, CHECKPOINT_ACTIONS
from skills.strategy_scanner.engine import _prepare


def inputs(length=9):
    days = pd.bdate_range('2024-01-02', periods=length)
    bars = []
    for sid in ('0050', '2330'):
        for day in days:
            bars.append(dict(date=day, stock_id=sid, open=100., high=105., low=95.,
                close=100., adjusted_close=100., volume=1000., amount=100_000.,
                quality=True, eligible=True))
    f, days, ids = _prepare(pd.DataFrame(bars), days, days[-1])
    events = pd.DataFrame([dict(cohort='original_red', event_id='first',
        stock_id='2330', signal_date=str(days[0].date()), signal_index=0)])
    return events, f, days, ids


def test_holding_day_count_starts_at_entry_and_next_day_is_calendar_only():
    events, f, days, ids = inputs(6)
    result = build_checkpoints(events, f, days, ids)
    assert result.checkpoint_age.tolist() == [3, 5]
    assert result.checkpoint_index.tolist() == [3, 5]
    assert result.checkpoint_date.tolist() == [str(days[3].date()), str(days[5].date())]
    assert result.next_exit_date.tolist() == [str(days[4].date()), None]
    assert result.checkpoint_known.all()


def test_every_event_kept_when_checkpoint_immature():
    events, f, days, ids = inputs(4)
    result = build_checkpoints(events, f, days, ids)
    assert len(result) == 2
    assert result.checkpoint_known.tolist() == [True, False]
    assert result.iloc[1].checkpoint_issue == 'immature_checkpoint'
    assert pd.isna(result.iloc[1].checkpoint_date)
    assert result.loc[1, list(CHECKPOINT_ACTIONS)].isna().all()


def test_entry_uses_its_own_adjustment_factor_and_not_signal_factor():
    events, f, days, ids = inputs()
    # A corporate-action adjustment changes between signal and entry.
    f['open'].loc[days[1], '2330'] = 50.
    f['close'].loc[days[1], '2330'] = 50.
    f['c'].loc[days[1], '2330'] = 100.
    f['c'].loc[days[3], '2330'] = 110.
    result = build_checkpoints(events, f, days, ids)
    assert result.iloc[0].entry_price_adj == 100.
    assert result.iloc[0].checkpoint_return == pytest.approx(.1)


def test_support_failure_requires_relative_weakness():
    events, f, days, ids = inputs()
    f['c'].loc[days[3], '2330'] = 90.
    result = build_checkpoints(events, f, days, ids).iloc[0]
    assert result.support_failed
    assert result.relative_checkpoint_return == pytest.approx(-.1)
    f['c'].loc[days[3], '0050'] = 80.
    result = build_checkpoints(events, f, days, ids).iloc[0]
    assert not result.support_failed
    assert result.relative_checkpoint_return == pytest.approx(.1)


def test_dry_strong_does_not_exit_and_both_recent_days_must_be_dry():
    events, f, days, ids = inputs()
    f['volume'].loc[days[2:4], '2330'] = 200.
    f['c'].loc[days[3], '2330'] = 110.
    result = build_checkpoints(events, f, days, ids).iloc[0]
    assert result.volume_two_day_ratio == .2
    assert not result.dry_weak
    f['c'].loc[days[3], '2330'] = 99.
    assert build_checkpoints(events, f, days, ids).iloc[0].dry_weak
    f['volume'].loc[days[2], '2330'] = 500.
    result = build_checkpoints(events, f, days, ids).iloc[0]
    assert result.volume_two_day_ratio == .5
    assert not result.dry_weak


def test_stalled_requires_no_gain_below_signal_and_relative_weakness():
    events, f, days, ids = inputs()
    f['open'].loc[days[1], '2330'] = 99.
    f['c'].loc[days[3], '2330'] = 99.
    f['c'].loc[days[3], '0050'] = 101.
    result = build_checkpoints(events, f, days, ids).iloc[0]
    assert result.stalled_weak
    assert not result.support_failed
    f['open'].loc[days[1], '2330'] = 98.
    assert not build_checkpoints(events, f, days, ids).iloc[0].stalled_weak


@pytest.mark.parametrize('field,index,value', [
    ('valid', 0, False), ('valid', 2, False), ('eligible', 1, None),
    ('volume', 0, 0.), ('volume', 2, 0.), ('c', 1, np.nan),
    ('l', 2, np.nan), ('open', 1, 0.),
])
def test_missing_own_window_is_unknown_instead_of_false(field, index, value):
    events, f, days, ids = inputs()
    if field in ('valid', 'eligible'):
        f[field] = f[field].astype('boolean')
    f[field].loc[days[index], '2330'] = value
    result = build_checkpoints(events, f, days, ids)
    assert not result.checkpoint_known.any()
    assert result.checkpoint_issue.eq('own_window_incomplete').all()
    assert result[list(CHECKPOINT_ACTIONS)].isna().all().all()


@pytest.mark.parametrize('field,value', [('valid', False), ('eligible', None),
                                        ('volume', 0.), ('c', np.nan)])
def test_missing_or_zero_benchmark_window_is_unknown(field, value):
    events, f, days, ids = inputs()
    if field in ('valid', 'eligible'):
        f[field] = f[field].astype('boolean')
    f[field].loc[days[2], '0050'] = value
    result = build_checkpoints(events, f, days, ids)
    assert result.checkpoint_issue.eq('benchmark_window_incomplete').all()
    assert result[list(CHECKPOINT_ACTIONS)].isna().all().all()


def test_benchmark_signal_day_does_not_enter_entry_to_checkpoint_window():
    events, f, days, ids = inputs()
    f['valid'].loc[days[0], '0050'] = False
    assert build_checkpoints(events, f, days, ids).checkpoint_known.all()


def test_missing_benchmark_or_stock_retains_unknown_events():
    events, f, days, ids = inputs()
    for missing, expected in [('0050', 'benchmark_not_covered'), ('2330', 'stock_not_covered')]:
        remaining = [sid for sid in ids if sid != missing]
        subset = {key: matrix.loc[:, remaining] for key, matrix in f.items()}
        result = build_checkpoints(events, subset, days, remaining)
        assert len(result) == 2
        assert result.checkpoint_issue.eq(expected).all()


def test_future_prices_cannot_change_checkpoint_and_true_prefix_matches():
    events, f, days, ids = inputs()
    full = build_checkpoints(events, f, days, ids)
    changed = deepcopy(f)
    for field in ('c', 'l', 'open', 'close', 'volume'):
        changed[field].loc[days[6:], :] *= 19.
    changed['valid'].loc[days[6:], :] = False
    pd.testing.assert_frame_equal(full, build_checkpoints(events, changed, days, ids))
    prefix = {key: matrix.loc[days[:6]] for key, matrix in f.items()}
    truncated = build_checkpoints(events, prefix, days[:6], ids)
    # Only availability of a future calendar date may differ, never a decision.
    pd.testing.assert_frame_equal(full.drop(columns='next_exit_date'),
                                  truncated.drop(columns='next_exit_date'))
    changed = deepcopy(f)
    changed['c'].loc[days[4:], '2330'] = .01
    changed['valid'].loc[days[4:], '2330'] = False
    pd.testing.assert_series_equal(full.iloc[0], build_checkpoints(events, changed, days, ids).iloc[0])


def test_empty_input_preserves_output_schema_and_nullable_actions():
    events, f, days, ids = inputs()
    result = build_checkpoints(events.iloc[:0], f, days, ids)
    assert result.empty and 'checkpoint_return' in result
    assert all(str(result[name].dtype) == 'boolean' for name in CHECKPOINT_ACTIONS)


def test_inconsistent_index_or_misaligned_matrix_is_rejected():
    events, f, days, ids = inputs()
    altered = events.copy()
    altered.loc[0, 'signal_index'] = 1
    with pytest.raises(ValueError, match='disagree'):
        build_checkpoints(altered, f, days, ids)
    f['c'] = f['c'][ids[::-1]]
    with pytest.raises(ValueError, match='axes'):
        build_checkpoints(events, f, days, ids)
