"""Fixed, causal post-entry diagnostics; execution and outcome labels are separate.

The signal is confirmed at T close and entry is T+1 open. Holding day three
therefore ends at T+3 close, with any resulting exit first possible at T+4.
Unavailable observations remain unknown, including for rules whose other
components would already be false. No market row after the checkpoint is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


CHECKPOINT_AGES = (3, 5)
CHECKPOINT_ACTIONS = ('support_failed', 'dry_weak', 'stalled_weak')
IDENTITIES = ('cohort', 'event_id', 'stock_id', 'signal_date', 'signal_index')
NUMERIC_COLUMNS = (
    'entry_price_adj', 'checkpoint_return', 'benchmark_checkpoint_return',
    'relative_checkpoint_return', 'signal_close_adj', 'signal_low_adj',
    'checkpoint_close_adj', 'volume_two_day_ratio',
)


def build_checkpoints(events, f, days, ids):
    """Expand every first-signal event into holding-day 3 and 5 observations.

    ``f`` must be the aligned matrices from ``strategy_scanner.engine._prepare``.
    All own bars from T through the checkpoint, and benchmark bars from entry
    through the checkpoint, must be valid, eligible and positively traded.
    ``next_exit_date`` is a calendar date only; it never implies a valid fill.
    A known final-date checkpoint may have no next exit date yet.
    """
    if not set(IDENTITIES).issubset(events.columns):
        raise ValueError('Checkpoint events are missing required identities')
    if events.duplicated(['cohort', 'event_id']).any():
        raise ValueError('Duplicate checkpoint event identity')
    days = pd.DatetimeIndex(days)
    if (days.hasnans or days.tz is not None or days.has_duplicates
            or not days.is_monotonic_increasing or not days.equals(days.normalize())):
        raise ValueError('Checkpoint calendar must contain sorted unique market dates')
    ids = list(ids)
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate checkpoint stock identity')
    required = ('c', 'l', 'open', 'close', 'volume', 'valid', 'eligible')
    if not set(required).issubset(f):
        raise ValueError('Missing prepared checkpoint matrices')
    for key in required:
        frame = f[key]
        if not frame.index.equals(days) or frame.columns.tolist() != ids:
            raise ValueError('Checkpoint matrix axes differ from calendar or stock IDs')
    arrays = {key: f[key].to_numpy(dtype=float) for key in required[:-2]}
    valid = (f['valid'].eq(True).fillna(False)
             & f['eligible'].eq(True).fillna(False)).to_numpy(dtype=bool)
    # Do not rely exclusively on the prepared validity flag when validating
    # adjusted numeric observations or when consumers supply modified fixtures.
    for key in ('c', 'open', 'close', 'volume'):
        valid &= np.isfinite(arrays[key]) & (arrays[key] > 0)
    own_valid = valid & np.isfinite(arrays['l']) & (arrays['l'] > 0)
    columns = {sid: i for i, sid in enumerate(ids)}
    benchmark = columns.get('0050')
    dates = [str(day.date()) for day in days]
    records = []
    for event in events.loc[:, IDENTITIES].to_dict('records'):
        raw_index = event['signal_index']
        if (isinstance(raw_index, (bool, np.bool_)) or not isinstance(raw_index,
                (int, np.integer, float, np.floating)) or not np.isfinite(raw_index)
                or raw_index < 0 or int(raw_index) != raw_index):
            raise ValueError('Checkpoint signal index must be a nonnegative integer')
        index = int(raw_index)
        if index < len(days) and event['signal_date'] != dates[index]:
            raise ValueError('Checkpoint signal date and index disagree')
        column = columns.get(event['stock_id'])
        for age in CHECKPOINT_AGES:
            checkpoint = index + age
            row = dict(event, checkpoint_age=age, checkpoint_index=checkpoint,
                checkpoint_date=dates[checkpoint] if checkpoint < len(days) else None,
                next_exit_date=dates[checkpoint+1] if checkpoint+1 < len(days) else None,
                checkpoint_known=False, checkpoint_issue=None)
            row.update({name: np.nan for name in NUMERIC_COLUMNS})
            row.update({name: pd.NA for name in CHECKPOINT_ACTIONS})
            issue = None
            if checkpoint >= len(days):
                issue = 'immature_checkpoint'
            elif column is None:
                issue = 'stock_not_covered'
            elif not own_valid[index:checkpoint+1, column].all():
                issue = 'own_window_incomplete'
            elif benchmark is None:
                issue = 'benchmark_not_covered'
            elif not valid[index+1:checkpoint+1, benchmark].all():
                issue = 'benchmark_window_incomplete'
            if issue:
                row['checkpoint_issue'] = issue
                records.append(row)
                continue
            entry = index + 1
            # Use each entry day's adjustment factor, not the signal-day factor.
            entry_price = arrays['open'][entry, column] * (
                arrays['c'][entry, column] / arrays['close'][entry, column])
            benchmark_entry = arrays['open'][entry, benchmark] * (
                arrays['c'][entry, benchmark] / arrays['close'][entry, benchmark])
            close = arrays['c'][checkpoint, column]
            signal_close = arrays['c'][index, column]
            signal_low = arrays['l'][index, column]
            own_return = close / entry_price - 1
            benchmark_return = arrays['c'][checkpoint, benchmark] / benchmark_entry - 1
            relative = own_return - benchmark_return
            volume_ratio = arrays['volume'][checkpoint-1:checkpoint+1, column].max() / arrays['volume'][index, column]
            row.update(checkpoint_known=True, entry_price_adj=float(entry_price),
                checkpoint_return=float(own_return),
                benchmark_checkpoint_return=float(benchmark_return),
                relative_checkpoint_return=float(relative),
                signal_close_adj=float(signal_close), signal_low_adj=float(signal_low),
                checkpoint_close_adj=float(close), volume_two_day_ratio=float(volume_ratio),
                support_failed=bool(close < signal_low and relative < 0),
                dry_weak=bool(volume_ratio < .5 and close < signal_close and relative < 0),
                stalled_weak=bool(close <= entry_price and close < signal_close and relative < 0))
            records.append(row)
    output_columns = list(IDENTITIES) + [
        'checkpoint_age', 'checkpoint_index', 'checkpoint_date', 'next_exit_date',
        'checkpoint_known', 'checkpoint_issue', *NUMERIC_COLUMNS, *CHECKPOINT_ACTIONS,
    ]
    result = pd.DataFrame(records, columns=output_columns)
    result['checkpoint_known'] = result['checkpoint_known'].astype(bool)
    for name in CHECKPOINT_ACTIONS:
        result[name] = pd.array(result[name], dtype='boolean')
    return result
