import numpy as np
import pandas as pd
import pytest

from scripts.publish_entry_context_terminal import market_days, write_sealed


def test_publication_is_idempotent_and_does_not_replace_other_evidence(tmp_path):
    path = tmp_path / 'report.json'
    sha = write_sealed(path, {'known': False, 'unknown': None})
    stamp = path.stat().st_mtime_ns
    sidecar_stamp = path.with_suffix('.sha256').stat().st_mtime_ns
    assert write_sealed(path, {'known': False, 'unknown': None}) == sha
    assert path.stat().st_mtime_ns == stamp
    assert path.with_suffix('.sha256').stat().st_mtime_ns == sidecar_stamp
    with pytest.raises(ValueError, match='different evidence'):
        write_sealed(path, {'known': True})
    assert path.with_suffix('.sha256').read_text().strip() == sha


def test_daily_context_covers_no_event_days_and_preserves_insufficient_breadth():
    dates = pd.bdate_range('2026-01-01', periods=63)
    ids = ['0050'] + [str(1000 + i) for i in range(625)]
    close = pd.DataFrame(10., index=dates, columns=ids)
    eligible = pd.DataFrame(True, index=dates, columns=ids)
    close.iloc[60, 1:251] = 11.
    # Exactly 500 valid stocks / 625 eligible = 80% coverage, boundary accepted.
    close.iloc[60, 501:] = np.nan
    # Next session, one more missing quote makes both minimum tests fail.
    close.iloc[61, 500] = np.nan
    events = pd.DataFrame({'signal_date': [str(dates[60].date())]})
    rows = market_days({'c': close, 'eligible': eligible}, dates, ids, events,
                      str(dates[60].date()), str(dates[61].date()))
    assert len(rows) == 2
    assert rows[0]['market_breadth_value'] == .5
    assert rows[0]['market_narrow'] is False  # threshold is strictly below 50%
    assert rows[0]['parent_candidates'] == 1
    assert rows[0]['valid60_stocks'] == 500
    assert rows[0]['eligible_stocks'] == 625  # ETF excluded
    assert rows[1]['market_breadth_value'] is None
    assert rows[1]['market_narrow'] is None
    assert rows[1]['parent_candidates'] == 0
