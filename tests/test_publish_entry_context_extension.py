from copy import deepcopy

import pytest

from scripts.publish_entry_context_extension import (
    local_path, publish, read_sealed, verify_continuity,
)
from scripts.publish_entry_context_terminal import write_sealed


@pytest.fixture
def history():
    day = dict(date='2026-10-06', market_breadth_value=.55, market_narrow=False,
               parent_candidates=1, eligible_stocks=1000, valid60_stocks=900)
    event = dict(stock_id='2330', signal_date='2026-10-06', peer_ids=['2454'],
                 contraction_ratio=.25, contraction=True, market_narrow=False,
                 peer_breadth=None, peer_and_narrow=None)
    previous = dict(schema='entry_context_terminal_v1', start='2026-09-01',
                    end='2026-10-06', dates=['2026-10-06'], day_context=[day],
                    live_qualified=False)
    report = dict(schema='entry60_current_signal_check_v1', start='2026-09-01',
                  end='2026-10-06', parent='legacy_course_breakout',
                  all_first_signals=[event], account_backtest=False, live_qualified=False)
    next_day = dict(day, date='2026-10-07', market_breadth_value=.45, market_narrow=True)
    next_event = dict(event, signal_date='2026-10-07', market_narrow=True)
    return previous, report, [deepcopy(event), next_event], [deepcopy(day), next_day]


def check(history):
    previous, report, rows, days = history
    return verify_continuity(previous, report, start='2026-09-01', end='2026-10-07',
                             rows=rows, day_context=days)


def test_extension_preserves_all_prior_rows_and_keeps_new_session(history):
    assert check(history) == dict(previous_end='2026-10-06', unchanged_dates=1,
        unchanged_parent_events=1, exact_comparison=True)


@pytest.mark.parametrize(('field', 'value'), [
    ('contraction_ratio', .250000000000001),
    ('contraction', False),
    ('contraction', 1),
    ('peer_breadth', False),
    ('peer_and_narrow', False),
    ('peer_ids', ['2317']),
    ('stock_id', '2317'),
])
def test_extension_rejects_any_changed_historical_value(history, field, value):
    history[2][0][field] = value
    with pytest.raises(ValueError, match='parent events or conditions changed'):
        check(history)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'backdated'])
def test_extension_rejects_changed_prior_event_membership(history, mutation):
    rows = history[2]
    if mutation == 'missing':
        rows.pop(0)
    elif mutation == 'duplicate':
        rows.insert(0, deepcopy(rows[0]))
    else:
        rows[1]['signal_date'] = '2026-10-06'
    with pytest.raises(ValueError, match='parent events or conditions changed'):
        check(history)


@pytest.mark.parametrize('mutation', ['missing_day', 'breadth', 'count'])
def test_extension_rejects_daily_history_revisions(history, mutation):
    days = history[3]
    if mutation == 'missing_day':
        days.pop(0)
    elif mutation == 'breadth':
        days[0]['market_breadth_value'] = .45
    else:
        days[0]['parent_candidates'] = 2
    with pytest.raises(ValueError, match='daily context changed'):
        check(history)


def test_extension_rejects_other_prior_scope_and_nonextension(history):
    previous, report, rows, days = history
    report['account_backtest'] = True
    with pytest.raises(ValueError, match='incompatible scope'):
        check(history)
    report['account_backtest'] = False
    with pytest.raises(ValueError, match='add a later session'):
        verify_continuity(previous, report, start='2026-09-01', end='2026-10-06',
                          rows=rows, day_context=days)


def test_extension_cannot_claim_a_cutoff_without_observed_session(history):
    history[3].pop()
    with pytest.raises(ValueError, match='observed market session'):
        check(history)


def test_previous_snapshot_content_and_sidecar_are_verified(tmp_path):
    path = tmp_path / 'prior.json'
    sha = write_sealed(path, {'sealed': True})
    descriptor = dict(path='prior.json', sha256=sha)
    assert read_sealed(tmp_path, descriptor) == {'sealed': True}
    path.with_suffix('.sha256').write_text('0' * 64)
    with pytest.raises(ValueError, match='checksum failed'):
        read_sealed(tmp_path, descriptor)
    path.with_suffix('.sha256').write_text(sha)
    path.write_text('{"sealed":false}')
    with pytest.raises(ValueError, match='checksum failed'):
        read_sealed(tmp_path, descriptor)


def test_publication_paths_cannot_escape_root(tmp_path):
    for bad in ('../escaped.json', str(tmp_path / 'absolute.json')):
        with pytest.raises(ValueError, match='relative to the project'):
            local_path(tmp_path, bad)
    assert local_path(tmp_path, 'new/report.json') == tmp_path / 'new/report.json'


def test_publisher_refuses_to_reuse_either_previous_output(tmp_path, history):
    previous, report, _, _ = history
    report_sha = write_sealed(tmp_path / 'old/report.json', report)
    previous['artifacts'] = dict(report=dict(path='old/report.json', sha256=report_sha))
    previous_sha = write_sealed(tmp_path / 'old.json', previous)
    for publication, output in [('old.json', 'new'), ('new.json', 'old')]:
        with pytest.raises(ValueError, match='previous evidence is frozen'):
            publish(tmp_path, bundle='does-not-need-to-exist', manifest_sha256='0' * 64,
                    output=output, publication=publication, end='2026-10-07',
                    previous_publication='old.json', previous_sha256=previous_sha)
    assert not (tmp_path / 'new').exists()
    assert not (tmp_path / 'new.json').exists()
