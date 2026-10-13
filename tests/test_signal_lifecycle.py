import pandas as pd
import pytest

from app.signal_lifecycle import SignalHistory, add_price_context


def test_retrigger_uses_previous_transition_not_continuation_or_future():
    days = ['2026-10-02', '2026-10-05', '2026-10-06', '2026-10-07', '2026-10-08', '2026-10-12', '2026-10-13']
    history = SignalHistory(days, {('4927', 'poc', days[0]): True,
        ('4927', 'poc', days[1]): False, ('4927', 'poc', days[-1]): True,
        ('4927', 'different_rule', days[-2]): True})
    result = history.describe('4927', 'poc', '2026-10-12', True)
    assert result['state'] == 'retrigger'
    assert result['previous_trigger_date'] == '2026-10-02'
    assert result['previous_match_date'] == '2026-10-05'
    assert result['sessions_since_previous_trigger'] == 5
    assert result['prior_known_triggers'] == 1
    priced = add_price_context(result, {'2026-10-02': 48.15, '2026-10-12': 57.9, '2026-10-13': 1000.})
    assert priced['price_change_since_previous_trigger'] == pytest.approx(.20249221183800614)


def test_unknown_previous_day_is_not_promoted_to_trigger_or_continuation():
    history = SignalHistory(['2026-10-02', '2026-10-05'], {('4927', 'poc', '2026-10-02'): True})
    assert history.describe('4927', 'poc', '2026-10-05', None)['state'] == 'unknown'
    assert history.describe('4927', 'poc', '2026-10-05', False)['state'] == 'continuing'
    result = history.describe('1101', 'poc', '2026-10-05', True)
    assert result['state'] == 'first_observed'
    assert result['coverage'] == 'known_records_only'
    assert result['previous_trigger_date'] is None


def test_history_window_is_120_sessions_including_today_not_all_time():
    days = [str(d.date()) for d in pd.bdate_range('2025-01-01', periods=121)]
    history = SignalHistory(days, {('4927', 'poc', days[0]): True})
    assert history.describe('4927', 'poc', days[-2], True)['state'] == 'retrigger'
    result = history.describe('4927', 'poc', days[-1], True)
    assert result['state'] == 'first_observed'
    assert result['window_start'] == days[1]
    assert result['observed_sessions'] == 120


@pytest.mark.parametrize('missing_price', [None, 0, float('nan')])
def test_price_context_never_substitutes_a_nearby_close(missing_price):
    history = SignalHistory(['2026-10-02', '2026-10-05'], {('4927', 'poc', '2026-10-02'): True})
    result = history.describe('4927', 'poc', '2026-10-05', True)
    assert add_price_context(result, {'2026-10-02': 48.15, '2026-10-05': missing_price})['price_change_since_previous_trigger'] is None


def test_known_continuation_proves_prior_match_without_inventing_trigger_date():
    history = SignalHistory(['2026-10-02', '2026-10-05'], {('4927', 'poc', '2026-10-02'): False})
    result = history.describe('4927', 'poc', '2026-10-05', True)
    assert result['state'] == 'retrigger'
    assert result['previous_trigger_date'] is None
