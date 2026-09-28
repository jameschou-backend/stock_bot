import pytest
from skills.peak_stop_research import PeakStop, bar_status


def test_user_1000_to_850_without_profit_arm():
    stop = PeakStop(1000, 0)
    assert stop.observe(1, 851) is None
    result = stop.observe(2, 850)
    assert result['threshold'] == 850 and result['fill_price'] is None


def test_new_high_ratchets_stop_and_rebound_does_not_cancel():
    stop = PeakStop(1000, 0)
    assert stop.observe(1, 1200) is None
    assert stop.observe(2, 1100) is None
    first = stop.observe(3, 1020)
    assert first['threshold'] == 1020
    assert stop.observe(4, 1500) == first


def test_gap_records_actual_observation_not_assumed_stop_fill():
    result = PeakStop(1000, 0).observe(1, 800)
    assert result['observed_price'] == 800 and result['threshold'] == 850
    assert result['fill_price'] is None


def test_same_ohlc_can_have_different_trigger_outcomes():
    # Open-low-high-close versus open-high-low-close, identical daily OHLC.
    low_first, high_first = PeakStop(900, 0), PeakStop(900, 0)
    assert all(low_first.observe(i, p) is None for i, p in enumerate([900, 800, 1000, 950], 1))
    assert [high_first.observe(i, p) for i, p in enumerate([900, 1000, 800, 950], 1)][2] is not None
    assert bar_status(900, 900, 1000, 800, 950)['status'] == 'intraday_order_ambiguous'


def test_previous_peak_or_final_close_can_prove_trigger():
    assert bar_status(1000, 950, 1000, 840, 950)['status'] == 'certain_trigger'
    assert bar_status(900, 900, 1000, 800, 840)['status'] == 'certain_trigger'
    assert bar_status(900, 900, 1000, 860, 950)['status'] == 'no_trigger'


def test_no_pre_entry_or_tied_tick_order_allowed():
    stop = PeakStop(1000, 10)
    for sequence in (9, 10):
        with pytest.raises(ValueError, match='ordered'):
            stop.observe(sequence, 800)
    with pytest.raises(ValueError):
        stop.observe(11, float('nan'))


def test_future_prices_do_not_change_trigger_prefix():
    first, second = PeakStop(1000, 0), PeakStop(1000, 0)
    a = [first.observe(i, p) for i, p in enumerate([1100, 1050, 934], 1)]
    b = [second.observe(i, p) for i, p in enumerate([1100, 1050, 934, 2000], 1)]
    assert a == b[:3]
