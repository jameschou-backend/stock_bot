import numpy as np
import pandas as pd
import pytest

from skills.surge_capture import labels, nonoverlap_starts, window_gate_counts, label_statistics, causal_gates


def frame(values):
    return pd.DataFrame({'1234': np.asarray(values, dtype=float), '0050': np.ones(len(values))*10},
                        index=pd.bdate_range('2020-01-01', periods=len(values)))


def test_future_label_is_not_a_signal_and_right_censored_is_unknown():
    c = frame([10, 11, 12, 13, 15, 16])
    out = labels(c, c, c.notna(), 4, .5)
    assert out['status']['1234'].tolist() == [5, 4, 1, 1, 1, 1]
    stats = label_statistics(out['status']['1234'], out['price_return']['1234'])
    assert stats['known'] == 2 and stats['surge_rate'] == .5 and stats['immature'] == 4


def test_halt_missing_price_and_identity_are_not_failures():
    c = frame([10, 11, np.nan, 13, 15, 16])
    e = pd.DataFrame(True, index=c.index, columns=c.columns)
    assert labels(c, c, e, 4, .5)['status'].iloc[0, 0] == 2
    c.iloc[2, 0] = 12
    e.iloc[2, 0] = False
    assert labels(c, c, e, 4, .5)['status'].iloc[0, 0] == 2
    e.iloc[0, 0] = False
    assert labels(c, c, e, 4, .5)['status'].iloc[0, 0] == 0


def test_dual_price_disagreement_and_unexplained_jump_are_unknown():
    c = frame([10, 11, 12, 13, 15, 16])
    o = c.copy()
    o.iloc[2, 0] *= 1.01
    assert labels(c, o, c.notna(), 4, .5)['status'].iloc[0, 0] == 3
    c.iloc[1, 0] = 15
    assert labels(c, c, c.notna(), 4, .5)['status'].iloc[0, 0] == 3


def test_benchmark_interior_halt_does_not_invent_or_require_price():
    c = frame([10, 11, 12, 13, 15, 16])
    c.iloc[2, 1] = np.nan
    e = pd.DataFrame(True, index=c.index, columns=c.columns)
    assert labels(c, c, e, 4, .5)['status'].iloc[0, 0] == 5
    c.iloc[4, 1] = np.nan
    assert labels(c, c, e, 4, .5)['status'].iloc[0, 0] == 2


def test_nonoverlap_skips_boundary_but_keeps_later_wave():
    assert nonoverlap_starts([0, 1, 4, 5, 9, 10], 4) == [0, 5, 10]


def test_last_close_signal_not_counted_as_captured_before_end():
    c = frame([10]*6).astype(bool)
    gates = {k: c.copy() for k in ['quality', 'breakout', 'relative', 'volume', 'signal']}
    gates['signal'].iloc[:4, 0] = False
    counts, reason = window_gate_counts(gates, 0, 0, 4)
    assert counts['signal'] == 0 and reason == 'signal'
    gates['quality'].iloc[:4, 0] = False
    assert window_gate_counts(gates, 0, 0, 4)[1] == 'quality'


def test_incomplete_axes_rejected():
    c = frame([10]*6)
    with pytest.raises(ValueError, match='identical axes'):
        labels(c, c.iloc[1:], c.notna(), 2, .5)


def test_future_prices_do_not_change_past_causal_gates():
    n = 210
    c = frame(10*np.exp(np.arange(n)*.006))
    c['0050'] = 10*np.exp(np.arange(n)*.001)
    volume = c*0+10_000_000
    volume.iloc[160, 0] *= 3
    frames = {k: c.copy() for k in ['close-official', 'close-quality', 'raw-close']}
    frames.update({'raw-volume': volume, 'eligibility': c.notna()})
    companies = pd.DataFrame([{'stock_id': '1234', 'listed_date': pd.Timestamp('2000-01-01')}])
    original = causal_gates(frames, companies)
    assert original['gates']['signal'].iloc[160, 0]
    changed = {k: v.copy() for k, v in frames.items()}
    for k in ['close-official', 'close-quality', 'raw-close', 'raw-volume']:
        changed[k].iloc[170:] *= .1
    changed = causal_gates(changed, companies)
    truncated = causal_gates({k: v.iloc[:170].copy() for k, v in frames.items()}, companies)
    for k, v in original['gates'].items():
        pd.testing.assert_frame_equal(v.iloc[:170], changed['gates'][k].iloc[:170])
        pd.testing.assert_frame_equal(v.iloc[:170], truncated['gates'][k])


def test_unknown_return_not_mistaken_for_loser():
    x = label_statistics([5, 4, 2, 1], [.6, -.1, None, None])
    assert x['known'] == 2 and x['negative_price_return'] == 1 and x['surge_rate'] == .5
