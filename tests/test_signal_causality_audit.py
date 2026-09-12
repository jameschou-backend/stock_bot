from copy import deepcopy

import pandas as pd
import pytest

from scripts.audit_signal_causality import project, transform
from scripts.prepare_million_signals import build_signals
from tests.test_million_signals import signal_inputs


@pytest.mark.parametrize('mode', ['truncate', 'mutate'])
def test_real_builder_keeps_signal_with_no_future_prices_and_no_source_mutation(mode):
    *frames, companies = signal_inputs()
    originals = [f.copy() for f in frames]
    full = build_signals(*frames, companies, start='2022-01-03', signal_end='2022-01-31')
    altered = transform(frames, '2022-01-05', mode)
    rebuilt = build_signals(*altered, companies, start='2022-01-03', signal_end='2022-01-05')
    expected = project(full, '2022-01-05')
    assert expected['entries']  # A comparison of two empty candidate pools is insufficient.
    assert project(rebuilt, '2022-01-05') == expected
    for frame, original in zip(frames, originals):
        pd.testing.assert_frame_equal(frame, original)
    if mode == 'truncate':
        for frame in altered:
            assert str(frame.index[-1].date()) == '2022-01-06'
            assert frame.iloc[-1].isna().all()


def test_projection_retains_decision_priority_and_group_evidence():
    result = build_signals(*signal_inputs(), start='2022-01-03', signal_end='2022-01-05')
    expected = project(result, '2022-01-05')
    changed = deepcopy(result)
    changed['entries'][0]['priority'] += 1
    assert project(changed, '2022-01-05') != expected
    changed = deepcopy(result)
    changed['diffusion']['groups'][0]['selected_ids'] = ['9999']
    assert project(changed, '2022-01-05') != expected
