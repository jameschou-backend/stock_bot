import numpy as np
import pandas as pd
import pytest
from scripts.research_pharmaessentia import describe, price_coverage


def timeline():
    rows = []
    for offset in (-1, 0):
        row = dict(event_id='case', offset=offset, above20=1, above60=0,
            ma_stack=0, distance_ma20=.01, distance_ma60=-.02, volume_ratio=2.5)
        for actor in ('foreign', 'trust', 'dealer'):
            row.update({f'{actor}_net5': -100 if offset == -1 else 900,
                f'{actor}_buy_streak': 0 if offset == -1 else 1,
                f'{actor}_streak3': 0, f'{actor}_net': 1000})
        rows.append(row)
    return pd.DataFrame(rows)


def test_case_join_preserves_failed_and_unknown_horizons_without_using_day_zero_flow():
    events = pd.DataFrame([dict(event_id='case', horizon=20, first_return=-.1),
        dict(event_id='case', horizon=60, first_return=np.nan)])
    r = describe(events, timeline())
    assert len(r) == 2 and r.first_return.iloc[0] == -.1 and pd.isna(r.first_return.iloc[1])
    assert r.prior_foreign_net5.tolist() == [-100, -100]
    assert r.prior_trust_buy_streak.tolist() == [0, 0]
    assert r.foreign_net.tolist() == [1000, 1000]


def test_missing_prior_stays_unknown():
    t = timeline(); t.loc[t.offset.eq(-1), 'trust_net5'] = np.nan
    r = describe(pd.DataFrame([dict(event_id='case')]), t)
    assert pd.isna(r.prior_trust_net5.iloc[0])


def test_duplicate_snapshots_rejected():
    t = timeline()
    with pytest.raises(pd.errors.MergeError):
        describe(pd.DataFrame([dict(event_id='case')]), pd.concat([t, t.iloc[[0]]]))


def test_coverage_does_not_advertise_missing_early_history():
    index = pd.to_datetime(['2022-01-03', '2024-01-25', '2024-01-26', '2025-01-02'])
    f = pd.DataFrame({'6446': [np.nan, 5., 0., 6.]}, index=index)
    actual = price_coverage([f] * 4)['adjusted']
    assert actual == dict(first='2024-01-25', last='2025-01-02', valid_days=2, by_year={2024: 1, 2025: 1})
