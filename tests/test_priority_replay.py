import numpy as np
import pandas as pd
import pytest
from skills.priority_replay import group_strength_scores
from scripts.research_priority import rolling_comparison


def test_scores_use_original_signal_calendar_and_ignore_future():
    days = pd.bdate_range('2025-01-01', periods=25)
    close = pd.DataFrame(100., index=days, columns=['1101', '1102'])
    close.loc[days[20], :] = [130., 110.]
    entry = dict(event_id='a', signal_date=str(days[20].date()), members=['1101'], group_members=['1101','1102'])
    initial = group_strength_scores(close, [entry])
    assert initial['a']['score'] == pytest.approx(.1)
    close.loc[days[21]:, :] = [1., 9999.]
    assert initial == group_strength_scores(close, [entry])
    close.loc[days[0], '1101'] = np.nan
    assert group_strength_scores(close, [entry])['a']['score'] is None


def test_missing_group_members_are_not_silently_dropped_from_coverage():
    days = pd.bdate_range('2025-01-01', periods=21)
    close = pd.DataFrame(100., index=days, columns=['1101','1102','1103'])
    entry = dict(event_id='a', signal_date=str(days[-1].date()), members=['1101'],
                 group_members=['1101','1102','1103','1104','1105'])
    assert group_strength_scores(close, [entry])['a']['score'] is None
    close['1104'] = 100.
    assert group_strength_scores(close, [entry])['a']['score'] == 0.


def test_rolling_comparison_requires_same_calendar():
    a = dict(daily=[dict(date='2025-01-01', nav=100)])
    b = dict(daily=[dict(date='2025-01-02', nav=100)])
    with pytest.raises(ValueError, match='calendar'):
        rolling_comparison(a, b)
