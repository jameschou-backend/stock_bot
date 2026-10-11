from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from skills.all_entry_prior_lookup import equivalent_type
from skills.million_replay import Replay
from skills.residual_tick_replay import TickPlanning


class Original(TickPlanning, Replay):
    pass


def outcome(provider, day, sid):
    try:return ('value', provider.prior(day, sid))
    except Exception as exc:return (type(exc).__name__, str(exc))


def test_exact_values_and_errors_for_missing_invalid_and_corporate_dates():
    days = pd.date_range('2024-01-01', periods=9)
    fields = {'close':pd.DataFrame({'2330':[2., np.nan, 4., 0., -1., np.inf, 8., 9., 100.]}, index=days)}
    def reference(sid, day, value):
        if day == '2024-01-09':raise ValueError('Unresolved action')
        return value / 2 if day == '2024-01-08' else value
    old, new = Original.__new__(Original), equivalent_type(Original).__new__(equivalent_type(Original))
    for obj in (old, new):
        obj.positions = dict(zip(days, range(len(days))))
        obj.days, obj.fields = days, fields
        obj.corporate = SimpleNamespace(reference_price=reference)
    for day in [*days, pd.Timestamp('2024-01-10')]:
        for sid in ('2330', '9999'):
            assert outcome(old, day, sid) == outcome(new, day, sid)
    assert new.prior(days[2], '2330') is None
    assert new.prior(days[7], '2330') == 4.
    # Today's close must never influence today's planning price.
    fields['close'].at[days[7], '2330'] = 9999.
    assert new.prior(days[7], '2330') == 4.


def test_unknown_prior_chain_cannot_be_silently_short_circuited():
    class Other(Original):
        def prior(self, *args):return 1.
    with pytest.raises(ValueError, match='renewed equivalence'):equivalent_type(Other)
