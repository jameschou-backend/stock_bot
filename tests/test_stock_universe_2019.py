from copy import deepcopy
import pandas as pd
import pytest
from skills.stock_universe_2019 import generate
from test_candidate_quality import frames


def inputs():
    values, companies = frames()
    days = pd.bdate_range('2024-01-02', periods=len(values['close-official']))
    for f in values.values():
        f.index = days
        f['5678'] = f['1234']
    companies = pd.concat([companies, companies.assign(stock_id='5678')], ignore_index=True)
    companies['name'] = companies['stock_id']
    values['raw-volume'].loc[days[190], ['1234', '5678']] *= 3
    signals = dict(entries=[], diffusion=dict(groups=[dict(month=str(m),
        cutoff_date=str((m.start_time-pd.Timedelta(days=1)).date()), selected_ids=['1234'],
        exclusions={}) for m in days.to_period('M').unique()]))
    return values, companies, signals, days


def test_expanded_universe_only_changes_membership_and_is_next_session():
    values, companies, signals, days = inputs()
    out = generate(values, companies, signals, str(days[-2].date()))
    assert [e['members'][0] for e in out['ungrouped300']] == ['1234']
    assert [e['members'][0] for e in out['liquid_universe']] == ['1234', '5678']
    for arm in ('ungrouped300', 'liquid_universe'):
        assert all(e['signal_date'] == str(days[190].date()) and e['entry_date'] == str(days[191].date()) for e in out[arm])


def test_future_truncation_cannot_change_candidates():
    values, companies, signals, days = inputs()
    cutoff = str(days[191].date())
    full = generate(values, companies, signals, str(days[-2].date()))
    limited = generate({k: f.iloc[:193] for k, f in values.items()}, companies, signals, cutoff)
    for arm in full:
        assert [e for e in full[arm] if e['signal_date'] <= cutoff] == limited[arm]


def test_zero_volume_is_unknown_history_not_a_low_comparison_base():
    values, companies, signals, days = inputs()
    values['raw-volume'].loc[days[189], ['1234', '5678']] = 0
    out = generate(values, companies, signals, str(days[-2].date()))
    assert not out['ungrouped300'] and not out['liquid_universe']


def test_rejects_monthly_membership_from_future():
    values, companies, signals, days = inputs()
    for g in signals['diffusion']['groups']:
        g['cutoff_date'] = '2025-01-01'
    with pytest.raises(ValueError, match='before signal'):
        generate(values, companies, signals, str(days[-2].date()))


def test_same_technical_filter_as_prior_2024_research():
    from scripts.prepare_rotation_2024 import generate as old_generate
    values, companies, signals, days = inputs()
    before = deepcopy(signals)
    old = old_generate(values, companies, signals, str(days[-2].date()))['relaxed']
    new = generate(values, companies, signals, str(days[-2].date()))['ungrouped300']
    def keys(rows):
        return sorted((r['signal_date'], r['entry_date'], r['members'][0], r['priority']) for r in rows)
    assert keys(old) == keys(new)
    assert signals == before
