import pandas as pd
import pytest

from skills.liquidity_candidates import filter_candidates


def fixture():
    days = pd.bdate_range('2024-01-01', periods=23)
    raw = pd.DataFrame({'1101': 100., '1102': 100.}, index=days)
    volume = raw * 1_000
    volume.iloc[20] = 10_000_000
    entries = [dict(event_id=sid, members=[sid], signal_date=str(days[20].date()),
                    entry_date=str(days[21].date()), priority=1.) for sid in raw]
    return raw, volume, entries


def test_signal_day_spike_is_rejected_even_with_future_liquidity():
    raw, volume, entries = fixture()
    volume.iloc[21:] = 1_000_000_000
    result, decisions = filter_candidates(raw, volume, entries, entries[0]['signal_date'])
    assert result['original'] == entries
    assert not result['median50m'] and not result['prior50m']
    assert decisions[0]['values']['mean20'] > 50_000_000
    assert not any(decisions[0]['passes'].values())


def test_retained_candidates_keep_exact_order_and_next_session():
    raw, volume, entries = fixture()
    volume['1102'] = 500_000
    result, _ = filter_candidates(raw, volume, entries, entries[0]['signal_date'])
    for arm in ('median50m', 'prior50m', 'persistent50m'):
        assert result[arm] == entries[1:]
    result['original'][0]['members'][0] = 'changed'
    assert entries[0]['members'] == ['1101']
    entries[0]['entry_date'] = entries[0]['signal_date']
    with pytest.raises(ValueError, match='one market session'):
        filter_candidates(raw, volume, entries, '2024-12-31')


def test_missing_prior_day_is_unknown_not_zero_and_future_is_irrelevant():
    raw, volume, entries = fixture()
    volume[:] = 500_000
    volume.iloc[0, 0] = float('nan')  # Outside today's 20 days; inside prior 20.
    full, diagnostic = filter_candidates(raw, volume, entries, '2024-12-31')
    assert diagnostic[0]['unknown_features'] == ['prior_mean20']
    assert diagnostic[0]['values']['prior_mean20'] is None
    assert len(full['median50m']) == 2 and len(full['prior50m']) == 1
    prefix, evidence = filter_candidates(raw.iloc[:22], volume.iloc[:22], entries, '2024-12-31')
    assert full == prefix and evidence == diagnostic
