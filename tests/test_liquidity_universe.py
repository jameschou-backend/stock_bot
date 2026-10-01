import pytest

from skills.liquidity_universe import expanded_entries


def test_expanded_identity_cannot_be_replaced_with_grouped_population():
    with pytest.raises(ValueError, match='identity'):
        expanded_entries(dict(original=[dict(event_id='2019-03-g009-3362')]))


def test_expanded_mapping_keeps_events_and_order_instead_of_resorting():
    events = [dict(event_id='liquid_universe-2024-01-02-1102'),
              dict(event_id='liquid_universe-2024-01-02-1101')]
    source = dict(original=events, cap40=events, median50m=events[1:],
                  prior50m=events, persistent50m=events[1:])
    result = expanded_entries(source)
    assert result['liquid_universe'] == events
    assert result['median50m'] == events[1:]
    assert 'original' not in result and 'cap40' not in result
