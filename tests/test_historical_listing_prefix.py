import numpy as np
import pandas as pd
import pytest

from skills.historical_listing_prefix import restore


def sources():
    days = pd.bdate_range('2021-01-04', periods=10)
    raw = pd.DataFrame({'0050': 100., '1101': np.arange(10) + 50.}, index=days)
    frames = {'raw-close': raw.copy(), 'raw-volume': raw * 10000,
              'close-official': raw.copy(), 'close-quality': raw * .25}
    for frame in frames.values():
        frame.loc[days[:4], '1101'] = np.nan
    prefix = pd.DataFrame({'stock_id': '1101', 'date': days[:4], 'close': raw['1101'].iloc[:4].values,
                           'volume': (raw['1101'].iloc[:4] * 10000).values})
    quality = {'1101': raw['1101'] * 2}
    plan = {'rows': [{'stock_id': '1101', 'start': str(days[0].date()), 'end_exclusive': str(days[4].date())}]}
    companies = pd.DataFrame([{'stock_id': '1101', 'listed_date': days[4]}])
    events = pd.DataFrame(columns=['stock_id', 'event_date', 'ratio'])
    return frames, companies, events, prefix, quality, plan


def test_restore_prefix_preserves_all_existing_history_and_units():
    args = sources()
    before = {k: v.copy() for k, v in args[0].items()}
    result, companies, anchors = restore(*args)
    for key, original in before.items():
        assert args[0][key].equals(original)  # Parent snapshot is untouched.
        assert result[key].iloc[4:].equals(original.iloc[4:])
        assert result[key]['1101'].iloc[:4].notna().all()
    assert companies.listed_date.iloc[0] == result['raw-close'].index[0]
    assert args[1].listed_date.iloc[0] != companies.listed_date.iloc[0]
    assert anchors[0]['restored_rows'] == 4
    assert np.allclose(result['close-quality']['1101'], result['raw-close']['1101'] * .25)


def test_adjusted_unit_rescale_has_no_effect_on_restored_series():
    args = list(sources())
    before, _, _ = restore(*args)
    args[4] = {'1101': args[4]['1101'] * 8}
    after, _, _ = restore(*args)
    assert before['close-quality'].equals(after['close-quality'])


def test_existing_quotes_cannot_be_overwritten():
    args = sources()
    args[0]['raw-close'].iloc[0, 1] = 99.
    with pytest.raises(ValueError, match='overwrite'):
        restore(*args)


def test_missing_price_unit_anchor_is_a_hard_failure():
    args = sources()
    args[4]['1101'].iloc[4:] = np.nan
    with pytest.raises(ValueError, match='No common adjusted-price unit anchor'):
        restore(*args)


def test_post_transfer_row_cannot_enter_prefix_repair():
    args = list(sources())
    args[3].loc[0, 'date'] = args[0]['raw-close'].index[4]
    with pytest.raises(ValueError, match='outside the verified transfer interval'):
        restore(*args)


def test_entirely_missing_old_quality_column_uses_explicit_independent_source():
    args = sources()
    args[0]['close-quality']['1101'] = np.nan
    result, _, anchors = restore(*args)
    assert result['close-quality']['1101'].equals(args[4]['1101'])
    assert anchors[0]['mode'] == 'previous_quality_column_entirely_missing'
