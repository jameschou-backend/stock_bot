import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from skills.market_context import asof
from scripts.research_surge_anatomy import clean


def inputs():
    days = pd.bdate_range('2024-01-01', periods=140)
    c = pd.DataFrame({'6446': np.linspace(100, 200, len(days)), '1234': np.linspace(100, 110, len(days)),
        '5678': 100., '2330': np.linspace(100, 120, len(days)), '0050': np.linspace(100, 130, len(days))}, index=days)
    raw = c.copy(); volume = c * 0 + 1_000_000
    companies = pd.DataFrame({'stock_id': ['6446', '1234', '5678', '2330'], 'name': ['Case', 'Peer A', 'Peer B', 'Other'],
        'industry': ['22', '22', '22', '24'], 'listed_date': pd.Timestamp('2020-01-01')})
    tr = c['0050'] * 100
    return c, raw, volume, companies, tr


def test_future_prices_cannot_change_prior_market_or_leaders():
    c, raw, volume, companies, tr = inputs(); day = c.index[110]
    baseline = asof(c, raw, volume, companies, tr, day)
    for f in (c, raw, volume): f.loc[f.index > day, '5678'] *= 100
    tr.loc[tr.index > day] *= 5
    actual = asof(c, raw, volume, companies, tr, day)
    assert clean(actual[0]) == clean(baseline[0])
    for a, b in zip(actual[1:], baseline[1:]): assert_frame_equal(a, b)


def test_case_does_not_make_peer_breadth_or_leadership_stronger():
    c, raw, volume, companies, tr = inputs(); day = c.index[-1]
    result, _, leaders = asof(c, raw, volume, companies, tr, day)
    assert result['peers_expected'] == 2
    assert result['peers_positive20'] == .5
    assert result['stock_relative_peers20'] > 0
    assert leaders[leaders.leader_kind.eq('price')].iloc[0].stock_id == '6446'
    assert result['stock_price_rank'] == 1


def test_stale_index_remains_unknown_without_substituting_0050():
    c, raw, volume, companies, tr = inputs()
    result, _, _ = asof(c, raw, volume, companies, tr.iloc[:-1], c.index[-1])
    assert np.isnan(result['taiex_tr_return20'])
    assert result['etf0050_return20'] > 0


def test_money_is_an_observed_activity_share_and_missing_sector_is_not_zero():
    c, raw, volume, companies, tr = inputs(); day = c.index[-1]
    result, sectors, _ = asof(c, raw, volume, companies, tr, day)
    a = raw.drop(columns='0050') * volume.drop(columns='0050')
    expected = a[['6446', '1234', '5678']].tail(5).sum().sum() / a.tail(5).sum().sum()
    assert np.isclose(result['bio_share5'], expected)
    assert np.isclose(sectors.share5.sum(), 1.)
    raw.loc[day, ['6446', '1234', '5678']] = np.nan
    missing, _, _ = asof(c, raw, volume, companies, tr, day)
    assert np.isnan(missing['bio_share5'])
    assert missing['bio_amount_known_today'] == 0
    assert missing['bio_amount_coverage_min25'] == 0


def test_not_yet_listed_company_cannot_be_a_past_leader():
    c, raw, volume, companies, tr = inputs(); day = c.index[110]
    companies.loc[companies.stock_id.eq('5678'), 'listed_date'] = c.index[120]
    raw['5678'] *= 100
    result, _, leaders = asof(c, raw, volume, companies, tr, day)
    assert '5678' not in set(leaders.stock_id)
    assert result['peers_expected'] == 1
