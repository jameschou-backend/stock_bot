import numpy as np
import pandas as pd
import pytest

from skills.flow_research import (company_universe, listing_mask, flow_scores,
                                 rolling_comparison, portfolio_diagnostics)
from skills.rule_research import simulate


def inputs():
    days = pd.bdate_range('2020-01-01', periods=340)
    close = pd.DataFrame({'0050': np.linspace(10, 20, len(days)),
                          '2330': np.linspace(10, 25, len(days))}, index=days)
    volume = close * 0 + 10_000_000
    volume.iloc[-5:] *= 2
    net = close * 0 + 200_000
    return close, close.copy(), volume, close + .1, close - 1, net


def test_official_universe_excludes_drs_and_rejects_duplicate_identity():
    rows = [{'公司代號': sid, '公司簡稱': name, '上市日期': '20220107', '產業別': '24'}
            for sid, name in [('3592', '瑞鼎'), ('9103', '美德醫療-DR'), ('910322', '康師傅-DR')]]
    otc = [{'SecuritiesCompanyCode': '5314', 'CompanyAbbreviation': '世紀',
            'DateOfListing': '19960916', 'SecuritiesIndustryCode': '24'}]
    company = company_universe(rows, otc)
    assert company.stock_id.tolist() == ['3592', '5314']
    with pytest.raises(ValueError, match='duplicated'):
        company_universe(rows + rows[:1], otc)


def test_listing_date_masks_indicator_history_not_only_entry_dates():
    args = inputs()
    day = args[0].index[100]
    company = pd.DataFrame({'stock_id': ['2330'], 'listed_date': [day]})
    mask = listing_mask(args[0].index, args[0].columns, company)
    scored = flow_scores(*(frame.where(mask) for frame in args))
    # There are only 240 post-listing prices; no 252-day signal is permitted.
    assert scored['price']['2330'].isna().all()
    assert mask['0050'].all()
    with pytest.raises(ValueError, match='Missing official'):
        listing_mask(args[0].index, pd.Index(['9999']), company)


def test_missing_institutional_row_cannot_be_treated_as_no_buying():
    args = inputs()
    good = flow_scores(*args)
    assert good['trust']['2330'].iloc[-1] > 0
    args[-1].iloc[-10, 1] = np.nan
    bad = flow_scores(*args)
    assert pd.isna(bad['trust']['2330'].iloc[-1])
    assert bad['price']['2330'].iloc[-1] == good['price']['2330'].iloc[-1]


def test_future_volume_and_trust_cannot_change_earlier_scores():
    args = inputs()
    before = flow_scores(*args)
    args[2].iloc[320:] *= 50
    args[-1].iloc[320:] *= -100
    after = flow_scores(*args)
    for name in before:
        pd.testing.assert_frame_equal(before[name].iloc[:320], after[name].iloc[:320])


def test_volume_comparison_excludes_recent_event_window_and_requires_up_close():
    args = inputs()
    # Recent 5-day volume = 1.5x the preceding 20; including recent days in
    # the denominator would incorrectly fail this deliberately boundary case.
    args[2].iloc[-5:] = 15_000_000
    assert flow_scores(*args)['volume']['2330'].iloc[-1] > 0
    args[1].iloc[-1, 1] = args[4].iloc[-1, 1]  # Close at the bottom of its range.
    assert pd.isna(flow_scores(*args)['volume']['2330'].iloc[-1])


def test_rolling_comparison_uses_common_full_windows():
    dates = pd.bdate_range('2020-01-01', periods=800)
    a = pd.DataFrame({'date': dates, 'equity': 1.001 ** np.arange(800)})
    b = pd.DataFrame({'date': dates, 'equity': 1.0005 ** np.arange(800)})
    result = rolling_comparison(a, b)
    assert result['252']['overlapping_windows'] == 548
    assert result['756']['overlapping_windows'] == 44
    assert result['252']['win_fraction'] == 1


def test_attribution_reconciles_and_detects_held_price_anomaly():
    close = pd.DataFrame({'2330': [10., 10., 20., 21., np.nan]},
                         index=pd.bdate_range('2020-01-01', periods=5))
    flags = close.notna()
    run = simulate(close, flags, close, start=close.index[1], topn=1)
    company = pd.DataFrame({'stock_id': ['2330'], 'name': ['台積電'], 'industry': ['24']})
    stats = portfolio_diagnostics(run, close, company)
    assert stats['per_stock_pnl_sum'] == pytest.approx(run.summary['total_return'])
    assert len(stats['large_move_exposures']) == 1
    assert stats['large_move_exposures'][0]['adjusted_return'] == 1
    assert run.summary['unliquidated_positions'] == 1
