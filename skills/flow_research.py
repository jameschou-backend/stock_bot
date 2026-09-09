"""Fixed research contrasts. No fitting, network calls, or production writes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skills.rule_research import scores_for


FLOW_NAMES = {
    'price': '價格強勢',
    'trust': '價格強勢＋投信',
    'volume': '價格強勢＋放量',
    'trust_volume': '價格強勢＋投信＋放量',
}


def company_universe(twse: list, tpex: list) -> pd.DataFrame:
    """Current official cohort, NOT a survivorship-free historical universe."""
    parts = []
    for rows, market, mapping in [
        (twse, 'TWSE', {'公司代號': 'stock_id', '公司簡稱': 'name',
                        '上市日期': 'listed_date', '產業別': 'industry'}),
        (tpex, 'TPEX', {'SecuritiesCompanyCode': 'stock_id', 'CompanyAbbreviation': 'name',
                        'DateOfListing': 'listed_date', 'SecuritiesIndustryCode': 'industry'}),
    ]:
        frame = pd.DataFrame(rows)[list(mapping)].rename(columns=mapping)
        frame = frame[frame.stock_id.str.fullmatch(r'[0-9]{4}')
                      & ~frame.name.str.contains('-DR', case=False, regex=False)].copy()
        frame['listed_date'] = pd.to_datetime(frame.listed_date, format='%Y%m%d', errors='raise')
        frame['market'] = market
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True).sort_values('stock_id')
    if result.empty or result.listed_date.isna().any() or result.stock_id.duplicated().any():
        raise ValueError('Official company identities/dates are empty, missing or duplicated')
    return result


def listing_mask(days: pd.DatetimeIndex, ids: pd.Index, companies: pd.DataFrame) -> pd.DataFrame:
    dates = companies.set_index('stock_id').listed_date.reindex(ids)
    missing = dates[dates.isna()].index.difference(['0050'])
    if len(missing):
        raise ValueError('Missing official listing dates: ' + ', '.join(missing))
    # 0050 is explicitly the benchmark, not an ordinary-stock candidate.
    dates.loc[dates.index == '0050'] = pd.Timestamp.min
    return pd.DataFrame(days.to_numpy()[:, None] >= dates.to_numpy()[None, :], index=days, columns=ids)


def flow_scores(close, raw_close, volume, high, low, trust_net):
    frames = (raw_close, volume, high, low, trust_net)
    if any(not f.index.equals(close.index) or not f.columns.equals(close.columns) for f in frames):
        raise ValueError('Research input matrices must align')
    base = scores_for(close, raw_close * volume)['near_high']
    base.loc[:, '0050'] = np.nan
    volume20 = volume.rolling(20, min_periods=20).sum()
    trust_ratio = trust_net.rolling(20, min_periods=20).sum() / volume20.where(volume20 > 0)
    trust = (trust_ratio.ge(.01)
             & trust_net.rolling(5, min_periods=5).sum().gt(0)
             & trust_net.gt(0).rolling(5, min_periods=5).sum().ge(3))
    # A missing institutional row is not a zero. Complete 20-day sums are required.
    past_volume = volume.shift(5).rolling(20, min_periods=20).mean()
    volume_ratio = volume.rolling(5, min_periods=5).mean() / past_volume.where(past_volume > 0)
    close_location = (raw_close - low) / (high - low).where(high > low)
    expansion = volume_ratio.ge(1.5) & close.gt(close.shift(5)) & close_location.ge(.7)
    return {'price': base, 'trust': base.where(trust), 'volume': base.where(expansion),
            'trust_volume': base.where(trust & expansion)}


def rolling_comparison(curve, benchmark):
    nav = curve.set_index('date').equity
    other = benchmark.set_index('date').equity
    if not nav.index.equals(other.index):
        raise ValueError('Benchmark and strategy dates must align')
    result = {}
    for n in (252, 756):
        a, b = nav.pct_change(n, fill_method=None), other.pct_change(n, fill_method=None)
        good = a.notna() & b.notna()
        excess = a[good] - b[good]
        result[str(n)] = {'overlapping_windows': int(good.sum()),
                          'win_fraction': float((excess > 0).mean()) if len(excess) else None,
                          'median_excess': float(excess.median()) if len(excess) else None}
    return result


def portfolio_diagnostics(run, close, companies):
    """Reconcile each stock's P&L including residual holdings and identify bad-price exposure."""
    names = companies.set_index('stock_id').name.to_dict()
    industries = companies.set_index('stock_id').industry.to_dict()
    units, pnl, anomalies, industry_peaks = {}, {}, [], {}
    trades = {}
    for trade in run.trades:
        trades.setdefault(pd.Timestamp(trade['date']), []).append(trade)
    previous = close.ffill().shift(1)
    nav = run.curve.set_index('date').equity
    total_turnover = 0.
    for day in nav.index[1:]:
        for sid, quantity in units.items():
            before, after = previous.at[day, sid], close.at[day, sid]
            if quantity > 1e-12 and pd.notna(before) and pd.notna(after) and abs(after / before - 1) > .5:
                anomalies.append({'date': str(day.date()), 'stock_id': sid,
                                  'adjusted_return': float(after / before - 1),
                                  'pnl_initial_equity': float(quantity * (after - before))})
        for trade in trades.get(day, []):
            sid, amount, cost = trade['stock_id'], trade['notional_initial_equity'], trade['cost_initial_equity']
            sign = 1 if trade['side'] == 'buy' else -1
            units[sid] = units.get(sid, 0.) + sign * amount / close.at[day, sid]
            if abs(units[sid]) < 1e-12:
                del units[sid]
            pnl[sid] = pnl.get(sid, 0.) - sign * amount - cost
            total_turnover += amount / nav.at[day]
        sector_values = {}
        for sid, quantity in units.items():
            mark = close.at[day, sid]
            if pd.isna(mark):
                mark = previous.at[day, sid]
            if quantity > 1e-12 and pd.notna(mark):
                sector = industries.get(sid, 'unknown')
                sector_values[sector] = sector_values.get(sector, 0.) + quantity * mark / nav.at[day]
        for sector, weight in sector_values.items():
            industry_peaks[sector] = max(industry_peaks.get(sector, 0.), weight)
    for sid, quantity in units.items():
        pnl[sid] += quantity * close[sid].ffill().iloc[-1]
    if not np.isclose(sum(pnl.values()), run.summary['total_return'], atol=1e-8):
        raise AssertionError('Per-stock P&L does not reconcile to portfolio NAV')
    top = sorted(pnl.items(), key=lambda x: (-x[1], x[0]))[:3]
    positive = sum(max(x, 0) for x in pnl.values())
    ratio = run.curve.iloc[1:].cash / run.curve.iloc[1:].equity
    return {'average_cash_fraction': float(ratio.mean()),
            'annual_one_way_turnover_using_end_day_nav': float(total_turnover / 2 / ((len(nav)-1)/252)),
            'top_profit_stocks': [{'stock_id': sid, 'name': names.get(sid, sid),
                                   'pnl_initial_equity': value} for sid, value in top],
            'top3_fraction_of_positive_pnl': sum(max(v, 0) for _, v in top) / positive if positive else None,
            'per_stock_pnl_sum': float(sum(pnl.values())),
            'large_move_exposures': anomalies,
            'peak_industry_weights': industry_peaks}
