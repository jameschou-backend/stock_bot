"""As-of market breadth, observed turnover rotation and backward-looking leaders."""
import numpy as np
import pandas as pd


def trailing_mean(frame, n):
    return frame.tail(n).sum(min_count=n) / n


def trailing_return(frame, n):
    if len(frame) <= n:
        return pd.Series(np.nan, index=frame.columns)
    return (frame.iloc[-1] / frame.iloc[-n-1] - 1).where(frame.tail(n+1).notna().sum().eq(n+1))


def rate(values):
    known = values.dropna()
    return float(known.mean()) if len(known) else np.nan


def price_table(close):
    c = close.where(np.isfinite(close) & close.gt(0))
    last = c.iloc[-1]
    out = pd.DataFrame({f'return{n}': trailing_return(c, n) for n in (5, 20, 60)})
    for n in (20, 60):
        ma = trailing_mean(c, n)
        out[f'distance_ma{n}'] = last / ma - 1
        out[f'above{n}'] = last.gt(ma).astype(float).where(last.notna() & ma.notna())
    old = c.iloc[:-1]
    hi20 = old.tail(20).max().where(old.tail(20).notna().sum().eq(20))
    low20 = old.tail(20).min().where(old.tail(20).notna().sum().eq(20))
    hi60 = old.tail(60).max().where(old.tail(60).notna().sum().eq(60))
    out['new_high20'] = last.gt(hi20).astype(float).where(last.notna() & hi20.notna())
    out['distance_high60'] = last / hi60 - 1
    out['range20'] = hi20 / low20 - 1
    return out


def group_stats(prices, ids):
    p = prices.loc[ids]
    return dict(expected=len(ids), return20_known=int(p.return20.notna().sum()),
        median20=p.return20.median(), positive20=rate(p.return20.gt(0).astype(float).where(p.return20.notna())),
        above60_known=int(p.above60.notna().sum()), above60=rate(p.above60),
        high20_known=int(p.new_high20.notna().sum()), new_high20=rate(p.new_high20))


def asof(close, raw, volume, companies, taiex, day, sid='6446'):
    """Return context, sector rows and leaders. No future outcomes enter this function."""
    day = pd.Timestamp(day)
    if day not in close.index or not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise ValueError('Invalid as-of date or market calendar')
    if companies.stock_id.duplicated().any() or companies.listed_date.isna().any():
        raise ValueError('Company identity or listing date invalid')
    for f in (raw, volume):
        if not f.index.equals(close.index) or not f.columns.equals(close.columns):
            raise ValueError('Price/volume axes differ')
    if not taiex.index.is_unique or not taiex.index.is_monotonic_increasing:
        raise ValueError('Invalid TAIEX TR calendar')
    active = companies[companies.stock_id.isin(close.columns) & companies.stock_id.ne('0050')
        & companies.stock_id.str.fullmatch(r'\d{4}') & pd.to_datetime(companies.listed_date).le(day)].set_index('stock_id')
    # Current classification is explicit research metadata, not historical membership proof.
    bio_ids = active.index[active.industry.eq('22')].tolist()
    peers = [x for x in bio_ids if x != sid]
    c = close.loc[:day].tail(121)
    prices = price_table(c)
    amount = (raw.loc[c.index] * volume.loc[c.index]).where(
        raw.loc[c.index].gt(0) & volume.loc[c.index].gt(0)
        & np.isfinite(raw.loc[c.index]) & np.isfinite(volume.loc[c.index]))
    amount = amount[active.index]
    listed = pd.DataFrame(c.index.to_numpy()[:, None] >= pd.to_datetime(active.listed_date).to_numpy()[None, :],
        index=c.index, columns=active.index)
    amount = amount.where(listed)
    adv = trailing_mean(amount, 20)
    prices['adv20_proxy'] = adv
    prices['volume_ratio'] = volume.loc[c.index].iloc[-1] / trailing_mean(volume.loc[c.index].iloc[:-1].where(lambda f: f.gt(0)), 20)
    total = amount.sum(axis=1, min_count=1)
    total5 = total.tail(5).sum(min_count=5)
    total_prev20 = total.iloc[:-5].tail(20).sum(min_count=20)

    def activity(ids):
        a = amount[ids]
        daily = a.sum(axis=1, min_count=1)
        recent = daily.tail(5).sum(min_count=5)
        prior = daily.iloc[:-5].tail(20).sum(min_count=20)
        expected = listed[ids].tail(25).sum(axis=1)
        coverage = a.tail(25).notna().sum(axis=1) / expected.replace(0, np.nan)
        return dict(share5=recent / total5 if total5 > 0 else np.nan,
            share_prev20=prior / total_prev20 if total_prev20 > 0 else np.nan,
            amount5_daily=recent / 5, amount_prev20_daily=prior / 20,
            amount_known_today=int(a.iloc[-1].notna().sum()), amount_coverage_min25=coverage.min())

    sectors = []
    for industry, group in active.groupby('industry', sort=True):
        row = dict(industry=industry, **group_stats(prices, group.index), **activity(group.index))
        row['share_change'] = row['share5'] - row['share_prev20']
        sectors.append(row)
    sectors = pd.DataFrame(sectors)
    for key in ('share5', 'share_change', 'median20'):
        sectors[key + '_rank'] = sectors[key].rank(ascending=False, method='min')
    context = {}
    for label, ids in (('market', active.index), ('bio', bio_ids), ('peers', peers)):
        context.update({label + '_' + k: v for k, v in group_stats(prices, ids).items()})
    for label, ids in (('bio', bio_ids), ('peers', peers)):
        context.update({label + '_' + k: v for k, v in activity(ids).items()})
    context['market_amount5_daily'] = total5 / 5
    context['market_amount_prev20_daily'] = total_prev20 / 20
    context['market_amount_coverage_min25'] = (amount.tail(25).notna().sum(axis=1)
        / listed.tail(25).sum(axis=1).replace(0, np.nan)).min()
    for label, stock in (('stock', sid), ('etf0050', '0050')):
        context.update({label + '_' + k: v for k, v in prices.loc[stock].items()})
    tr = taiex.reindex(c.index).to_frame('TAIEX_TR')
    context.update({'taiex_tr_' + k: v for k, v in price_table(tr).loc['TAIEX_TR'].items()})
    context['stock_relative_peers20'] = context['stock_return20'] - context['peers_median20']
    context['stock_relative_market20'] = context['stock_return20'] - context['market_median20']
    context['stock_relative_0050_20'] = context['stock_return20'] - context['etf0050_return20']
    leaders = []
    pool = prices.loc[bio_ids].join(active[['name']])
    pool.index.name = 'stock_id'
    pool = pool.reset_index()
    for kind, key in (('turnover', 'adv20_proxy'), ('price', 'return20')):
        eligible = pool.dropna(subset=[key])
        if kind == 'price':
            eligible = eligible[eligible.adv20_proxy.ge(50e6)]
        ranked = eligible.sort_values([key, 'stock_id'], ascending=[False, True]).copy()
        ranked['rank'] = np.arange(1, len(ranked) + 1)
        position = ranked.loc[ranked.stock_id.eq(sid), 'rank']
        context['stock_' + kind + '_rank'] = int(position.iloc[0]) if len(position) else np.nan
        context['stock_' + kind + '_rank_denominator'] = len(ranked)
        selected = ranked.head(3).copy(); selected['leader_kind'] = kind
        leaders.extend(selected.to_dict('records'))
    return context, sectors, pd.DataFrame(leaders)
