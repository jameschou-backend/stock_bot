"""Price-level broker rows -> causal, explicitly limited branch diagnostics.

Offline research only. Net purchases are a flow, not beneficial ownership or
verified intraday round trips. Do not use legacy raw_broker_trades aggregates.
"""
import numpy as np
import pandas as pd


NAMED_BRANCHES = {
    '9A9g': '永豐金內湖', '9853': '元大南屯', '961F': '富邦公益',
    '7001': '兆豐嘉義', '984K': '元大館前',
}


def branch_snapshot(raw, stock_id, signal_date):
    required = {'date', 'stock_id', 'securities_trader_id', 'buy', 'sell'}
    unknown = lambda reason: dict(known=False, reason=reason)
    if raw.empty:
        return unknown('missing_raw_source')
    if not required.issubset(raw):
        raise ValueError('Missing broker columns')
    if (set(raw.stock_id.astype(str)) != {stock_id}
            or set(raw.date.astype(str)) != {signal_date}
            or raw.securities_trader_id.isna().any()
            or raw.securities_trader_id.astype(str).str.strip().eq('').any()):
        raise ValueError('Wrong broker snapshot identity')
    values = raw[['buy', 'sell']].apply(pd.to_numeric, errors='raise')
    if (not np.isfinite(values).all().all() or values.lt(0).any().any()
            or values.ge(2**53).any().any()
            or values.ne(np.floor(values)).any().any()):
        raise ValueError('Broker quantities must be finite nonnegative integer shares')
    # Validate with Python integers before int64 groupby can overflow.
    if any(sum(map(int, values[c])) >= 2**53 for c in ('buy', 'sell')):
        raise ValueError('Broker snapshot exceeds exact quantity range')
    grouped = values.astype('int64').groupby(raw.securities_trader_id.astype(str)).sum()
    buy, sell = int(grouped.buy.sum()), int(grouped.sell.sum())
    if buy <= 0 or abs(buy-sell) > max(1, buy*.001):
        return unknown('raw_market_imbalance')
    grouped['net'] = grouped.buy - grouped.sell
    buyers = grouped[grouped.net.gt(0)].reset_index().sort_values(
        ['net', 'securities_trader_id'], ascending=[False, True])
    if len(buyers) < 5:
        return unknown('fewer_than_five_positive_branches')
    top = buyers.head(5)
    ranks = {sid: i for i, sid in enumerate(buyers.securities_trader_id, 1)}
    named = {}
    for sid, label in NAMED_BRANCHES.items():
        row = grouped.loc[sid] if sid in grouped.index else None
        named[sid] = dict(label=label, observed=row is not None,
            buy=int(row.buy) if row is not None else None,
            sell=int(row.sell) if row is not None else None,
            net=int(row.net) if row is not None else None,
            positive_rank=ranks.get(sid),
            positive_observed=bool(row is not None and row.net > 0))
    share = float(top.net.sum()/buy)
    direction = float(top.net.sum()/(top.buy.sum()+top.sell.sum()))
    return dict(known=True, stock_id=stock_id, signal_date=signal_date,
        raw_rows=len(raw), branch_count=len(grouped),
        buy_branch_count=int(grouped.net.gt(0).sum()),
        sell_branch_count=int(grouped.net.lt(0).sum()),
        total_buy_shares=buy, total_sell_shares=sell,
        market_net_shares=buy-sell,
        top5_net_share=share, top5_directional_ratio=direction,
        top5=top.to_dict('records'), named=named,
        named_positive=any(x['positive_observed'] for x in named.values()),
        concentrated_directional=share >= .10 and direction >= .30)
