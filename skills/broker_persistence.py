"""Causal branch selection and explicit unknown states for broker persistence."""
import math
import numpy as np
import pandas as pd


def select_branches(raw, sid, day):
    required = {'date','stock_id','securities_trader_id','buy','sell'}
    if raw.empty or not required.issubset(raw):
        return [], 'missing_raw_source'
    if (set(raw.stock_id.astype(str)) != {sid} or set(raw.date.astype(str)) != {day}
            or raw.securities_trader_id.isna().any()):
        raise ValueError('Wrong raw broker identity')
    values = raw[['buy','sell']].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values).all().all() or values.lt(0).any().any():
        raise ValueError('Invalid raw broker quantities')
    grouped = values.groupby(raw.securities_trader_id.astype(str)).sum()
    buy, sell = grouped.buy.sum(), grouped.sell.sum()
    if buy <= 0 or abs(buy-sell) > max(1, buy*.001):
        return [], 'raw_market_imbalance'
    grouped['net'] = grouped.buy-grouped.sell
    grouped = grouped.reset_index().sort_values(['net','securities_trader_id'], ascending=[False, True])
    top = grouped[grouped.net.gt(0)].head(5)
    if len(top) != 5:
        return [], 'fewer_than_five_positive_branches'
    return top.to_dict('records'), None


def validate_interval(frame, sid, broker, start, end):
    if frame.empty:
        return pd.DataFrame(columns=['date','buy','sell'])
    required = {'date','stock_id','securities_trader_id','buy_volume','sell_volume'}
    if not required.issubset(frame):
        raise ValueError('Broker interval schema mismatch')
    if set(frame.stock_id.astype(str)) != {sid} or set(frame.securities_trader_id.astype(str)) != {broker}:
        raise ValueError('Broker interval identity mismatch')
    x = frame.rename(columns={'buy_volume':'buy','sell_volume':'sell'})[['date','buy','sell']].copy()
    x['date'] = pd.to_datetime(x.date)
    if x.date.duplicated().any() or x.date.isna().any() or not x.date.between(start, end).all():
        raise ValueError('Duplicate/outside broker dates')
    x[['buy','sell']] = x[['buy','sell']].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(x[['buy','sell']]).all().all() or x[['buy','sell']].lt(0).any().any():
        raise ValueError('Invalid broker interval quantities')
    return x.sort_values('date')


def persistence(branches, frames, days, signal_day, window):
    if window not in (5, 20):
        raise ValueError('Unregistered window')
    i = days.get_loc(pd.Timestamp(signal_day))
    if len(branches) != 5 or i < window-1:
        return dict(known=False, reason='missing_branch_selection_or_warmup')
    window_days = days[i-window+1:i+1]
    positive_days, buy, sell, persistent = 0, 0., 0., 0
    evidence = []
    for branch in branches:
        broker = str(branch['securities_trader_id'])
        frame = frames[broker]
        if frame.empty:
            return dict(known=False, reason='empty_branch_interval', broker=broker)
        indexed = frame.set_index('date')
        signal = indexed.reindex([pd.Timestamp(signal_day)])
        if signal[['buy','sell']].isna().any().any():
            return dict(known=False, reason='missing_signal_day', broker=broker)
        if any(abs(float(signal.iloc[0][k])-branch[k]) > .001 for k in ('buy','sell')):
            return dict(known=False, reason='signal_day_revision_conflict', broker=broker)
        rows = indexed.reindex(window_days)
        if rows[['buy','sell']].isna().any().any():
            return dict(known=False, reason='missing_branch_market_day', broker=broker)
        n = int((rows.buy > rows.sell).sum())
        b, s = float(rows.buy.sum()), float(rows.sell.sum())
        passed = n >= math.ceil(.6*window) and b > s
        persistent += int(passed)
        positive_days += n; buy += b; sell += s
        evidence.append(dict(broker=broker, positive_days=n, buy=b, sell=s, persistent=passed))
    if buy+sell <= 0:
        return dict(known=False, reason='zero_turnover')
    return dict(known=True, score=(buy-sell)/(buy+sell)*positive_days/(5*window),
                passed=persistent >= 3, persistent_branches=persistent, branches=evidence,
                signal_date=str(pd.Timestamp(signal_day).date()), window=window)


from skills.five_axis_replay import FiveAxisReplay


class BrokerPersistenceReplay(FiveAxisReplay):
    def __init__(self, *args, broker_mode='control', broker_signals=None, **kwargs):
        if broker_mode not in ('control', 'rank'):
            raise ValueError('Unknown broker replay mode')
        super().__init__(*args, arm='capacity', **kwargs)
        self.broker_mode = broker_mode
        self.broker_signals = broker_signals or {}
        self.broker_decisions = []

    def corporate_day(self, day):
        income = super().corporate_day(day)
        if self.broker_mode == 'rank':
            def score(event):
                signal = self.broker_signals[event['event_id']]
                if not signal['known'] or pd.Timestamp(signal['signal_date']) >= day:
                    raise ValueError('Broker evidence unavailable before execution')
                return -signal['score']
            # Stable sort: equal broker scores retain the causal turnover order.
            self.events[day].sort(key=score)
            for rank, event in enumerate(self.events[day], 1):
                self.broker_decisions.append(dict(date=str(day.date()), rank=rank,
                    event_id=event['event_id'], score=self.broker_signals[event['event_id']]['score']))
        return income
