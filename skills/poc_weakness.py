"""Causal close warnings on a sealed holding path; never simulate a trade."""
from collections import Counter

import numpy as np
import pandas as pd

RULES = {
    'two_black': '仍有價格獲利＋兩黑K',
    'below_ma20_two': '仍有價格獲利＋連兩日低於MA20',
    'two_black_below_ma20': '仍有價格獲利＋兩黑K且低於MA20',
}


def below(left, right):
    tolerance = np.maximum(1e-12, np.maximum(abs(left), abs(right))*1e-12)
    return left.lt(right-tolerance)


def indicators(close, quotes):
    if (not close.index.is_unique or not close.index.is_monotonic_increasing
            or not close.columns.is_unique or quotes.duplicated(['date', 'stock_id']).any()):
        raise ValueError('Warning inputs require unique ordered observations')
    raw = {k: quotes.pivot(index='date', columns='stock_id', values=k).reindex(
        index=close.index, columns=close.columns) for k in ('open', 'high', 'low', 'close', 'volume')}
    valid = close.gt(0) & np.isfinite(close)
    for frame in raw.values():
        valid &= frame.gt(0) & np.isfinite(frame)
    bad = valid & (raw['open'].lt(raw['low']) | raw['open'].gt(raw['high'])
                   | raw['close'].lt(raw['low']) | raw['close'].gt(raw['high']))
    if bad.any().any():
        raise ValueError('Warning input contains invalid OHLC')
    black = valid & raw['close'].lt(raw['open']) & below(close, close.shift(1))
    two = black & black.shift(1, fill_value=False)
    ma20 = close.where(close.gt(0) & np.isfinite(close)).rolling(20, min_periods=20).mean()
    under = valid & below(close, ma20)
    return dict(two_black=two, below_ma20_two=under & under.shift(1, fill_value=False),
                two_black_below_ma20=two & under), ma20


def diagnose(account, close, quotes, episodes):
    signals, ma20 = indicators(close, quotes)
    days = [str(d.date()) for d in close.index]
    pos = {d: i for i, d in enumerate(days)}
    end = account['daily'][-1]['date']
    nav = {r['date']: r['nav'] for r in account['daily']}
    holdings = {(h['event_id'], h['date']): h for h in account['holdings'] if h['qty'] > 0}
    if len(holdings) != sum(h['qty'] > 0 for h in account['holdings']):
        raise ValueError('Duplicate daily cohort holding')
    cohorts = []
    for cohort in account['cohorts']:
        eid, sid, entry = (cohort[k] for k in ('event_id', 'stock_id', 'entry_date'))
        sells = sorted((r for r in account['orders'] if r['event_id'] == eid and r['side'] == 'sell'),
                       key=lambda r: (r['date'], r['signal_date']))
        decision = sells[0]['signal_date'] if sells else None
        target = sells[0]['date'] if sells else None
        if decision and (decision not in pos or target != days[pos[decision]+1]):
            raise ValueError('Original first exit order must follow its decision by one market session')
        first_fill = next((r['date'] for r in account['trades']
                           if r['event_id'] == eid and r['side'] == 'sell'), None)
        anchor = float(close.at[pd.Timestamp(entry), sid])
        if not np.isfinite(anchor) or anchor <= 0:
            raise ValueError('Missing entry close anchor')
        cutoff = decision or end
        closes = close[sid]
        observations, warnings = [], {}
        for i in range(pos[entry]+1, pos[cutoff]):
            day = days[i]
            h = holdings.get((eid, day))
            if not h:
                continue
            if h.get('stale') or h.get('mark_date', day) != day:
                raise ValueError('Stale held observation cannot be a warning')
            current = float(closes.iloc[i])
            if not np.isfinite(current) or current <= 0:
                raise ValueError('Missing held close')
            profit = current > anchor + max(abs(anchor), abs(current), 1)*1e-12
            active = [key for key in RULES if profit and bool(signals[key][sid].iloc[i])]
            observations.append(dict(date=day, rules=active))
            for key in active:
                if key in warnings:
                    continue
                known_peak = float(closes.iloc[pos[entry]:i+1].max())
                # Future columns are descriptive outcomes only, never signal inputs.
                future = closes.iloc[i+1:pos[cutoff]+1]
                if not future.notna().all() or not future.gt(0).all():
                    raise ValueError('Missing post-warning outcome; cannot call it no rebound')
                later_max = float(future.max()) if len(future) else current
                warnings[key] = dict(date=day, next_session=days[i+1],
                    lead_sessions=pos[decision]-i if decision else None,
                    close_return_from_entry=current/anchor-1, known_peak_drawdown=current/known_peak-1,
                    weight=h['market_value']/nav[day], adjusted_close=current,
                    ma20=float(ma20[sid].iloc[i]) if pd.notna(ma20[sid].iloc[i]) else None,
                    outcome_end=cutoff, outcome_censored=decision is None,
                    subsequent_max_return=later_max/current-1,
                    later_new_holding_high=later_max > known_peak+max(abs(known_peak), abs(later_max), 1)*1e-12,
                    return_to_original_decision=float(closes.iloc[pos[cutoff]])/current-1,
                    signal_count=sum(key in o['rules'] for o in observations))
        for key, warning in warnings.items():
            warning['signal_count'] = sum(key in o['rules'] for o in observations)
        cohorts.append(dict(event_id=eid, stock_id=sid, name=cohort['name'], entry_date=entry,
            original_reason=sells[0]['reason'] if sells else None,
            original_decision=decision, original_target=target, first_original_fill=first_fill,
            warnings=warnings, observed_held_closes=len(observations)))
    summaries = {}
    for key in RULES:
        fired = [c for c in cohorts if key in c['warnings']]
        resolved = [c for c in fired if c['original_decision']]
        w = [c['warnings'][key] for c in resolved]
        leads = [r['lead_sessions'] for r in w]
        summaries[key] = dict(cohorts=len(cohorts), triggered=len(fired), resolved=len(resolved),
            censored=len(fired)-len(resolved),
            exit_reasons=dict(Counter(c['original_reason'] for c in resolved)),
            lead_median=float(np.median(leads)) if leads else None,
            lead_min=min(leads) if leads else None, lead_max=max(leads) if leads else None,
            subsequent_higher_close=sum(r['subsequent_max_return'] > 1e-12 for r in w),
            subsequent_new_high=sum(r['later_new_holding_high'] for r in w),
            original_decision_higher=sum(r['return_to_original_decision'] > 1e-12 for r in w),
            original_decision_lower=sum(r['return_to_original_decision'] < -1e-12 for r in w))
    by_id = {c['event_id']: c for c in cohorts}
    details = []
    for e in episodes:
        rows = []
        for c in e['cohorts']:
            if c['gross_pnl'] >= 0:
                continue
            original = by_id[c['event_id']]
            warning_rows = {}
            for key, w in original['warnings'].items():
                location = ('before_peak' if w['date'] < e['peak_date'] else
                            'during_decline' if w['date'] <= e['trough_date'] else 'after_trough')
                warning_rows[key] = dict(w, location=location)
            rows.append(dict(original, warnings=warning_rows, group=c['group'],
                peak_weight=c['peak_weight'], interval_gross_pnl=c['gross_pnl'],
                exit_during_decline=bool(original['original_target'] and
                    e['peak_date'] < original['original_target'] <= e['trough_date'])))
        details.append(dict(rank=e['rank'], peak_date=e['peak_date'], trough_date=e['trough_date'],
            drawdown=e['drawdown'], recovery_date=e['recovery_date'], rows=rows))
    return dict(cohorts=cohorts, rule_summaries=summaries, episodes=details,
                total_cohorts=len(cohorts), original_exit_counts=dict(Counter(
                    c['original_reason'] or 'open' for c in cohorts)))
