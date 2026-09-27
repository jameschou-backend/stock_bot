"""As-of early-strength features, sequential confirmations and separate labels."""
import numpy as np
import pandas as pd

from skills.first_bar import EXCLUDED as FIRST_EXCLUDED, END, _path

EXCLUDED = FIRST_EXCLUDED | {'6446'}
FILTERS = ('all', 'turn')
TIMINGS = ('first', 'confirm3', 'retest5')


def event_features(events, close):
    """The volume launch itself cannot create its own prior-strength signal."""
    c = close.where(np.isfinite(close) & close.gt(0))
    r5 = c / c.shift(5) - 1
    relative = r5.sub(r5['0050'], axis=0)
    complete = c.notna().rolling(11, min_periods=11).sum().eq(11)
    r20 = c / c.shift(20) - 1
    distance = c / c.rolling(60, min_periods=60).mean() - 1
    rows = []
    for e in events.to_dict('records'):
        pos = c.index.get_loc(pd.Timestamp(e['signal_date'])) - 1
        sid = e['stock_id']
        known = pos >= 10 and bool(complete.iloc[pos][sid] and complete.iloc[pos]['0050'])
        recent = float(relative.iloc[pos][sid]) if known else None
        prior = float(relative.iloc[pos-5][sid]) if known else None
        rows.append(dict(e, named_case=sid in EXCLUDED, strength_date=str(c.index[pos].date()) if pos >= 0 else None,
            strength_known=known, relative5=recent, previous_relative5=prior,
            turn=bool(recent > 0 and prior <= 0) if known else None,
            prior_return20=float(r20.iloc[pos][sid]) if pos >= 20 else None,
            prior_distance60=float(distance.iloc[pos][sid]) if pos >= 59 else None))
    result = pd.DataFrame(rows)
    result['turn'] = pd.array(result['turn'], dtype='boolean')
    for key in ('relative5', 'previous_relative5', 'prior_return20', 'prior_distance60'):
        result[key] = pd.to_numeric(result[key]).astype(float)
    return result


def _bar(close, raw, volume, ohlc, pos, sid):
    day = close.index[pos]
    values = {k: ohlc[k].at[day, sid] for k in ('open', 'high', 'low', 'close', 'volume')}
    price, adjusted, shares = raw.at[day, sid], close.at[day, sid], volume.at[day, sid]
    if not all(np.isfinite(x) and x > 0 for x in (*values.values(), price, adjusted, shares)):
        return None
    if (abs(values['close'] - price) > 1e-8 or values['volume'] != shares or
            values['high'] < max(values['open'], values['close']) or
            values['low'] > min(values['open'], values['close']) or values['low'] > values['high']):
        return None
    factor = adjusted / price
    return dict(close=adjusted, low=values['low'] * factor, open=values['open'] * factor, volume=shares)


def decisions(events, computed, close, raw, volume, ohlc, *, end=END):
    """Process observations in order; later success cannot revive a broken setup."""
    days = close.index
    rows = []
    for e in events.to_dict('records'):
        pos = days.get_loc(pd.Timestamp(e['signal_date'])); sid = e['stock_id']
        for selection in FILTERS:
            for timing in TIMINGS:
                row = dict(event_id=e['event_id'], stock_id=sid, launch_date=e['signal_date'],
                    named_case=e['named_case'], selection=selection, timing=timing,
                    arm=f'{selection}_{timing}', selected=True, state='pending',
                    decision_date=None, entry_signal_date=None, entry_date=None, wait_sessions=None)
                if selection == 'turn' and not e['strength_known']:
                    row.update(selected=None, state='unknown_strength', decision_date=e['signal_date'])
                elif selection == 'turn' and not e['turn']:
                    row.update(selected=False, state='filtered', decision_date=e['signal_date'])
                else:
                    signal = pos if timing == 'first' else None
                    count = 3 if timing == 'confirm3' else 5
                    observed_volume = []
                    if timing != 'first':
                        for j in range(pos+1, pos+count+1):
                            if j >= len(days) or days[j] > pd.Timestamp(end):
                                row['state'] = 'unmatured_confirmation'; break
                            date = str(days[j].date())
                            bar = _bar(close, raw, volume, ohlc, j, sid)
                            if bar is None:
                                row.update(state='unknown_confirmation_data', decision_date=date); break
                            if not computed['eligible'].at[days[j], sid]:
                                row.update(state='ineligible', decision_date=date); break
                            if bar['close'] < close.iloc[pos-1][sid]:
                                row.update(state='support_failed', decision_date=date); break
                            observed_volume.append(bar['volume'])
                            if timing == 'confirm3':
                                trigger = (j == pos+3 and bar['close'] >= close.iloc[pos][sid] and
                                    np.mean(observed_volume) < volume.iloc[pos][sid])
                            else:
                                trigger = (bar['low'] <= close.iloc[pos][sid] <= bar['close'] and
                                    bar['close'] >= bar['open'] and bar['volume'] < volume.iloc[pos][sid])
                            if trigger:
                                signal = j; break
                        else:
                            row.update(state='not_triggered', decision_date=str(days[pos+count].date()))
                    if signal is not None:
                        row.update(state='triggered', decision_date=str(days[signal].date()),
                            entry_signal_date=str(days[signal].date()), wait_sessions=signal-pos)
                        if signal+1 < len(days) and days[signal+1] <= pd.Timestamp(end):
                            row['entry_date'] = str(days[signal+1].date())
                        else:
                            row['state'] = 'unmatured_entry'
                rows.append(row)
    result = pd.DataFrame(rows)
    result['selected'] = pd.array(result['selected'], dtype='boolean')
    return result


def orders(decision_table, computed):
    result = {f'{f}_{t}': [] for f in FILTERS for t in TIMINGS}
    for e in decision_table.to_dict('records'):
        if e['named_case'] or e['state'] != 'triggered':
            continue
        sid = e['stock_id']; date = e['entry_signal_date']; day = pd.Timestamp(date)
        liq = dict(as_of=date, complete_20_sessions=True, observations=20,
            adv20_shares=float(computed['shares'].at[day, sid]),
            mean_turnover20_twd=float(computed['adv'].at[day, sid]))
        result[e['arm']].append(dict(event_id=e['event_id'], members=[sid], stock_id=sid,
            signal_date=date, entry_date=e['entry_date'], priority=liq['mean_turnover20_twd'],
            feature_cutoff_date=date, group_cutoff_date=date, membership_point_in_time=False,
            membership_snapshot_date='2026-09-27', liquidity_at_signal=liq,
            liquidity_before_entry=dict(liq), launch_date=e['launch_date']))
    return {k: sorted(v, key=lambda r: (r['signal_date'], -r['priority'], r['stock_id'])) for k, v in result.items()}


def outcomes(events, decision_table, close, quality):
    """Future labels do not feed event_features, decisions or orders."""
    days = close.index; positions = {str(d.date()): i for i, d in enumerate(days)}
    records = []
    # Compute common paths only once per launch/horizon, not once per arm.
    labels = {}
    for e in events.to_dict('records'):
        pos = positions[e['signal_date']]; sid = e['stock_id']
        for horizon in (20, 60):
            end = pos+horizon+1
            path = _path(close, quality, sid, pos+1, end)
            date = str(days[end].date()) if end < len(days) else None
            label = dict(horizon=horizon, exit_date=date,
                phase='discovery' if date and date <= '2024-12-31' else
                    'replication' if e['signal_date'] >= '2025-01-01' else 'boundary',
                year=int(e['signal_date'][:4]), baseline_return=None, benchmark_return=None, baseline_surge=None)
            if path is not None:
                a = path[0]; ret = float(a[-1, 0]/a[0, 0]-1); bm = float(a[-1, 1]/a[0, 1]-1)
                goal, excess = (.30, .20) if horizon == 20 else (.50, .30)
                label.update(baseline_return=ret, benchmark_return=bm, baseline_surge=ret >= goal and ret-bm >= excess)
            labels[(e['event_id'], horizon)] = label
    for e in decision_table.to_dict('records'):
        sid = e['stock_id']; pos = positions[e['launch_date']]
        for horizon in (20, 60):
            r = dict(e, **labels[(e['event_id'], horizon)], reference_return=None, matched_benchmark_return=None,
                entered_surge=None, opportunity_return=None, paired_difference=None, common_excess=None,
                entry_false_start5=None, mae=None, mfe=None)
            if e['state'] in ('filtered', 'ineligible', 'support_failed', 'not_triggered'):
                r['opportunity_return'] = 0.
            elif e['state'] == 'triggered':
                begin = positions[e['entry_date']]
                path = _path(close, quality, sid, begin, pos+horizon+1)
                if path is not None:
                    a = path[0]; ret = float(a[-1, 0]/a[0, 0]-1); bm = float(a[-1, 1]/a[0, 1]-1)
                    goal, excess = (.30, .20) if horizon == 20 else (.50, .30)
                    r.update(reference_return=ret, matched_benchmark_return=bm, opportunity_return=ret,
                        entered_surge=ret >= goal and ret-bm >= excess,
                        mae=float((a[:, 0]/a[0, 0]-1).min()), mfe=float((a[:, 0]/a[0, 0]-1).max()))
                # Five subsequent sessions from the actual entry, not from the launch.
                follow = _path(close, quality, sid, begin, begin+5)
                if follow is not None:
                    r['entry_false_start5'] = bool((follow[0][1:, 0] < close.iloc[pos-1][sid]).any())
            if r['baseline_return'] is not None and r['opportunity_return'] is not None:
                r['paired_difference'] = r['opportunity_return'] - r['baseline_return']
                r['common_excess'] = r['opportunity_return'] - r['benchmark_return']
            records.append(r)
    return pd.DataFrame(records)


def summarize(table, *, annual=False):
    rows = []
    groups = ['year' if annual else 'phase', 'horizon', 'arm']
    for key, part in table[~table.named_case].groupby(groups, sort=True):
        entered = part[part.state.eq('triggered')]
        known = entered.dropna(subset=['reference_return'])
        pairs = part.dropna(subset=['paired_difference'])
        selected_pairs = pairs[pairs.selected.eq(True)]
        winners = part[part.baseline_surge.eq(True)]
        unknown = ~part.state.isin(['filtered', 'ineligible', 'support_failed', 'not_triggered', 'triggered'])
        winner_entered = winners.state.eq('triggered')
        winner_unknown = ~winners.state.isin(['filtered', 'ineligible', 'support_failed', 'not_triggered', 'triggered'])
        rows.append(dict(zip(groups, key), events=len(part), stocks=part.stock_id.nunique(),
            selected=int(part.selected.eq(True).sum()), triggered=len(entered), decision_unknown=int(unknown.sum()),
            entered_known=len(known), entered_unknown=len(entered)-len(known),
            entered_mean=known.reference_return.mean(), entered_median=known.reference_return.median(),
            entered_excess=(known.reference_return-known.matched_benchmark_return).mean(),
            entered_surge_rate=known.entered_surge.astype(float).mean(),
            baseline_known=int(part.baseline_return.notna().sum()), baseline_surges=len(winners),
            captured_baseline_surges=int(winner_entered.sum()), missed_baseline_surges=int((~winner_entered & ~winner_unknown).sum()),
            unresolved_baseline_surges=int(winner_unknown.sum()),
            captured_surge_fraction=float(winner_entered.mean()) if len(winners) else None,
            opportunity_paired_known=len(pairs), opportunity_mean=pairs.opportunity_return.mean(),
            paired_baseline_mean=pairs.baseline_return.mean(), opportunity_difference=pairs.paired_difference.mean(),
            common_excess=pairs.common_excess.mean(), selected_paired_known=len(selected_pairs),
            selected_opportunity_difference=selected_pairs.paired_difference.mean(),
            median_wait=entered.wait_sessions.median(),
            entry_false_known=int(entered.entry_false_start5.notna().sum()),
            entry_false_start5=entered.entry_false_start5.dropna().astype(float).mean(),
            mean_mae=known.mae.mean(), mean_mfe=known.mfe.mean()))
    return pd.DataFrame(rows)
