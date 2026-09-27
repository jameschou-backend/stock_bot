"""Causal post-launch warnings; future labels live in a separate evaluator."""
import numpy as np
import pandas as pd

from skills.early_strength import EXCLUDED
from skills.first_bar import _path
from skills.million_replay import COMMISSION, SLIPPAGE

ARMS = ('hold', 'origin', 'dry_weak', 'heavy_red', 'rotation_weak', 'combined')
WARNINGS = ('dry_weak', 'heavy_red', 'rotation_weak')


def peer_features(close, raw, volume, companies, wanted):
    """Vectorized leave-one-out industry context with dated coverage checks."""
    if companies.stock_id.duplicated().any() or companies.listed_date.isna().any():
        raise ValueError('Unique company identities and listing dates required')
    for f in (raw, volume):
        if not f.index.equals(close.index) or not f.columns.equals(close.columns):
            raise ValueError('Price/volume axes differ')
    cohort = companies[companies.stock_id.isin(close.columns) & companies.stock_id.str.fullmatch(r'\d{4}')
        & companies.stock_id.ne('0050')].set_index('stock_id')
    ids = cohort.index; days = close.index
    active = pd.DataFrame(days.to_numpy()[:, None] >= pd.to_datetime(cohort.listed_date).to_numpy()[None, :], index=days, columns=ids)
    c = close[ids].where(active & np.isfinite(close[ids]) & close[ids].gt(0))
    amount = (raw[ids]*volume[ids]).where(active & np.isfinite(raw[ids]) & raw[ids].gt(0)
        & np.isfinite(volume[ids]) & volume[ids].gt(0))
    r5 = (c/c.shift(5)-1).where(c.notna().rolling(6, min_periods=6).sum().eq(6))
    ma = c.rolling(20, min_periods=20).mean()
    above = c.ge(ma).astype(float).where(c.notna() & ma.notna())
    total = amount.sum(axis=1, min_count=1)
    market_ok = (amount.notna().sum(axis=1)/active.sum(axis=1).replace(0, np.nan)).rolling(25, min_periods=25).min().ge(.9)
    total5 = total.rolling(5, min_periods=5).sum()
    total20 = total.shift(5).rolling(20, min_periods=20).sum()
    results = {}
    for _, group in cohort.groupby('industry', sort=True):
        members = group.index; targets = [sid for sid in members if sid in wanted]
        if not targets: continue
        amounts = amount[members]; group_total = amounts.sum(axis=1, min_count=1)
        expected = active[members].sum(axis=1); observed = amounts.notna().sum(axis=1)
        price_count = r5[members].notna().sum(axis=1)
        above_count = above[members].notna().sum(axis=1); above_sum = above[members].sum(axis=1, min_count=1)
        # Removing one's rank from sorted rows gives the exact median of peers,
        # including ties and missing values, without a per-event groupby.
        values = r5[members].to_numpy(float)
        sorted_values = np.sort(values, axis=1)
        ranks = np.argsort(np.argsort(values, axis=1, kind='stable'), axis=1, kind='stable')
        row_indices = np.arange(len(days)); counts = np.isfinite(values).sum(axis=1)
        for sid in targets:
            col = members.get_loc(sid); own_known = np.isfinite(values[:, col])
            n = counts-own_known; safe_n = np.maximum(n, 1)
            lo = (safe_n-1)//2; hi = safe_n//2
            lo += (own_known & (ranks[:, col] <= lo)); hi += (own_known & (ranks[:, col] <= hi))
            median = (sorted_values[row_indices, np.minimum(lo, len(members)-1)]
                + sorted_values[row_indices, np.minimum(hi, len(members)-1)])/2
            median[n == 0] = np.nan
            peer_n = expected-active[sid].astype(int)
            peer_observed = observed-amounts[sid].notna().astype(int)
            peer_sum = group_total-amounts[sid].fillna(0)
            peer_sum = peer_sum.where(peer_observed.gt(0))
            coverage = peer_observed/peer_n.replace(0, np.nan)
            recent = peer_sum.rolling(5, min_periods=5).sum()
            prior = peer_sum.shift(5).rolling(20, min_periods=20).sum()
            share5 = recent/total5; share20 = prior/total20
            pc = price_count-r5[sid].notna().astype(int)
            ac = above_count-above[sid].notna().astype(int)
            breadth = (above_sum-above[sid].fillna(0))/ac.replace(0, np.nan)
            known = (market_ok & coverage.rolling(25, min_periods=25).min().ge(.9)
                & peer_n.rolling(25, min_periods=25).min().ge(5) & (pc/peer_n).ge(.9)
                & (ac/peer_n).ge(.9) & prior.gt(0) & total5.gt(0) & total20.gt(0) & r5[sid].notna())
            results[sid] = pd.DataFrame(dict(peer_count=peer_n, coverage25=coverage.rolling(25, min_periods=25).min(),
                amount_ratio=(recent/5)/(prior/20), share_change=share5-share20, above20=breadth,
                peer_return5=median, stock_return5=r5[sid], known=known), index=days)
    return results


def observations(events, close, raw, volume, ohlc, companies):
    """Only dates T+1..T+10 are eligible for a new early-failure warning."""
    peers = peer_features(close, raw, volume, companies, set(events.stock_id))
    ids = close.columns; days = close.index
    q = {k: v.reindex(index=days, columns=ids).to_numpy(float) for k, v in ohlc.items()}
    c, p, v = (f.to_numpy(float) for f in (close, raw, volume))
    valid = np.isfinite(c) & (c > 0) & np.isfinite(p) & (p > 0) & np.isfinite(v) & (v > 0)
    for a in q.values(): valid &= np.isfinite(a) & (a > 0)
    valid &= ((np.abs(q['close']-p) <= 1e-8) & (q['volume'] == v)
        & (q['high'] >= np.maximum(q['open'], q['close'])) & (q['low'] <= np.minimum(q['open'], q['close'])))
    with np.errstate(divide='ignore', invalid='ignore'):
        adjusted_low = q['low']*(c/p)
        adjusted_open = q['open']*(c/p)
    rows = []
    for e in events.to_dict('records'):
        pos = days.get_loc(pd.Timestamp(e['signal_date'])); sid = e['stock_id']; col = ids.get_loc(sid)
        normal = v[pos-20:pos, col]
        normal_mean = float(normal.mean()) if len(normal) == 20 and np.isfinite(normal).all() and (normal > 0).all() else np.nan
        midpoint = (adjusted_open[pos, col]+c[pos, col])/2
        for j in range(pos+1, min(pos+11, len(days))):
            dry = heavy = None
            if j == pos+1: dry = False
            elif valid[j-2:j+1, col].all() and np.isfinite(normal_mean):
                dry = bool(v[j, col] < .7*normal_mean and c[j, col] < c[j-1, col] < c[j-2, col]
                    and adjusted_low[j, col] < adjusted_low[j-1, col] < adjusted_low[j-2, col])
            if valid[j, col] and np.isfinite(normal_mean):
                black = q['close'][j, col] < q['open'][j, col]
                span = q['high'][j, col]-q['low'][j, col]
                location = (q['close'][j, col]-q['low'][j, col])/span if span > 0 else 1.
                heavy = bool(black and v[j, col] >= 1.5*normal_mean and location <= .35 and c[j, col] < midpoint)
            context = peers[sid].iloc[j].to_dict() if sid in peers else dict(known=False)
            rotation = bool(context['amount_ratio'] < .8 and context['share_change'] < 0 and context['above20'] < .4
                and context['stock_return5'] < context['peer_return5']) if context['known'] else None
            rows.append(dict(event_id=e['event_id'], stock_id=sid, launch_date=e['signal_date'],
                date=str(days[j].date()), offset=j-pos, normal_volume=normal_mean,
                volume_ratio_normal=v[j, col]/normal_mean, close=c[j, col], origin=c[pos-1, col],
                dry_weak=dry, heavy_red=heavy, rotation_weak=rotation, **{'peer_'+k: val for k,val in context.items()}))
    result = pd.DataFrame(rows)
    for k in WARNINGS: result[k] = pd.array(result[k], dtype='boolean')
    # Stable types even if a truncated sample contains no missing observations.
    for k in ('normal_volume', 'volume_ratio_normal', 'close', 'origin', 'peer_peer_count', 'peer_coverage25',
              'peer_amount_ratio', 'peer_share_change', 'peer_above20', 'peer_peer_return5', 'peer_stock_return5'):
        if k not in result: result[k] = np.nan
        result[k] = pd.to_numeric(result[k]).astype(float)
    return result


def _nullable_or(values):
    return True if any(v is not None and bool(v) for v in values) else None if any(v is None for v in values) else False


def decisions(events, obs, close):
    rows = []; days = close.index; c = close.to_numpy(float)
    lookups = {k: {int(r['offset']): r for r in f.to_dict('records')} for k, f in obs.groupby('event_id', sort=False)}
    for e in events.to_dict('records'):
        pos = days.get_loc(pd.Timestamp(e['signal_date'])); col = close.columns.get_loc(e['stock_id'])
        for horizon in (20, 60):
            endpoint = pos+horizon+1
            for arm in ARMS:
                row = dict(event_id=e['event_id'], stock_id=e['stock_id'], launch_date=e['signal_date'],
                    named_case=e['stock_id'] in EXCLUDED, horizon=horizon, arm=arm, state='holding',
                    signal_date=None, signal_offset=None, reason=None, exit_date=None, delayed_exit_date=None,
                    endpoint=str(days[endpoint].date()) if endpoint < len(days) else None)
                if arm != 'hold':
                    for offset in range(1, horizon+1):
                        j = pos+offset
                        if j >= len(days): row['state'] = 'unmatured'; break
                        price = c[j, col]
                        reason = None; unknown = False
                        if not np.isfinite(price) or price <= 0:
                            unknown = True
                        elif price < c[pos-1, col]: reason = 'origin'
                        elif arm not in ('hold', 'origin') and offset <= 10:
                            observation = lookups[e['event_id']][offset]
                            keys = WARNINGS if arm == 'combined' else (arm,)
                            values = [None if pd.isna(observation[k]) else bool(observation[k]) for k in keys]
                            trigger = _nullable_or(values)
                            unknown = trigger is None
                            if trigger: reason = next(k for k, val in zip(keys, values) if val is True)
                        if reason or unknown:
                            row.update(state='unknown' if unknown else 'triggered', signal_date=str(days[j].date()),
                                signal_offset=offset, reason=reason)
                            if reason:
                                if j+1 < len(days): row['exit_date'] = str(days[j+1].date())
                                if j+2 < len(days): row['delayed_exit_date'] = str(days[j+2].date())
                            break
                if row['state'] == 'holding' and endpoint >= len(days): row['state'] = 'unmatured'
                rows.append(row)
    result = pd.DataFrame(rows); result['signal_offset'] = pd.to_numeric(result.signal_offset).astype(float)
    return result


def fee_diagnostic(ret, *, sid, slip_multiplier=1):
    """Proportional sensitivity only, not rounded broker costs or portfolio P&L."""
    slip = SLIPPAGE*slip_multiplier
    tax = .001 if sid == '0050' else .003
    return (1+ret)*(1-COMMISSION-tax-slip)/(1+COMMISSION+slip)-1


def outcomes(events, decision_table, close, quality):
    rows = []; days = close.index; dates = {str(d.date()): i for i,d in enumerate(days)}
    choices = {(e['event_id'], e['horizon'], e['arm']): e for e in decision_table.to_dict('records')}
    for e in events.to_dict('records'):
        pos = dates[e['signal_date']]; sid = e['stock_id']
        for horizon in (20, 60):
            end = pos+horizon+1; path = _path(close, quality, sid, pos+1, end)
            first_failure = None; failure10 = None
            if path is not None:
                hits = np.flatnonzero(path[0][:10, 0] < close.iloc[pos-1][sid])
                failure10 = bool(len(hits)); first_failure = pos+1+int(hits[0]) if len(hits) else None
            def evaluate(choice, extra=0):
                if path is None or choice['state'] in ('unknown', 'unmatured'): return None, None
                selected = choice['delayed_exit_date'] if extra else choice['exit_date']
                exit_pos = min(dates[selected], end) if selected is not None else end
                if choice['state'] == 'triggered' and selected is None and end < len(days)-extra:
                    return None, None
                return float(path[0][exit_pos-pos-1, 0]/path[0][0, 0]-1), exit_pos
            origin, _ = evaluate(choices[(e['event_id'], horizon, 'origin')])
            origin_stress, _ = evaluate(choices[(e['event_id'], horizon, 'origin')], 1)
            for arm in ARMS:
                decision = choices[(e['event_id'], horizon, arm)]
                r = dict(decision, phase='discovery' if decision['endpoint'] and decision['endpoint'] <= '2024-12-31' else
                    'replication' if e['signal_date'] >= '2025-01-01' else 'boundary', year=int(e['signal_date'][:4]),
                    hold_return=None, benchmark_return=None, exit_return=None, origin_return=origin,
                    delta_hold=None, delta_origin=None, fee_return=None, fee_excess=None, fee_delta_origin=None,
                    stress_fee_return=None, stress_delta_origin=None, hold_surge=None, retained_surge=None,
                    failure10=failure10, first_failure_date=str(days[first_failure].date()) if first_failure is not None else None,
                    warning_before_failure=None, exit_before_failure=None, lead_sessions=None,
                    early_exit=False, cut_surge=None, lost_return=None)
                if path is not None:
                    hold = float(path[0][-1, 0]/path[0][0, 0]-1); bm = float(path[0][-1, 1]/path[0][0, 1]-1)
                    goal, excess = (.30, .20) if horizon == 20 else (.50, .30)
                    r.update(hold_return=hold, benchmark_return=bm, hold_surge=hold >= goal and hold-bm >= excess)
                    ret, exit_pos = evaluate(decision); stressed, _ = evaluate(decision, 1)
                    if ret is not None:
                        fee = fee_diagnostic(ret, sid=sid); early = exit_pos < end
                        warning = decision['reason'] in WARNINGS
                        r.update(exit_return=ret, delta_hold=ret-hold, delta_origin=ret-origin if origin is not None else None,
                            fee_return=fee, fee_excess=fee-fee_diagnostic(bm, sid='0050'),
                            fee_delta_origin=fee-fee_diagnostic(origin, sid=sid) if origin is not None else None,
                            retained_surge=ret >= goal and ret-bm >= excess, early_exit=early,
                            cut_surge=bool(r['hold_surge'] and early and ret < hold), lost_return=hold-ret if early else 0.)
                        if first_failure is not None:
                            signal = dates[decision['signal_date']] if decision['signal_date'] else end
                            r['warning_before_failure'] = bool(warning and signal < first_failure)
                            r['exit_before_failure'] = bool(warning and exit_pos < first_failure)
                            r['lead_sessions'] = first_failure-exit_pos if r['exit_before_failure'] else None
                    if stressed is not None:
                        r['stress_fee_return'] = fee_diagnostic(stressed, sid=sid, slip_multiplier=2)
                        if origin_stress is not None:
                            r['stress_delta_origin'] = r['stress_fee_return']-fee_diagnostic(origin_stress, sid=sid, slip_multiplier=2)
                rows.append(r)
    return pd.DataFrame(rows)


def summarize(table, *, annual=False):
    rows = []; groups = ['year' if annual else 'phase', 'horizon', 'arm']
    for keys, part in table[~table.named_case].groupby(groups, sort=True):
        paired = part.dropna(subset=['delta_origin']); known = part.dropna(subset=['exit_return'])
        failed = part[part.failure10.eq(True)]; failed_known = failed.dropna(subset=['exit_return'])
        winners = part[part.hold_surge.eq(True)]; win_known = winners.dropna(subset=['exit_return'])
        rows.append(dict(zip(groups, keys), events=len(part), result_known=len(known), result_unknown=len(part)-len(known),
            paired_origin_known=len(paired), mean_return=known.exit_return.mean(), median_return=known.exit_return.median(),
            delta_hold=known.delta_hold.mean(), delta_origin=paired.delta_origin.mean(),
            fee_return=known.fee_return.mean(), fee_excess=known.fee_excess.mean(),
            fee_delta_origin=paired.fee_delta_origin.mean(), stress_delta_origin=paired.stress_delta_origin.mean(),
            early_exits=int(known.early_exit.sum()), origin10_events=len(failed), origin10_known=len(failed_known),
            warnings_before=int(failed_known.warning_before_failure.eq(True).sum()),
            exits_before=int(failed_known.exit_before_failure.eq(True).sum()), median_lead=failed_known.lead_sessions.median(),
            exits_without_failure10=int((known.early_exit & known.failure10.eq(False)).sum()),
            hold_surges=len(winners), hold_surges_known=len(win_known), cut_surges=int(win_known.cut_surge.eq(True).sum()),
            retained_surges=int(win_known.retained_surge.eq(True).sum()), winner_mean_lost=win_known.lost_return.mean()))
    return pd.DataFrame(rows)
