"""Fixed institutional-selling / price-resilience hypothesis, with unknowns."""
from copy import deepcopy
import numpy as np
import pandas as pd

from skills.early_strength import EXCLUDED
from skills.surge_anatomy import features as eligibility
from skills.launch_warning import fee_diagnostic

ARMS = ('all_known', 'buy_pressure', 'sell_pressure', 'resilient', 'absorption', 'sell_weak')


def candidates(close, raw, volume, companies, normalized):
    """All fixed anchors and eligible stocks; never accepts a future label."""
    if normalized.duplicated(['date','stock_id']).any(): raise ValueError('Duplicate institutional day')
    f = eligibility(close, raw, volume, companies); days = close.index; ids = f['eligible'].columns
    net = {k:normalized.pivot(index='date', columns='stock_id', values=k).reindex(index=days, columns=ids)
           for k in ('foreign','trust','dealer')}
    rows = []; v = volume[ids].where(np.isfinite(volume[ids]) & volume[ids].gt(0))
    company_rows = f['companies'].to_dict('index')
    for pos in np.flatnonzero((days >= '2022-01-03') & (days <= '2026-09-09'))[::21]:
        eligible = ids[f['eligible'].iloc[pos]]
        adv = f['adv20'].iloc[pos].to_dict()
        momentum = f['numeric']['momentum20'].iloc[pos].to_dict()
        relative = f['numeric']['relative20'].iloc[pos].to_dict()
        adv_shares = v.iloc[pos-19:pos+1].mean().to_dict()
        for lag in (1, 3):
            s = pos-lag; start = s-5
            prices = close.iloc[start:s+1][[*eligible, '0050']]
            valid = (np.isfinite(prices) & prices.gt(0)).all()
            r5 = prices.iloc[-1]/prices.iloc[0]-1
            floor = prices.iloc[1:].min()/prices.iloc[0]-1
            sums = {k:frame.iloc[s-4:s+1][eligible].sum(min_count=5) for k,frame in net.items()}
            shares = v.iloc[s-4:s+1][eligible].sum(min_count=5)
            ratio = (sums['foreign']+sums['trust'])/shares
            for sid in eligible:
                price_known = bool(valid[sid] and valid['0050'])
                known = price_known and np.isfinite(ratio[sid])
                holds = bool(r5[sid] >= 0 and r5[sid] >= r5['0050'] and floor[sid] >= -.02) if price_known else None
                sell = bool(ratio[sid] <= -.03) if known else None
                buy = bool(ratio[sid] >= .03) if known else None
                row = dict(event_id=f'abs-{days[pos].date()}-{sid}', stock_id=sid,
                    name=company_rows[sid]['name'], industry=str(company_rows[sid]['industry']),
                    signal_date=str(days[pos].date()), flow_lag=lag, feature_date=str(days[s].date()),
                    window_start=str(days[s-4].date()), price_base_date=str(days[start].date()),
                    named_case=sid in EXCLUDED, adv20=float(adv[sid]),
                    adv20_shares=float(adv_shares[sid]),
                    momentum20=float(momentum[sid]), relative20=float(relative[sid]),
                    flow_ratio=float(ratio[sid]), foreign5=float(sums['foreign'][sid]),
                    trust5=float(sums['trust'][sid]), dealer5=float(sums['dealer'][sid]),
                    shares5=float(shares[sid]), return5=float(r5[sid]) if price_known else np.nan,
                    excess5=float(r5[sid]-r5['0050']) if price_known else np.nan,
                    min_close_return5=float(floor[sid]) if price_known else np.nan,
                    feature_known=known, price_holds=holds)
                states = (True, buy, sell, holds, bool(sell and holds), bool(sell and not holds))
                row.update({arm:float(value) if known else np.nan for arm,value in zip(ARMS,states)})
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame['price_holds'] = pd.array(frame.price_holds, dtype='boolean')
    return frame


def orders(table, days):
    """Next-session orders based only on candidate table and dated liquidity."""
    result = {}; positions = {str(d.date()):i for i,d in enumerate(days)}
    base = table[table.flow_lag.eq(1) & ~table.named_case]
    for arm in ARMS:
        selected = []
        for r in base[base[arm].eq(1)].to_dict('records'):
            pos = positions[r['signal_date']]
            if pos+1 >= len(days) or days[pos+1] > pd.Timestamp('2026-09-09'): continue
            liquid = dict(as_of=r['signal_date'], complete_20_sessions=True, observations=20,
                adv20_shares=r['adv20_shares'], mean_turnover20_twd=r['adv20'])
            selected.append(dict(event_id=r['event_id'], stock_id=r['stock_id'], members=[r['stock_id']],
                signal_date=r['signal_date'], entry_date=str(days[pos+1].date()), priority=r['adv20'],
                feature_cutoff_date=r['signal_date'], group_cutoff_date=r['signal_date'],
                institutional_feature_date=r['feature_date'], membership_point_in_time=False,
                membership_snapshot_date='2026-09-27', liquidity_at_signal=liquid,
                liquidity_before_entry=deepcopy(liquid), absorption_ratio=r['flow_ratio']))
        result[arm] = sorted(selected, key=lambda r:(r['signal_date'], -r['priority'], r['stock_id']))
    return result


def outcomes(table, close, quality):
    """Attach independent future labels after the candidate/order tables exist."""
    if not quality.index.equals(close.index) or not quality.columns.equals(close.columns):
        raise ValueError('Quality price axes differ')
    rows = []; days = close.index
    for date, group in table.drop_duplicates('event_id').groupby('signal_date', sort=True):
        p = days.get_loc(pd.Timestamp(date)); ids = group.stock_id.tolist()
        for horizon in (20, 60):
            end = p+horizon+1; matured = end < len(days)
            valid = pd.Series(False,index=ids); ret = alt = pd.Series(np.nan,index=[*ids,'0050'])
            reasons = pd.Series('unmatured',index=ids)
            if matured:
                a,b = [f.iloc[p+1:end+1][[*ids,'0050']] for f in (close,quality)]
                ret,alt = a.iloc[-1]/a.iloc[0]-1,b.iloc[-1]/b.iloc[0]-1
                missing = (~(np.isfinite(a)&a.gt(0))).any() | (~(np.isfinite(b)&b.gt(0))).any()
                jump = (a/a.shift(1)-1).abs().gt(.15).any() | (b/b.shift(1)-1).abs().gt(.15).any()
                disputed = (ret-alt).abs().gt(.05)
                ok = ~(missing|jump|disputed)
                valid = ok[ids] & bool(ok['0050'])
                reasons[:] = 'known'
                reasons.loc[disputed[ids]] = 'adjustment_disagreement'
                reasons.loc[jump[ids]] = 'price_jump'
                reasons.loc[missing[ids]] = 'price_missing'
                if not ok['0050']: reasons[:] = 'benchmark_unresolved'
            for sid in ids:
                known = bool(valid[sid]); r,bm = (float(ret[sid]),float(ret['0050'])) if known else (np.nan,np.nan)
                goal,excess = (.30,.20) if horizon==20 else (.50,.30)
                endpoint = str(days[end].date()) if matured else None
                rows.append(dict(event_id=f'abs-{date}-{sid}', horizon=horizon, endpoint=endpoint,
                    entry_date=str(days[p+1].date()) if p+1<len(days) else None,
                    phase='discovery' if endpoint and endpoint <= '2024-12-31' else
                        'replication' if date >= '2025-01-01' else 'boundary',
                    year=int(date[:4]), label_reason=reasons[sid], forward_return=r, benchmark_return=bm,
                    excess_return=r-bm, surge=(r>=goal and r-bm>=excess) if known else None,
                    fee_return=fee_diagnostic(r,sid=sid),
                    fee_excess=fee_diagnostic(r,sid=sid)-fee_diagnostic(bm,sid='0050'),
                    stress_fee_return=fee_diagnostic(r,sid=sid,slip_multiplier=2),
                    stress_fee_excess=fee_diagnostic(r,sid=sid,slip_multiplier=2)-fee_diagnostic(bm,sid='0050',slip_multiplier=2)))
    return table.merge(pd.DataFrame(rows), on='event_id', validate='many_to_many')


def statistics(table, *, annual=False):
    rows = []; groups = ['flow_lag','year' if annual else 'phase','horizon']
    for keys, group in table[~table.named_case].groupby(groups, sort=True):
        for arm in ARMS:
            part = group[group[arm].eq(1)]; known = part.dropna(subset=['forward_return'])
            common_hits = int((group.feature_known & group.surge.eq(True)).sum())
            hits = int(known.surge.eq(True).sum())
            rows.append(dict(zip(groups,keys), arm=arm, eligible=len(group), feature_unknown=int((~group.feature_known).sum()),
                signals=len(part), known=len(known), outcome_unknown=len(part)-len(known), stocks=part.stock_id.nunique(),
                mean_return=known.forward_return.mean(), median_return=known.forward_return.median(),
                mean_excess=known.excess_return.mean(), fee_return=known.fee_return.mean(), fee_excess=known.fee_excess.mean(),
                stress_fee_return=known.stress_fee_return.mean(), stress_fee_excess=known.stress_fee_excess.mean(),
                surge_hits=hits, surge_rate=hits/len(known) if len(known) else np.nan,
                common_surges=common_hits, capture_rate=hits/common_hits if common_hits else np.nan,
                eligible_surges=int(group.surge.eq(True).sum())))
    return pd.DataFrame(rows)


def matched_pairs(table):
    """Match BEFORE outcomes: fixed date, industry, pressure band, past features."""
    main = table[~table.named_case & table.feature_known].copy(); pairs=[]
    main['pressure_band'] = np.where(main.flow_ratio <= -.10,0,np.where(main.flow_ratio <= -.05,1,2))
    for (lag,date,industry,band), part in main[main.sell_pressure.eq(1)].groupby(
            ['flow_lag','signal_date','industry','pressure_band'],sort=True):
        controls = part[part.sell_weak.eq(1)].sort_values('stock_id')
        scale = np.array([part.momentum20.std(ddof=0),np.log(part.adv20).std(ddof=0)])
        scale = np.where(np.isfinite(scale)&(scale>0),scale,1.)
        for r in part[part.absorption.eq(1)].to_dict('records'):
            if controls.empty:
                chosen=None; distance=np.nan
            else:
                differences=np.column_stack([controls.momentum20-r['momentum20'],np.log(controls.adv20/r['adv20'])])/scale
                distances=np.square(differences).sum(axis=1); at=int(np.argmin(distances))
                chosen=controls.iloc[at].event_id; distance=float(distances[at])
            pairs.append(dict(flow_lag=lag,signal_date=date,event_id=r['event_id'],control_id=chosen,distance=distance))
    return pd.DataFrame(pairs,columns=['flow_lag','signal_date','event_id','control_id','distance'])


def comparisons(table,pairs):
    """Date-block descriptive uncertainty, including missing/unmatched counts."""
    attached = pairs.merge(table[['flow_lag','event_id','horizon','phase','fee_return']],on=['flow_lag','event_id'],validate='one_to_many')
    attached = attached.merge(table[['flow_lag','event_id','horizon','fee_return']].rename(
        columns={'event_id':'control_id','fee_return':'control_return'}),on=['flow_lag','control_id','horizon'],how='left',validate='many_to_one')
    attached['delta']=attached.fee_return-attached.control_return
    records=[]
    for (lag,phase,horizon), part in table[~table.named_case].groupby(['flow_lag','phase','horizon'],sort=True):
        comparisons_ = [('matched_sell_weak',attached[(attached.flow_lag==lag)&(attached.phase==phase)&(attached.horizon==horizon)])]
        # Incremental flow information among price-resilient stocks, date-balanced.
        resilient=part[part.resilient.eq(1)]
        date_rows=[]
        for date,g in resilient.groupby('signal_date'):
            a=g[g.absorption.eq(1)].fee_return.dropna(); b=g[g.sell_pressure.eq(0)].fee_return.dropna()
            date_rows.append(dict(signal_date=date,delta=a.mean()-b.mean() if len(a) and len(b) else np.nan))
        comparisons_.append(('resilient_non_selling',pd.DataFrame(date_rows,columns=['signal_date','delta'])))
        for label,g in comparisons_:
            known=g.dropna(subset=['delta']); date_values=known.groupby('signal_date').delta.mean().to_numpy()
            lo=hi=np.nan
            if len(date_values)>=2:
                rng=np.random.default_rng(42)
                means=rng.choice(date_values,size=(1000,len(date_values)),replace=True).mean(axis=1)
                lo,hi=np.quantile(means,[.025,.975])
            records.append(dict(flow_lag=lag,phase=phase,horizon=horizon,comparison=label,observations=len(g),
                known=len(known),unknown=len(g)-len(known),dates=len(date_values),
                event_mean_delta=known.delta.mean(),date_mean_delta=date_values.mean() if len(date_values) else np.nan,
                date_bootstrap_low=lo,date_bootstrap_high=hi))
    return attached,pd.DataFrame(records)
