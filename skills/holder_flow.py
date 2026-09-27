"""Holder concentration crossed with separate institutional directions.

These are observable proxies, not identities, intent or executable returns.
"""
from copy import deepcopy
import numpy as np
import pandas as pd
from skills.theme_chips import align

FLOWS = ('both_buy', 'foreign_sell_trust_buy', 'foreign_buy_trust_sell', 'both_sell', 'zero_actor')
ARMS = ('all_known', 'concentration', 'relative_strength', 'concentrated_strength',
        *('cs_'+state for state in FLOWS[:-1]))
ACCOUNT_ARMS = ARMS[2:]
KEYS = ['event_id', 'flow_lag', 'chip_lag']


def combine(base, weekly, lag):
    """Only input observations; no forward outcome may affect a condition."""
    if base.duplicated(['event_id', 'flow_lag']).any():
        raise ValueError('Duplicate candidate event/lag')
    result = align(base, weekly, lag)
    result['chip_lag'] = lag
    foreign, trust = result.foreign5, result.trust5
    flow_known = np.isfinite(foreign) & np.isfinite(trust) & np.isfinite(result.shares5) & result.shares5.gt(0)
    result['foreign_ratio'] = (foreign/result.shares5).where(flow_known)
    result['trust_ratio'] = (trust/result.shares5).where(flow_known)
    result['actor_state'] = np.select([
        ~flow_known, (foreign>0)&(trust>0), (foreign<0)&(trust>0),
        (foreign>0)&(trust<0), (foreign<0)&(trust<0)], ['unknown', *FLOWS[:-1]], default='zero_actor')
    c = result.large_pct_delta4.ge(.005) & result.small_pct_delta4.lt(0) & result.large_units_delta4.gt(0)
    d = result.large_pct_delta4.le(-.005) & result.small_pct_delta4.gt(0) & result.large_units_delta4.lt(0)
    result['holder_state'] = np.select([~result.chip_known, c, d], ['unknown', 'concentrated', 'distributed'], default='neutral')
    known = flow_known & result.chip_known & np.isfinite(result.relative20)
    strong = result.relative20.ge(.10)
    result['joint_known'] = known
    states = (pd.Series(True, index=result.index), c, strong, c&strong,
              *(c&strong&result.actor_state.eq(state) for state in FLOWS[:-1]))
    for arm, state in zip(ARMS, states):
        result[arm] = state.astype(float).where(known)
    return result


def orders(table, days):
    """Fixed main-lag orders, without future labels or execution assumptions."""
    source = table[table.flow_lag.eq(1) & table.chip_lag.eq(8) & ~table.named_case]
    positions = {str(day.date()):i for i,day in enumerate(days)}
    result = {}
    for arm in ACCOUNT_ARMS:
        rows = []
        for r in source[source[arm].eq(1)].to_dict('records'):
            pos = positions[r['signal_date']]
            if pos+1 >= len(days) or days[pos+1] > pd.Timestamp('2026-09-09'):
                continue
            liquidity = dict(as_of=r['signal_date'], complete_20_sessions=True, observations=20,
                             adv20_shares=r['adv20_shares'], mean_turnover20_twd=r['adv20'])
            rows.append(dict(event_id=r['event_id'], stock_id=r['stock_id'], members=[r['stock_id']],
                signal_date=r['signal_date'], entry_date=str(days[pos+1].date()), priority=r['adv20'],
                feature_cutoff_date=r['signal_date'], group_cutoff_date=r['signal_date'],
                institutional_feature_date=r['feature_date'], holder_observed_date=r['observed_date'],
                holder_available_date=r['available_date'], membership_point_in_time=False,
                membership_snapshot_date='2026-09-27', liquidity_at_signal=liquidity,
                liquidity_before_entry=deepcopy(liquidity), actor_state=r['actor_state']))
        result[arm] = sorted(rows, key=lambda r:(r['signal_date'], -r['priority'], r['stock_id']))
    return result


def metrics(part, group):
    known = part.dropna(subset=['forward_return'])
    hits = int(known.surge.eq(True).sum())
    total_hits = int((group.joint_known & group.surge.eq(True)).sum())
    return dict(eligible=len(group), feature_unknown=int((~group.joint_known).sum()),
        signals=len(part), known=len(known), outcome_unknown=len(part)-len(known),
        stocks=part.stock_id.nunique(), dates=known.signal_date.nunique(),
        mean_return=known.forward_return.mean(), median_return=known.forward_return.median(),
        fee_return=known.fee_return.mean(), fee_excess=known.fee_excess.mean(),
        stress_fee_return=known.stress_fee_return.mean(), stress_fee_excess=known.stress_fee_excess.mean(),
        surge_hits=hits, surge_rate=hits/len(known) if len(known) else np.nan,
        common_surges=total_hits, capture_rate=hits/total_hits if total_hits else np.nan,
        small_sample=len(known)<30 or known.signal_date.nunique()<5)


def summaries(table, annual=False, cells=False):
    rows=[]
    groups=['flow_lag', 'chip_lag', 'year' if annual else 'phase', 'horizon']
    for keys, group in table[~table.named_case].groupby(groups, sort=True):
        identity=dict(zip(groups,keys))
        if cells:
            for state in ('concentrated','distributed','neutral'):
                for strong in (0,1):
                    for actor in FLOWS:
                        part=group[group.joint_known & group.holder_state.eq(state)
                                   & group.relative_strength.eq(strong) & group.actor_state.eq(actor)]
                        rows.append(dict(identity,holder_state=state,strong=strong,actor_state=actor,**metrics(part,group)))
        else:
            for arm in ARMS:
                rows.append(dict(identity,arm=arm,**metrics(group[group[arm].eq(1)],group)))
    return pd.DataFrame(rows)


def matched_pairs(table):
    """Fixed date/industry controls, selected solely by existing features."""
    source=table[~table.named_case & table.joint_known & table.relative_strength.eq(1)]
    rows=[]
    def match(part, treated, controls, kind):
        if treated.empty:
            return
        controls=controls.sort_values('stock_id')
        scale=np.array([part.momentum20.std(ddof=0),np.log(part.adv20).std(ddof=0)])
        scale=np.where(np.isfinite(scale)&(scale>0),scale,1.)
        for r in treated.to_dict('records'):
            chosen=None; distance=np.nan
            if not controls.empty:
                values=np.column_stack([controls.momentum20-r['momentum20'], np.log(controls.adv20/r['adv20'])])/scale
                squared=np.square(values).sum(axis=1); index=int(np.argmin(squared))
                chosen=controls.iloc[index].event_id; distance=float(squared[index])
            rows.append(dict(kind=kind,flow_lag=r['flow_lag'],chip_lag=r['chip_lag'],
                signal_date=r['signal_date'],event_id=r['event_id'],actor_state=r['actor_state'],
                control_id=chosen,distance=distance))
    groups=['flow_lag','chip_lag','signal_date','industry']
    for _, part in source.groupby(groups+['actor_state'],sort=True):
        match(part,part[part.concentration.eq(1)],part[part.concentration.eq(0)],'concentration')
    for _, part in source[source.concentration.eq(1)].groupby(groups,sort=True):
        match(part,part[part.actor_state.eq('foreign_sell_trust_buy')],part[part.actor_state.eq('both_buy')],'divergence_vs_both_buy')
    return pd.DataFrame(rows,columns=['kind','flow_lag','chip_lag','signal_date','event_id','actor_state','control_id','distance'])


def comparisons(table,pairs):
    labels=table[KEYS+['horizon','phase','fee_return','surge']]
    if labels.duplicated(KEYS+['horizon']).any():raise ValueError('Duplicate outcome keys')
    attached=pairs.merge(labels,on=KEYS,validate='many_to_many')
    control=labels.drop(columns='phase').rename(columns={'event_id':'control_id','fee_return':'control_return','surge':'control_surge'})
    attached=attached.merge(control,on=['flow_lag','chip_lag','control_id','horizon'],how='left',validate='many_to_one')
    attached['return_delta']=attached.fee_return-attached.control_return
    attached['hit_delta']=pd.to_numeric(attached.surge,errors='coerce')-pd.to_numeric(attached.control_surge,errors='coerce')
    records=[]
    for keys,group in attached.groupby(['kind','flow_lag','chip_lag','phase','horizon'],sort=True):
        scopes=['all_flows',*FLOWS] if keys[0]=='concentration' else ['foreign_sell_trust_buy']
        for scope in scopes:
            part=group if scope=='all_flows' else group[group.actor_state.eq(scope)]
            for metric in ('return_delta','hit_delta'):
                known=part.dropna(subset=[metric])
                date_values=known.groupby('signal_date')[metric].mean().to_numpy()
                lo=hi=np.nan
                if len(date_values)>=5:
                    rng=np.random.default_rng(20260927)
                    means=rng.choice(date_values,size=(1000,len(date_values)),replace=True).mean(axis=1)
                    lo,hi=np.quantile(means,[.025,.975])
                records.append(dict(zip(['kind','flow_lag','chip_lag','phase','horizon'],keys),
                    scope=scope,metric=metric,observations=len(part),matched=int(part.control_id.notna().sum()),
                    known=len(known),unknown=len(part)-len(known),dates=len(date_values),
                    event_mean_delta=known[metric].mean(),date_mean_delta=date_values.mean() if len(date_values) else np.nan,
                    date_bootstrap_low=lo,date_bootstrap_high=hi))
    return attached,pd.DataFrame(records)
