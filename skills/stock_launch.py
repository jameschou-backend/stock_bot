"""Retrospective individual-stock launch anatomy with causal features and controls."""
import numpy as np
import pandas as pd
from skills.surge_anatomy import features as previous_features,_known_comparison

TARGETS={'2308':'台達電','3026':'禾伸堂','2327':'國巨','2492':'華新科','2344':'華邦電',
         '2408':'南亞科','1303':'南亞','2303':'聯電','3481':'群創'}
RULES=['all_eligible','relative_strength','breakout60','turnover_heat','breakout_strength','money_strength','early_rotation']
NUMERIC=['momentum20','momentum60','relative20','volume_ratio','volatility_ratio','distance_high60','turnover_ratio']
OBSERVED=['breakout60','relative_strength','rising_trend','compression','near_high','turnover_heat','early_rotation']
HORIZONS={20:(.30,.20),60:(.50,.30)}


def features(close,raw,volume,companies):
    base=previous_features(close,raw,volume,companies)
    amount=(raw*volume).where((raw>0)&(volume>0)&np.isfinite(raw)&np.isfinite(volume))
    ratio=amount.rolling(5,min_periods=5).mean()/amount.shift(5).rolling(20,min_periods=20).mean()
    ratio=ratio[base['eligible'].columns];base['numeric']['turnover_ratio']=ratio
    rules=base['rules'];relative=base['numeric']['relative20'];mom60=base['numeric']['momentum60']
    rules['turnover_heat']=_known_comparison(ratio.ge(1.5),ratio)
    rules['breakout_strength']=rules['breakout60']&rules['relative_strength']
    rules['money_strength']=rules['turnover_heat']&_known_comparison(relative.gt(0),relative)
    rules['early_rotation']=rules['money_strength']&rules['near_high']&_known_comparison(mom60.le(.30),mom60)
    return base


def outcomes(close,quality,horizon):
    if horizon not in HORIZONS:raise ValueError('Only the two preregistered horizons are supported')
    if not close.index.equals(quality.index) or not close.columns.equals(quality.columns):raise ValueError('Unaligned prices')
    if not close.index.is_unique or not close.index.is_monotonic_increasing or not close.columns.is_unique or '0050' not in close:
        raise ValueError('Invalid market calendar or benchmark')
    h=horizon+1;returns=[];known=[]
    for c in (close,quality):
        valid=(c>0)&np.isfinite(c)
        complete=valid.rolling(h,min_periods=h).sum().shift(-h).eq(h)
        jump=(c/c.shift(1)-1).abs().gt(.15).rolling(h,min_periods=h).max().shift(-h).eq(1)
        returns.append(c.shift(-h)/c.shift(-1)-1);known.append(valid&complete&~jump)
    good=known[0]&known[1]&(returns[0]-returns[1]).abs().le(.05)
    good=good.mul(good['0050'],axis=0).astype(bool)
    excess=returns[0].sub(returns[0]['0050'],axis=0)
    absolute,relative=HORIZONS[horizon]
    event=((returns[0]>=absolute)&(excess>=relative)).astype('boolean').where(good,pd.NA)
    return dict(event=event,forward_return=returns[0].where(good),excess_return=excess.where(good),
                alternate_return=returns[1].where(good),known=good)


def snapshot(computed,label,calendar,pos,horizon,*,only=None,eligible_only=True):
    day=calendar[pos];ids=computed['eligible'].columns
    eligible=computed['eligible'].iloc[pos]
    selected=ids[eligible] if eligible_only else ids
    if only is not None:selected=selected.intersection(only)
    out=computed['companies'].loc[selected,['name','industry','market']].copy()
    out['stock_id']=out.index;out['signal_date']=str(day.date());out['horizon']=horizon
    mature=pos+horizon+1<len(calendar)
    out['entry_date']=str(calendar[pos+1].date()) if pos+1<len(calendar) else None
    out['exit_date']=str(calendar[pos+horizon+1].date()) if mature else None
    out['phase']=('discovery' if mature and out['exit_date'].iloc[0]<='2024-12-31' else
                  'replication' if day>=pd.Timestamp('2025-01-01') else 'boundary') if len(out) else pd.Series(dtype=str)
    out['eligible']=eligible.loc[selected];out['label_mature']=mature
    adv=computed['adv20'].iloc[pos];all_adv=adv[eligible]
    bins=np.minimum((all_adv.rank(method='average',pct=True)*5).astype(int),4)
    out['adv20']=adv.loc[selected];out['liquidity_bin']=bins.reindex(selected)
    for key in ('event','forward_return','excess_return','alternate_return'):
        out[key]=label[key].iloc[pos].loc[selected]
    out['label_known']=label['known'].iloc[pos].loc[selected]
    for key in sorted(set(NUMERIC)|set(RULES)|set(OBSERVED)):
        source=computed['numeric'] if key in NUMERIC else computed['rules']
        out[key]=source[key].iloc[pos].loc[selected]
    return out.reset_index(drop=True)


def episode_positions(panel,calendar,horizon):
    selected=[];positions={str(d.date()):i for i,d in enumerate(calendar)}
    for sid,part in panel.groupby('stock_id',sort=True):
        next_pos=-1
        for row in part.sort_values('signal_date').to_dict('records'):
            pos=positions[row['signal_date']]
            if row['eligible'] and row['event'] is not pd.NA and pd.notna(row['event']) and bool(row['event']) and pos>=next_pos:
                selected.append((sid,pos));next_pos=pos+horizon+1
    return selected


def choose_controls(pool,case,excluded):
    candidates=pool[(~pool.stock_id.isin(excluded))&pool.event.eq(False).fillna(False)
                    &pool.market.eq(case['market'])&pool.industry.eq(case['industry'])
                    &pool.liquidity_bin.eq(case['liquidity_bin'])].copy()
    candidates['match_distance']=(np.log(candidates.adv20)-np.log(case['adv20'])).abs()
    return candidates.sort_values(['match_distance','stock_id']).head(3)


def trajectory(computed,calendar,sid,anchor,case_id,role):
    result=[]
    for offset in (-20,-10,-5,0):
        pos=anchor+offset
        if pos<0:continue
        row=dict(case_id=case_id,role=role,stock_id=sid,offset=offset,signal_date=str(calendar[pos].date()),
                 eligible=bool(computed['eligible'].iloc[pos][sid]))
        for key in [*NUMERIC,*OBSERVED]:
            row[key]=(computed['numeric'] if key in NUMERIC else computed['rules'])[key].iloc[pos][sid]
        result.append(row)
    return result


def paired_summary(timeline):
    results=[]
    if timeline.empty:return results
    for (horizon,offset),part in timeline.groupby(['horizon','offset'],sort=True):
        cases=part[part.role.eq('case')].set_index('case_id')
        controls=part[part.role.eq('control')].groupby('case_id')
        for key in [*NUMERIC,*OBSERVED]:
            left=pd.to_numeric(cases[key],errors='coerce').astype(float)
            right=controls[key].agg(lambda x:pd.to_numeric(x,errors='coerce').mean())
            pair=pd.concat([left.rename('case'),right.rename('control')],axis=1).dropna()
            results.append(dict(horizon=int(horizon),offset=int(offset),feature=key,
                events=len(cases),matched_events=len(pair),
                case_mean=pair['case'].mean(),control_mean=pair['control'].mean(),
                mean_difference=(pair['case']-pair['control']).mean(),
                median_paired_difference=(pair['case']-pair['control']).median()))
    return results


def causal_checks(close,raw,volume,companies,cutoffs):
    """Compare every historical feature, rule and eligibility, never outcomes."""
    baseline=features(close,raw,volume,companies);results=[]
    for cutoff in cutoffs:
        day=pd.Timestamp(cutoff)
        for mode in ('truncate','mutate'):
            frames=[]
            for source in (close,raw,volume):
                frame=source.loc[:day].copy() if mode=='truncate' else source.copy()
                if mode=='mutate':frame.loc[frame.index>day]*=3.7
                frames.append(frame)
            actual=features(*frames,companies)
            count=0
            for group in ('numeric','rules'):
                for key in baseline[group]:
                    pd.testing.assert_frame_equal(baseline[group][key].loc[:day],actual[group][key].loc[:day]);count+=1
            pd.testing.assert_frame_equal(baseline['eligible'].loc[:day],actual['eligible'].loc[:day]);count+=1
            results.append(dict(cutoff=str(day.date()),mode=mode,compared_matrices=count,passed=True))
    return results
