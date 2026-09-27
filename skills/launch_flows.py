"""Causal institutional and moving-average descriptions, never trading orders."""
import numpy as np
import pandas as pd

PARTS={'foreign':['Foreign_Investor','Foreign_Dealer_Self'], 'trust':['Investment_Trust'],
    'dealer':['Dealer_self','Dealer_Hedging'], 'dealer_self':['Dealer_self'],
    'dealer_hedging':['Dealer_Hedging']}
RULES=('above20','above60','ma_stack','foreign_positive5','trust_positive5',
    'foreign_streak3','trust_streak3','both_positive5','streak_and_trend','concentrated_both')


def normalize(raw):
    required={'date','stock_id','name','buy','sell'}
    if not required.issubset(raw):raise ValueError('Missing institutional columns')
    f=raw[list(required)].copy();f['date']=pd.to_datetime(f.date)
    if not f.stock_id.map(lambda s:isinstance(s,str) and len(s)==4 and s.isdecimal()).all():
        raise ValueError('Invalid ordinary-stock identity')
    if f.duplicated(['date','stock_id','name']).any():raise ValueError('Duplicate institutional category')
    allowed={p for v in PARTS.values() for p in v}
    if not set(f.name).issubset(allowed|{'Dealer'}):raise ValueError('Unknown institutional category')
    for k in ('buy','sell'):
        f[k]=pd.to_numeric(f[k],errors='coerce')
        if not (np.isfinite(f[k])&f[k].ge(0)&f[k].mod(1).eq(0)).all():
            raise ValueError('Invalid institutional share amount')
    f['net']=f.buy-f.sell
    source=f.pivot(index=['date','stock_id'],columns='name',values='net')
    unsupported=source['Dealer'].notna() if 'Dealer' in source else pd.Series(False,index=source.index)
    wide=source.reindex(columns=sorted(allowed)).mask(unsupported,axis=0)
    result=pd.DataFrame({actor:wide[names].sum(axis=1,min_count=len(names)) for actor,names in PARTS.items()})
    result['total']=result[['foreign','trust','dealer']].sum(axis=1,min_count=3)
    result['schema_supported']=~unsupported
    return result.reset_index().sort_values(['date','stock_id']).reset_index(drop=True)


def truth(value,*known):
    mask=pd.DataFrame(True,index=value.index,columns=value.columns)
    for x in known:mask &= np.isfinite(x)
    return value.astype(float).where(mask)


def price_features(close):
    if not close.index.is_unique or not close.index.is_monotonic_increasing or not close.columns.is_unique:
        raise ValueError('Invalid price axes')
    c=close.where(np.isfinite(close)&close.gt(0));out={};mas={}
    for n in (5,10,20,60,120):
        ma=c.rolling(n,min_periods=n).mean();mas[n]=ma
        out[f'ma{n}']=ma;out[f'distance_ma{n}']=c/ma-1
        out[f'above{n}']=truth(c.gt(ma),c,ma)
    slope20=mas[20]/mas[20].shift(5)-1;slope60=mas[60]/mas[60].shift(20)-1
    out.update(slope20=slope20,slope60=slope60,
        ma_stack=truth(c.gt(mas[20])&mas[20].gt(mas[60])&slope20.gt(0)&slope60.gt(0),c,mas[20],mas[60],slope20,slope60),
        distance_high20=c/c.shift(1).rolling(20,min_periods=20).max()-1,
        relative20=(c/c.shift(20)-1).sub(c['0050']/c['0050'].shift(20)-1,axis=0))
    return out


def streak(net,sign):
    """Exact run length capped at 20; an unknown day cannot be crossed."""
    values=net.to_numpy(float)*sign;out=np.full(values.shape,np.nan);state=np.full(values.shape[1],np.nan)
    lower=np.zeros(values.shape[1])
    for i,row in enumerate(values):
        known=np.isfinite(row);positive=known&(row>0)
        lower=np.where(positive,lower+1,0)
        state=np.where(~known,np.nan,np.where(positive,np.minimum(state+1,20),0))
        state=np.where(lower>=20,20,state);out[i]=state
    return pd.DataFrame(out,index=net.index,columns=net.columns)


def flow_features(normalized,days,volume):
    if normalized.duplicated(['date','stock_id']).any():raise ValueError('Duplicate normalized institution day')
    out={}
    for actor in (*PARTS,'total'):
        net=normalized.pivot(index='date',columns='stock_id',values=actor).reindex(days)
        out[actor+'_net']=net
        for n in (5,10,20):out[f'{actor}_net{n}']=net.rolling(n,min_periods=n).sum()
        v=volume.reindex(index=days,columns=net.columns).where(lambda x:np.isfinite(x)&x.gt(0))
        out[actor+'_ratio5']=out[actor+'_net5']/v.rolling(5,min_periods=5).sum()
        for n in (5,10):out[f'{actor}_buy_days{n}']=net.gt(0).astype(float).where(net.notna()).rolling(n,min_periods=n).sum()
        out[actor+'_buy_streak']=streak(net,1);out[actor+'_sell_streak']=streak(net,-1)
        count=net.gt(0).astype(float).where(net.notna()).rolling(3,min_periods=3).sum()
        out[actor+'_streak3']=truth(count.eq(3),count)
    return out


def snapshots(events,price,flow,days):
    rows=[];positions={str(d.date()):i for i,d in enumerate(days)}
    for event in events.to_dict('records'):
        pos=positions[event['signal_date']];sid=event['stock_id']
        for offset in (-20,-10,-5,-1,0):
            i=pos+offset
            if i<0:continue
            row=dict(event,offset=offset,feature_date=str(days[i].date()))
            for k,frame in {**price,**flow}.items():
                row[k]=float(frame.at[days[i],sid]) if sid in frame else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def attach_rules(outcomes,timeline,lag):
    price=timeline[timeline.offset.eq(0)]
    flow=timeline[timeline.offset.eq(-lag)]
    keys=['event_id']
    pcols=['above20','above60','ma_stack']
    fcols=['foreign_net5','trust_net5','foreign_streak3','trust_streak3']
    r=outcomes.merge(price[keys+pcols],on=keys,validate='many_to_one').merge(flow[keys+fcols],on=keys,validate='many_to_one')
    for actor in ('foreign','trust'):
        x=r[actor+'_net5'];r[actor+'_positive5']=x.gt(0).astype('boolean').where(x.notna(),pd.NA)
        r[actor+'_streak3']=r[actor+'_streak3'].astype('boolean')
    for k in pcols:r[k]=r[k].astype('boolean')
    # Require each component to be known even when another component is false.
    def combine(a,b,op='and'):
        a=a.astype('boolean');b=b.astype('boolean')
        return (a&b if op=='and' else a|b).where(a.notna()&b.notna(),pd.NA)
    r['both_positive5']=combine(r.foreign_positive5,r.trust_positive5)
    r['streak_and_trend']=combine(combine(r.foreign_streak3,r.trust_streak3,'or'),r.ma_stack)
    r['concentrated_both']=combine(r.concentration,r.both_positive5)
    r['flow_lag']=lag
    return r


def statistics(frame):
    result=[]
    main=frame[~frame.named_case]
    for (lag,horizon,phase),part in main.groupby(['flow_lag','horizon','phase'],sort=True):
        for scope,sub in [('all_first_bars',part),('within_trend',part[part.ma_stack.eq(True).fillna(False)])]:
            rules=RULES if scope=='all_first_bars' else ('foreign_positive5','trust_positive5','both_positive5')
            for rule in rules:
                for state in (True,False):
                    f=sub[sub[rule].eq(state).fillna(False)];known=f.dropna(subset=['first_return','first_excess','first_surge'])
                    result.append(dict(flow_lag=int(lag),horizon=int(horizon),phase=phase,scope=scope,rule=rule,value=state,
                        cohort_events=len(sub),condition_unknown=int(sub[rule].isna().sum()),events=len(f),
                        known=len(known),outcome_unknown=len(f)-len(known),stocks=known.stock_id.nunique(),
                        surge_hits=int(known.first_surge.eq(True).sum()),surge_rate=known.first_surge.astype(float).mean(),
                        mean_return=known.first_return.mean(),median_return=known.first_return.median(),
                        mean_excess=known.first_excess.mean(),false_start_known=int(f.false_start5.notna().sum()),
                        false_start5=f.false_start5.dropna().astype(float).mean()))
    return result
