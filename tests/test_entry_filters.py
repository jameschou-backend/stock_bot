import numpy as np
import pandas as pd
from skills.entry_filters import apply_filters,filter_features


def inputs():
    days=pd.bdate_range('2020-01-01',periods=200)
    c=pd.DataFrame({'1234':10*np.exp(np.arange(200)*.002),
                    '0050':10*np.exp(np.arange(200)*.001)},index=days)
    frames={k:c.copy() for k in ('close-official','close-quality','raw-close')}
    frames.update({'raw-volume':c*0+10_000_000,'eligibility':c.notna()})
    companies=pd.DataFrame([{'stock_id':'1234','listed_date':pd.Timestamp('2000-01-01')}])
    q=pd.DataFrame([dict(date=d,stock_id=s,open=v*.99,high=v*1.01,low=v*.95,close=v)
                    for d,row in c.iterrows() for s,v in row.items()])
    entries=[dict(event_id='x',signal_date=str(days[170].date()),entry_date=str(days[171].date()),members=['1234'],priority=.1)]
    return frames,companies,q,entries


def test_no_retiming_and_unknown_breadth_is_excluded():
    f,c,q,e=inputs();out,rows=apply_filters(e,filter_features(f,c,q))
    assert out['control3']==e and out['not_extended3']==e and out['strong_close3']==e
    assert out['breadth3']==[] and rows[0]['reasons']['breadth3']=='unknown'
    assert out['control3'][0] is not e[0]


def test_flat_or_invalid_bar_is_unknown_not_passing():
    f,c,q,e=inputs();mask=(q.stock_id=='1234')&(q.date==e[0]['signal_date'])
    q.loc[mask,'low']=q.loc[mask,'close'];q.loc[mask,'high']=q.loc[mask,'close']
    out,rows=apply_filters(e,filter_features(f,c,q))
    assert not out['strong_close3'] and rows[0]['reasons']['strong_close3']=='unknown'


def test_future_mutation_cannot_change_filters():
    f,c,q,e=inputs();original=apply_filters(e,filter_features(f,c,q))
    for key in ('close-official','close-quality','raw-close','raw-volume'):
        f[key].iloc[172:]*=.1
    q.loc[q.date>pd.Timestamp(e[0]['signal_date']),['open','high','low','close']]*=.1
    assert apply_filters(e,filter_features(f,c,q))==original


def test_strict_filter_thresholds_do_not_select_hindsight_winners():
    f,c,q,e=inputs();features=filter_features(f,c,q);d=pd.Timestamp(e[0]['signal_date'])
    features['distance20'].at[d,'1234']=.16
    features['close_location'].at[d,'1234']=.69
    features['breadth'].at[d]=.6;features['breadth5'].at[d]=.7
    selected,_=apply_filters(e,features)
    assert selected['control3']==e and all(not selected[a] for a in selected if a!='control3')


def test_breadth_excludes_etf_and_illiquid_names_from_denominator():
    days=pd.bdate_range('2020-01-01',periods=200)
    ids=[str(1100+i) for i in range(40)]
    c=pd.DataFrame({sid:10*np.exp(np.arange(200)*(.001 if i<20 else -.001))
                    for i,sid in enumerate(ids)},index=days)
    c['0050']=10*np.exp(np.arange(200)*.002)
    frames={k:c.copy() for k in ('close-official','close-quality','raw-close')}
    frames.update({'raw-volume':c*0+10_000_000,'eligibility':c.notna()})
    frames['raw-volume'].loc[:,ids[30:]]=1
    companies=pd.DataFrame([dict(stock_id=s,listed_date=pd.Timestamp('2000-01-01')) for s in ids])
    q=pd.DataFrame([dict(date=d,stock_id=s,open=v*.99,high=v*1.01,low=v*.95,close=v)
                    for d,row in c.iterrows() for s,v in row.items()])
    f=filter_features(frames,companies,q)
    assert f['breadth_count'].iloc[-1]==30
    assert f['breadth'].iloc[-1]==20/30
    assert f['breadth5'].iloc[-1]==20/30
