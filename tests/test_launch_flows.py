import numpy as np
import pandas as pd
import pytest
from skills.launch_flows import normalize,streak,price_features,flow_features,attach_rules


def raw():
    return pd.DataFrame([dict(date='2025-01-02',stock_id='3491',name=n,buy=b,sell=s) for n,b,s in
        [('Foreign_Investor',100,20),('Foreign_Dealer_Self',10,2),('Investment_Trust',0,0),
         ('Dealer_self',20,30),('Dealer_Hedging',50,70)]])


def test_components_are_not_double_counted_and_zero_is_known():
    r=normalize(raw()).iloc[0]
    assert r.foreign==88 and r.trust==0 and r.dealer==-30 and r.total==58


def test_missing_component_remains_unknown_but_other_actors_are_usable():
    f=raw();r=normalize(f[f.name.ne('Foreign_Dealer_Self')]).iloc[0]
    assert pd.isna(r.foreign) and r.trust==0 and pd.isna(r.total)


@pytest.mark.parametrize('problem',['duplicate','negative','fraction','missing','category','identity'])
def test_invalid_raw_evidence_fails_explicitly(problem):
    f=raw()
    if problem=='duplicate':f=pd.concat([f,f.iloc[:1]])
    elif problem=='negative':f.loc[0,'buy']=-1
    elif problem=='fraction':f['buy']=f.buy.astype(float);f.loc[0,'buy']=1.5
    elif problem=='missing':f.loc[0,'buy']=np.nan
    elif problem=='category':f.loc[0,'name']='Unknown_Category'
    else:f.loc[0,'stock_id']='00631L'
    with pytest.raises(ValueError):normalize(f)


def test_unsupported_emerging_market_schema_is_explicitly_unknown_not_merged():
    f=raw();f.loc[3,'name']='Dealer'
    r=normalize(f).iloc[0]
    assert not r.schema_supported and pd.isna(r.foreign) and pd.isna(r.dealer) and pd.isna(r.trust)


def test_streak_does_not_cross_unknown_and_zero_interrupts_buying():
    f=pd.DataFrame({'a':[0,1,2,np.nan,1,2,3,0,-1,-2]})
    result=streak(f,1).a
    assert result.iloc[2]==2 and pd.isna(result.iloc[6]) and result.iloc[7]==0
    assert streak(f,-1).a.iloc[-1]==2
    assert streak(pd.DataFrame({'a':[1]*22}),1).a.iloc[-1]==20


def test_missing_market_day_invalidates_sums_but_three_observed_buys_are_known():
    days=pd.bdate_range('2025-01-01',periods=25)
    f=pd.DataFrame({'date':days,'stock_id':'3491',**{k:1. for k in ('foreign','trust','dealer','dealer_self','dealer_hedging','total')}})
    f=f.drop(index=3);v=pd.DataFrame({'3491':100.},index=days)
    r=flow_features(f,days,v)
    assert pd.isna(r['trust_net5'].iloc[7,0]) and r['trust_net5'].iloc[8,0]==5
    assert r['trust_streak3'].iloc[6,0]==1 and pd.isna(r['trust_buy_streak'].iloc[6,0])
    assert r['trust_ratio5'].iloc[8,0]==.01


def test_ma_trend_requires_slopes_and_future_prices_cannot_change_history():
    days=pd.bdate_range('2024-01-01',periods=160)
    c=pd.DataFrame({'3491':np.arange(160)+100.,'0050':100.},index=days)
    before=price_features(c);assert before['ma_stack'].iloc[120,0]==1
    assert before['ma20'].iloc[120,0]==pytest.approx(c['3491'].iloc[101:121].mean())
    c.iloc[130:,0]*=3;after=price_features(c)
    for k in before:pd.testing.assert_frame_equal(before[k].iloc[:130],after[k].iloc[:130])


def test_prelaunch_flow_is_separate_from_launch_day_and_unknown_not_negative():
    result=pd.DataFrame([dict(event_id='a',concentration=True)])
    records=[]
    for off,value in [(-1,-100),(0,100)]:
        records.append(dict(event_id='a',offset=off,above20=1,above60=1,ma_stack=1,
            foreign_net5=value,trust_net5=np.nan,foreign_streak3=0,trust_streak3=np.nan))
    timeline=pd.DataFrame(records)
    assert not attach_rules(result,timeline,1).foreign_positive5.iloc[0]
    assert attach_rules(result,timeline,0).foreign_positive5.iloc[0]
    assert pd.isna(attach_rules(result,timeline,1).both_positive5.iloc[0])
