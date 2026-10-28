import numpy as np
import pandas as pd
import pytest

from skills.course_bundle import Config, configurations, prepare_features, signals, simulate
from scripts.research_course_bundle import choose_validation


def fixture_prices(n=100):
    days=pd.bdate_range('2021-01-01',periods=n)
    rows=[];inst=[];money=[]
    for k,sid in enumerate(('0050','1101','1102','1103')):
        for i,day in enumerate(days):
            px=100+i*(.2+k*.05)
            vol=10_000_000+i*1000
            rows.append(dict(date=day,stock_id=sid,open=px-.2,high=px+.3,low=px-.4,
                close=px,volume=vol,adjusted_close=px,eligible=True,quality=True,
                source_disagreement=False,source_return_known=True))
            inst.append(dict(date=day,stock_id=sid,trust_net=1000,foreign_net=2000))
            money.append(dict(date=day,stock_id=sid,Trading_money=px*vol,
                Trading_Volume=vol,close=px))
    return pd.DataFrame(rows),pd.DataFrame(inst),pd.DataFrame(money)


def execution_fixture():
    days=pd.bdate_range('2024-01-01',periods=7)
    ids=pd.Index(['0050','1101','1102'])
    frame=lambda x:pd.DataFrame(x,index=days,columns=ids)
    f=dict(days=days,ids=ids,open=frame(100.),low=frame(99.),close=frame(100.),
        raw_open=frame(100.),volume=frame(10_000_000.),trade=frame(True),
        conflict=frame(False),halt=frame(False),ma={5:frame(90.),10:frame(90.)})
    f['mark_close']=f['close']
    rank=frame(np.nan);rank.loc[days[1],'1101']=90
    return f,dict(ranking=rank),str(days[2].date()),str(days[-1].date())


def test_fixed_search_has_24_unique_configurations():
    values=configurations()
    assert len(values)==len({v.name for v in values})==24


def test_features_and_signals_are_prefix_invariant():
    bars,inst,money=fixture_prices()
    cut=bars.date.unique()[75]
    full=prepare_features(bars,inst,money)
    short=prepare_features(bars[bars.date<=cut],inst[inst.date<=cut],money[money.date<=cut])
    for a,b in zip(full['ranks'],short['ranks']):
        pd.testing.assert_frame_equal(a.loc[:cut],b)
    pd.testing.assert_frame_equal(signals(full,Config())['ranking'].loc[:cut],signals(short,Config())['ranking'])
    changed=bars.copy();changed.loc[changed.date>cut,['close','adjusted_close']]*=3
    revised=prepare_features(changed,inst,money)
    pd.testing.assert_frame_equal(signals(full,Config())['ranking'].loc[:cut],signals(revised,Config())['ranking'].loc[:cut])


def test_missing_institutional_row_is_not_zero_and_turnover_is_actual():
    bars,inst,money=fixture_prices()
    day=bars.date.max()
    inst=inst[~((inst.date==day)&(inst.stock_id=='1101'))]
    money.loc[(money.date==pd.bdate_range(end=day,periods=2)[0])&(money.stock_id=='1102'),'Trading_money']=1
    f=prepare_features(bars,inst,money)
    assert not f['common'].at[day,'1101']
    assert not f['common'].at[day,'1102']
    assert not f['common']['0050'].any()
    assert f['raw_close'].at[day,'1102']*f['volume'].at[day,'1102']>500_000_000


def test_next_open_not_signal_close_and_no_future_budget():
    f,s,start,end=execution_fixture()
    f['close'].loc[pd.Timestamp(start),'1101']=105
    run=simulate(f,s,Config(),start,end)
    buy=run['trades'][0]
    assert buy['date']==start
    assert buy['signal_date']<buy['date']
    assert buy['adjusted_price']==pytest.approx(100.1)
    assert run['decisions'][0]['target_budget']==200_000
    assert run['curve'].cash.min()>=0


def test_stop_active_on_entry_day():
    f,s,start,end=execution_fixture()
    f['low'].at[pd.Timestamp(start),'1101']=90
    run=simulate(f,s,Config(),start,end)
    buy,sell=run['trades']
    assert buy['date']==sell['date']==start
    assert sell['reason']=='hard_stop'
    assert sell['adjusted_price']==pytest.approx(100.1*.93*.999)
    assert run['summary']['completed_trades']==1
    assert run['summary']['final_net_nav']-1_000_000==pytest.approx(sell['net_pnl'])


def test_gap_stop_fills_at_adverse_open_and_ma_is_delayed():
    f,s,start,end=execution_fixture();day=f['days'][3]
    f['open'].at[day,'1101']=80;f['low'].at[day,'1101']=79;f['close'].at[day,'1101']=81
    run=simulate(f,s,Config(),start,end)
    assert run['trades'][1]['adjusted_price']==pytest.approx(80*.999)
    assert run['trades'][1]['reason']=='hard_stop_gap'
    f,s,start,end=execution_fixture();day=f['days'][3]
    f['ma'][10].at[day,'1101']=101
    run=simulate(f,s,Config(),start,end)
    assert run['trades'][1]['date']==str(f['days'][4].date())
    assert run['trades'][1]['reason']=='previous_close_below_ma'


def test_locked_stop_is_pending_even_if_price_recovers():
    f,s,start,end=execution_fixture();day=f['days'][3]
    f['trade'].at[day,'1101']=False;f['low'].at[day,'1101']=80
    run=simulate(f,s,Config(),start,end)
    assert run['trades'][1]['reason']=='delayed_hard_stop'
    assert run['trades'][1]['date']==str(f['days'][4].date())


def test_gap_entry_is_skipped_and_missing_held_price_halts():
    f,s,start,end=execution_fixture()
    f['open'].at[pd.Timestamp(start),'1101']=110
    run=simulate(f,s,Config(),start,end)
    assert not run['trades'] and run['summary']['gap_rejections']==1
    f,s,start,end=execution_fixture()
    f['close'].at[f['days'][3],'1101']=np.nan
    with pytest.raises(ValueError,match='Held-position'):
        simulate(f,s,Config(),start,end)


def test_terminal_mark_has_cost_estimate_without_fictitious_closed_trade():
    f,s,start,end=execution_fixture()
    run=simulate(f,s,Config(),start,end)
    assert run['summary']['completed_trades']==0
    assert run['summary']['terminal_positions']==1
    assert run['curve'].iloc[-1].equity<run['curve'].iloc[-1].gross_equity


@pytest.mark.parametrize('empty_bar',[0,np.nan])
def test_only_verified_halts_receive_valuation_marks_not_signals(empty_bar):
    bars,inst,money=fixture_prices();day=bars.date.unique()[75]
    mask=(bars.date==day)&(bars.stock_id=='1101')
    bars.loc[mask,['open','high','low','close','volume']]=empty_bar
    bars.loc[mask,'adjusted_close']=np.nan
    bars.loc[mask,['eligible','quality','source_return_known']]=False
    halt=dict(stock_id='1101',start=str(pd.Timestamp(day).date()),end=str((pd.Timestamp(day)+pd.offsets.BDay()).date()))
    f=prepare_features(bars,inst,money,halts=[halt])
    assert pd.isna(f['close'].at[day,'1101'])
    assert f['mark_close'].at[day,'1101']>0
    assert not f['trade'].at[day,'1101']
    unknown=prepare_features(bars,inst,money)
    assert pd.isna(unknown['mark_close'].at[day,'1101'])


def test_zero_halt_bar_does_not_trigger_a_stop():
    f,s,start,end=execution_fixture();day=f['days'][3]
    f['halt'].at[day,'1101']=True;f['trade'].at[day,'1101']=False
    f['low'].at[day,'1101']=0;f['open'].at[day,'1101']=0
    run=simulate(f,s,Config(),start,end)
    assert len(run['trades'])==1
    assert run['summary']['verified_halt_hold_days']==1


def test_benchmark_halt_preserves_adjusted_value_and_has_no_split_sale():
    f,s,start,end=execution_fixture();day=f['days'][3]
    f['halt'].at[day,'0050']=True;f['trade'].at[day,'0050']=False
    f['mark_close']=f['close'].copy()
    f['close'].at[day,'0050']=np.nan
    f['open'].at[day,'0050']=np.nan
    run=simulate(f,None,Config(),start,end,benchmark=True)
    assert len(run['trades'])==1
    assert run['summary']['verified_halt_hold_days']==1
    curve=run['curve'].set_index('date')
    assert curve.at[day,'equity']==curve.at[f['days'][2],'equity']


def test_selection_reads_validation_and_tiebreaks_drawdown():
    rows=[dict(name='a',summary=dict(total_return=.2,max_drawdown=-.3,traded_notional=100)),
          dict(name='b',summary=dict(total_return=.2,max_drawdown=-.1,traded_notional=200))]
    assert choose_validation(rows)['name']=='b'
