import numpy as np
import pandas as pd
import pytest

from skills.course_bundle import Config, FAMILIES, prepare_features, signals, simulate
from skills.course_deep_research import (
    clean_signal_close, intervene, shift_institutional_availability,
    select_training, trade_attribution, overlap_diagnostics, paired_block_interval, forward_signal_study,
)


def features(n=280):
    days = pd.bdate_range('2020-01-01', periods=n)
    bars, inst, money = [], [], []
    for j, sid in enumerate(('0050', '1101', '1102')):
        for i, day in enumerate(days):
            px = 100 + i*(.05 + j*.02) + np.sin(i/9)
            volume = 10_000_000 + i*1000
            bars.append(dict(date=day, stock_id=sid, open=px-.2, high=px+.5,
                low=px-.5, close=px, adjusted_close=px, volume=volume,
                eligible=True, quality=True, source_return_known=True, source_disagreement=False))
            inst.append(dict(date=day, stock_id=sid, trust_net=1000., foreign_net=2000.))
            money.append(dict(date=day, stock_id=sid, close=px, Trading_Volume=volume, Trading_money=volume*px))
    bars, inst, money = map(pd.DataFrame, (bars, inst, money))
    return bars, inst, money


@pytest.mark.parametrize('name', ['momentum_60', 'market_ma200', 'exit_ma20_confirm2', 'first_signal_10'])
def test_interventions_are_prefix_invariant_and_do_not_mutate_reference(name):
    bars, inst, money = features()
    cutoff = sorted(bars.date.unique())[240]
    f = prepare_features(bars, inst, money)
    original = signals(f, Config()).get('ranking').copy()
    before_ma = f['ma'][10].copy()
    _, whole = intervene(f, Config(), name, clean_signal_close(bars))
    short_bars = bars[bars.date <= cutoff]
    g = prepare_features(short_bars, inst[inst.date <= cutoff], money[money.date <= cutoff])
    _, short = intervene(g, Config(), name, clean_signal_close(short_bars))
    pd.testing.assert_frame_equal(whole['ranking'].loc[:cutoff], short['ranking'])
    pd.testing.assert_frame_equal(original, signals(f, Config())['ranking'])
    pd.testing.assert_frame_equal(before_ma, f['ma'][10])


def test_confirmation_waits_two_closes_but_does_not_delay_hard_stop():
    bars, inst, money = features(100)
    f = prepare_features(bars, inst, money); clean = clean_signal_close(bars)
    days = f['days']; sid = '1101'
    # All conditions for the selector are bypassed with an explicit synthetic
    # order, so the test isolates the exit confirmation and protective stop.
    f['ma'][10].loc[:, sid] = 0
    f['ma'][10].loc[days[-5]:, sid] = 1000
    g, s = intervene(f, Config(), 'exit_ma10_confirm2', clean)
    assert pd.isna(g['ma'][10].at[days[-5], sid])
    assert g['ma'][10].at[days[-4], sid] == 1000
    rank = pd.DataFrame(np.nan, index=days, columns=f['ids'])
    rank.at[days[-6], sid] = 90
    g['low'] = g['low'].copy(); g['low'].at[days[-5], sid] = 1
    run = simulate(g, {'ranking':rank}, Config(), str(days[-5].date()), str(days[-1].date()))
    assert run['trades'][1]['reason'] == 'hard_stop'
    assert run['trades'][1]['date'] == run['trades'][0]['date']


def test_institutional_delay_does_not_skip_an_absent_session():
    days = pd.bdate_range('2024-01-01', periods=4)
    frame = pd.DataFrame({'date':[days[0], days[2]], 'stock_id':['1101']*2, 'trust_net':[10,20]})
    x = shift_institutional_availability(frame, days)
    assert x.date.tolist() == [days[1], days[3]]


def test_training_selection_ignores_future_results_and_can_choose_cash():
    rows = [dict(arm='a', status='completed', summary=dict(completed_trades=30,cagr=.1,max_drawdown=-.1), future_return=-.9),
            dict(arm='b', status='completed', summary=dict(completed_trades=30,cagr=.05,max_drawdown=-.2), future_return=10)]
    assert select_training(rows) == 'a'
    for row in rows: row['summary']['cagr'] = -.01
    assert select_training(rows) is None


def test_fixed_units_attribution_reconciles_commission_tax_and_slippage():
    days = pd.bdate_range('2024-01-01', periods=210)
    market = pd.Series(100., index=days)
    buy = dict(stock_id='1101', side='BUY', date=str(days[-3].date()), signal_date=str(days[-4].date()),
        adjusted_units=100., adjusted_price=100.1, notional=10010., fee=20.)
    proceeds = 100*109.89 - 20 - 100*109.89*.003
    sell = dict(stock_id='1101', side='SELL', date=str(days[-1].date()), adjusted_units=100.,
        adjusted_price=109.89, fee=20., tax=100*109.89*.003, net_pnl=proceeds-10030., reason='test')
    frame = trade_attribution([buy, sell], days, market)
    assert frame.same_units_gross_pnl.iloc[0] == pytest.approx(1000.)
    assert frame.net_pnl.iloc[0] == pytest.approx(sell['net_pnl'])
    assert frame.slippage.iloc[0] == pytest.approx(21.)


def test_overlap_detects_redundant_family_instead_of_counting_as_unique():
    days = pd.bdate_range('2024-01-01', periods=2)
    frame = lambda x: pd.DataFrame(x,index=days,columns=['1101'])
    flags = {name:frame(False) for name in FAMILIES}
    flags['volume_momentum'][:] = True
    flags['trust_consecutive'][:] = True
    rows, pair, meta = overlap_diagnostics(dict(ranking=frame(90),flags=flags), days[0],days[-1])
    assert sum(r['unique_stock_days'] for r in rows) == 0
    assert pair.at['volume_momentum','trust_consecutive'] == 2
    assert meta['membership_distribution']['2'] == 2


def test_paired_bootstrap_preserves_identical_paths_and_rejects_unpaired_dates():
    x = pd.DataFrame({'date':pd.bdate_range('2024-01-01',periods=60),'equity':np.arange(60)+100.})
    r = paired_block_interval(x,x)
    assert r['low'] == r['high'] == r['annualized_mean_difference'] == 0
    with pytest.raises(ValueError, match='identical dates'):
        paired_block_interval(x,x.iloc[1:])


def test_forward_study_uses_next_open_and_counts_unknown_outcomes_as_censored():
    bars,inst,money=features(100)
    f=prepare_features(bars,inst,money);days=f['days'];sid='1101'
    ranking=pd.DataFrame(np.nan,index=days,columns=f['ids'])
    ranking.at[days[70],sid]=90
    sig=dict(ranking=ranking,momentum=ranking,
        flags={name:ranking.notna() if name=='volume_momentum' else ranking.isna()&False for name in FAMILIES})
    rows,censor=forward_signal_study(f,sig,days[70],days[70],horizons=(5,))
    expected=f['close'].at[days[75],sid]*.999*(1-.001425-.003)/(f['open'].at[days[71],sid]*1.001*(1+.001425))-1
    assert rows[0]['mean_net_return']==pytest.approx(expected)
    assert censor[0]['retained']==1
    f['mark_close']=f['mark_close'].copy();f['mark_close'].at[days[73],sid]=np.nan
    rows,censor=forward_signal_study(f,sig,days[70],days[70],horizons=(5,))
    assert not rows and censor[0]['retained']==0
