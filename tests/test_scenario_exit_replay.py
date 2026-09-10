"""Exit-policy integration tests use synthetic prices and zero network providers."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from skills.million_replay import Replay
from skills.scenario_exit_replay import ExitSignals, ScenarioExitReplay


ENTRY = 130


class Feeds:
    def __init__(self, quotes, *, lower_limit_dates=(), odd_volume=100_000):
        self.quotes = quotes.set_index(['date', 'stock_id'])
        self.days = sorted(quotes.date.unique())
        self.lower_limit_dates = set(lower_limit_dates)
        self.odd_volume = odd_volume

    def get_limits(self, sid):
        return {str(pd.Timestamp(day).date()): dict(
            upper=100_000., lower=(float(self.quotes.loc[(day, sid), 'close'])
                                  if (sid, pd.Timestamp(day)) in self.lower_limit_dates else .001))
                for day in self.days}

    def get_odd(self, day, sid, market):
        price = float(self.quotes.loc[(pd.Timestamp(day), sid), 'close'])
        return dict(odd_shares=self.odd_volume, odd_last=price,
                    odd_bid=price-.01, odd_ask=price+.01, bid_qty=10_000, ask_qty=10_000)


class Corporate:
    def __init__(self, events=None):
        self.events = events or {}

    def prepare(self, sid):
        pass

    def on_date(self, sid, day):
        return deepcopy(self.events.get((sid, day), []))


def fixture(*, n=280, paths=None, events=None, mutate_quotes=None, corporate=None,
            lower_limit_dates=(), end_index=None, slots=3):
    days = pd.bdate_range('2021-01-04', periods=n)
    adjusted = pd.DataFrame({'0050':100.+np.arange(n)*.1, '1101':np.full(n,100.)}, index=days)
    for sid, values in (paths or {}).items():
        adjusted[sid] = values
    stocks = [sid for sid in adjusted.columns if sid != '0050']
    # Trading/accounting data is independent of the policy feature input. Tests
    # can remove a feature observation without manufacturing a trading halt.
    quotes = pd.DataFrame([dict(date=day, stock_id=sid, open=price, high=price+1,
                                low=price-1, close=price, volume=2_000_000)
                           for day in days for sid, price in [('0050',100.)]+[(sid,50.) for sid in stocks]])
    if mutate_quotes:
        mutate_quotes(quotes, days)
    companies = pd.DataFrame([dict(stock_id=sid, name=sid, market='TWSE') for sid in stocks])
    if events is None:
        events = [(ENTRY, '1101', 'first')]
    entries = [dict(event_id=identity, members=[sid], priority=.1,
                    signal_date=str(days[index-1].date()), entry_date=str(days[index].date()))
               for index,sid,identity in events]
    kwargs = dict(start=str(days[ENTRY-1].date()), end=str(days[-1 if end_index is None else end_index].date()), slots=slots)
    args = (quotes, companies, days, entries,
            Feeds(quotes, lower_limit_dates=lower_limit_dates), corporate or Corporate())
    return days, adjusted, args, kwargs


def run(mode, **options):
    days, adjusted, args, kwargs = fixture(**options)
    replay = ScenarioExitReplay(*args, exit_signals=ExitSignals(adjusted, days), mode=mode, **kwargs)
    account = replay.run()
    assert len(account['daily']) == len(days[(days>=pd.Timestamp(kwargs['start'])) & (days<=pd.Timestamp(kwargs['end']))])
    previous = account['settings']['initial_cash']
    for day in account['daily']:
        assert day['opening_nav'] == pytest.approx(previous)
        assert day['nav'] == pytest.approx(day['cash']+day['market_value']+day['receivable'])
        previous = day['nav']
    return replay, account, days


def stock_sales(account, sid='1101', event_id=None):
    return [row for row in account['trades'] if row['stock_id']==sid and row['side']=='sell'
            and (event_id is None or row['event_id']==event_id)]


def test_fixed63_wrapper_reproduces_the_entire_original_account_exactly():
    days, adjusted, args, kwargs = fixture()
    original = Replay(*args, **kwargs).run()
    wrapper = ScenarioExitReplay(*args, exit_signals=ExitSignals(adjusted, days), mode='fixed63', **kwargs)
    result = wrapper.run()
    assert result == original
    assert set(result) == set(original)
    assert result['cohorts'][0]['exit_date'] == str(days[ENTRY+63].date())


def test_overflowed_relative_return_cannot_authorize_strong_trend_extension():
    days, adjusted, _, _ = fixture()
    adjusted['1101'] = 100. + np.arange(len(days))
    adjusted.loc[days[ENTRY-20], '1101'] = 1e-308
    with np.errstate(over='ignore', invalid='ignore'):
        signals = ExitSignals(adjusted, days)
    assert pd.isna(signals.relative20.loc[days[ENTRY], '1101'])
    assert not signals.strong.loc[days[ENTRY], '1101']


def test_loss_signal_uses_yesterday_and_does_not_fill_at_the_stop_threshold():
    path = np.full(280,100.); path[ENTRY+2:] = 87.
    replay, account, days = run('loss12', paths={'1101':path})
    sales = stock_sales(account)
    assert {t['date'] for t in sales} == {str(days[ENTRY+3].date())}
    assert all(t['reference_price']==50. for t in sales)
    state = replay.exit_states['first']
    assert state['entry_index']==ENTRY and state['entry_price']==100.
    assert state['trigger_reason']=='loss12'
    assert state['signal_date']==str(days[ENTRY+2].date())
    assert state['target_date']==str(days[ENTRY+3].date())
    assert state['target_index']==ENTRY+3


@pytest.mark.parametrize('blocked_by', ['lower_limit','partial_volume'])
def test_trigger_latches_across_an_unfilled_or_partial_exit_and_a_rebound(blocked_by):
    path = np.full(280,100.); path[ENTRY+2] = 87.; path[ENTRY+3:] = 140.
    days = pd.bdate_range('2021-01-04',periods=280)
    def mutation(quotes, dates):
        if blocked_by=='partial_volume':
            quotes.loc[quotes.stock_id.eq('1101') & quotes.date.eq(dates[ENTRY+3]), 'volume'] = 100_000
    blocked = [('1101',days[ENTRY+3])] if blocked_by=='lower_limit' else []
    replay, account, days = run('loss12', paths={'1101':path}, mutate_quotes=mutation, lower_limit_dates=blocked)
    sales = stock_sales(account)
    assert sales[-1]['date']==str(days[ENTRY+4].date())
    if blocked_by=='partial_volume':
        assert sales[0]['date']==str(days[ENTRY+3].date())
        assert sum(t['qty'] for t in sales if t['date']==str(days[ENTRY+3].date())) < account['cohorts'][0]['bought_qty']
    else:
        assert not any(t['date']==str(days[ENTRY+3].date()) for t in sales)
    assert sum(t['qty'] for t in sales)==account['cohorts'][0]['bought_qty']
    state = replay.exit_states['first']
    assert state['trigger_reason']=='loss12' and state['target_index']==ENTRY+3


def test_trailing_arm_retains_historical_peak_after_profit_falls_below_twenty_percent():
    path = np.full(280,100.)
    path[ENTRY+1] = 121.; path[ENTRY+2] = 115.; path[ENTRY+3:] = 106.
    replay, account, days = run('trail20_12', paths={'1101':path})
    assert {t['date'] for t in stock_sales(account)} == {str(days[ENTRY+4].date())}
    assert replay.exit_states['first']['peak_price']==121.
    assert replay.exit_states['first']['trigger_reason']=='trailing12'


def test_two_weak_closes_must_both_belong_to_the_actual_holding_period():
    path = np.full(280,100.); path[ENTRY-1] = 95.; path[ENTRY] = 90.; path[ENTRY+1:] = 85.
    replay, account, days = run('weak20', paths={'1101':path})
    assert {t['date'] for t in stock_sales(account)} == {str(days[ENTRY+2].date())}
    assert replay.exit_states['first']['trigger_reason']=='trend_break'


def test_missing_adjacent_close_does_not_fill_or_skip_a_hole_in_the_ma20_window():
    path = np.full(280,100.); path[ENTRY] = 95.; path[ENTRY+1] = np.nan; path[ENTRY+2:] = 90.
    replay, account, _ = run('weak20', paths={'1101':path}, end_index=ENTRY+10)
    assert not stock_sales(account)
    assert replay.exit_states['first']['trigger_reason'] is None


def test_missing_entry_close_is_not_replaced_by_the_first_later_observation():
    path = np.full(280,100.); path[ENTRY] = np.nan; path[ENTRY+2:] = 70.
    replay, account, days = run('loss12', paths={'1101':path})
    assert replay.exit_states['first']['entry_price'] is None
    assert {t['date'] for t in stock_sales(account)} == {str(days[ENTRY+63].date())}


def test_strong_trend_extends_daily_but_never_past_original_entry_plus_126():
    path = 100.+np.arange(280)*.5
    replay, account, days = run('trend126', paths={'1101':path})
    assert {t['date'] for t in stock_sales(account)} == {str(days[ENTRY+126].date())}
    assert replay.exit_states['first']['trigger_reason']=='hard_time126'
    assert replay.exit_states['first']['target_index']==ENTRY+126
    # The original schedule remains a stable reference in the old account fields.
    assert account['cohorts'][0]['due_index']==ENTRY+63
    assert account['cohorts'][0]['due_date']==str(days[ENTRY+63].date())


@pytest.mark.parametrize('missing', ['stock','market'])
def test_unknown_signal_at_original_expiry_cannot_authorize_an_extension(missing):
    stock=100.+np.arange(280)*.5; market=100.+np.arange(280)*.1
    (stock if missing=='stock' else market)[ENTRY+62] = np.nan
    replay, account, days = run('trend126', paths={'1101':stock,'0050':market})
    assert {t['date'] for t in stock_sales(account)} == {str(days[ENTRY+63].date())}
    assert replay.exit_states['first']['trigger_reason']=='time63_weak'


def test_extended_position_latches_when_strong_trend_fails_and_does_not_wait_until_126():
    path=100.+np.arange(280)*.5
    path[ENTRY+69:] = 100.
    replay, account, days = run('trend126', paths={'1101':path})
    assert {t['date'] for t in stock_sales(account)} == {str(days[ENTRY+70].date())}
    assert replay.exit_states['first']['trigger_reason']=='time63_weak'


def test_cash_dividend_is_locked_before_exit_and_paid_after_the_position_is_gone():
    path=np.full(280,100.); path[ENTRY+2:] = 87.
    days=pd.bdate_range('2021-01-04',periods=280)
    ex,pay=str(days[ENTRY+3].date()),str(days[ENTRY+7].date())
    corporate=Corporate({('1101',ex):[dict(stock_id='1101',action_id='cash',kind='cash_dividend',
                                         cash_per_share=1.,pay_date=pay)]})
    _,account,_=run('loss12',paths={'1101':path},corporate=corporate)
    assert {t['date'] for t in stock_sales(account)}=={ex}
    entitlement=next(a for a in account['corporate_actions'] if a['kind']=='cash_dividend')
    payment=next(c for c in account['cash_ledger'] if c['kind']=='dividend_payment')
    assert entitlement['entitled_qty']==account['cohorts'][0]['bought_qty']
    assert payment['date']==pay and payment['cash_change']==entitlement['entitlement_value']


def test_late_share_delivery_inherits_the_latched_early_exit_despite_price_recovery():
    path=np.full(280,100.); path[ENTRY+2]=87.; path[ENTRY+3:]=140.
    days=pd.bdate_range('2021-01-04',periods=280)
    ex,delivery=str(days[ENTRY+2].date()),str(days[ENTRY+8].date())
    corporate=Corporate({('1101',ex):[dict(stock_id='1101',action_id='shares',kind='stock_dividend',
                                         shares_per_share=1.,pay_date=delivery)]})
    replay,account,_=run('loss12',paths={'1101':path},corporate=corporate)
    sales=stock_sales(account)
    assert {t['date'] for t in sales}=={str(days[ENTRY+3].date()),delivery}
    assert sum(t['qty'] for t in sales)==account['cohorts'][0]['bought_qty']*2
    assert account['cohorts'][0]['exit_date']==delivery
    assert replay.exit_states['first']['target_index']==ENTRY+3
    assert not account['receivables']


def test_exit_releases_a_slot_before_the_same_sessions_new_frozen_candidate():
    path=np.full(280,100.); path[ENTRY+2:]=87.
    events=[(ENTRY,'1101','first'),(ENTRY+3,'1102','second')]
    _,account,days=run('loss12',paths={'1101':path,'1102':np.full(280,100.)},events=events,slots=1)
    assert [c['event_id'] for c in account['cohorts']]==['first','second']
    assert account['cohorts'][0]['exit_date']==account['cohorts'][1]['entry_date']==str(days[ENTRY+3].date())


def test_untradable_share_receivable_keeps_slot_until_delivery_and_latched_sale():
    path=np.full(280,100.); path[ENTRY+2]=87.; path[ENTRY+3:]=140.
    days=pd.bdate_range('2021-01-04',periods=280)
    ex,delivery=str(days[ENTRY+2].date()),str(days[ENTRY+8].date())
    corporate=Corporate({('1101',ex):[dict(stock_id='1101',action_id='shares',kind='stock_dividend',
                                         shares_per_share=1.,pay_date=delivery)]})
    events=[(ENTRY,'1101','first'),(ENTRY+3,'1102','blocked'),(ENTRY+8,'1102','after_delivery')]
    _,account,_=run('loss12',paths={'1101':path,'1102':np.full(280,100.)},events=events,slots=1,corporate=corporate)
    assert [c['event_id'] for c in account['cohorts']]==['first','after_delivery']
    assert any(o['event_id']=='blocked' and o['failure']=='slots_full' for o in account['orders'])
    assert account['cohorts'][0]['exit_date']==account['cohorts'][1]['entry_date']==delivery


def test_new_entry_in_same_ticker_resets_anchor_peak_and_exit_latch():
    path=np.full(280,100.); path[ENTRY+1]=150.; path[ENTRY+2:ENTRY+5]=87.; path[ENTRY+5:]=200.
    events=[(ENTRY,'1101','first'),(ENTRY+5,'1101','second')]
    replay,account,days=run('loss12',paths={'1101':path},events=events,end_index=ENTRY+15)
    assert replay.exit_states['first']['trigger_reason']=='loss12'
    second=replay.exit_states['second']
    assert second['entry_index']==ENTRY+5 and second['entry_price']==200.
    assert second['peak_price']==200. and second['trigger_reason'] is None
    assert not stock_sales(account,event_id='second')
    assert account['cohorts'][1]['exit_date'] is None


def test_future_adjusted_and_execution_prices_do_not_change_prefix_decisions_or_trades():
    stock=100.+np.arange(280)*.5
    first,account,days=run('adaptive',paths={'1101':stock})
    cutoff=ENTRY+10
    changed=stock.copy(); changed[cutoff+1:]=np.linspace(1000.,1.,280-cutoff-1)
    def prices(quotes,dates):
        mask=quotes.stock_id.eq('1101') & quotes.date.gt(dates[cutoff])
        quotes.loc[mask,['open','close']]=180.
        quotes.loc[mask,'high']=181.; quotes.loc[mask,'low']=179.
    second,altered,_=run('adaptive',paths={'1101':changed},mutate_quotes=prices)
    through=str(days[cutoff].date())
    for name in ('daily','trades','orders','holdings','cash_ledger','corporate_actions'):
        assert [r for r in account[name] if r['date']<=through]==[r for r in altered[name] if r['date']<=through]
    assert [r for r in first.exit_decisions if r['date']<=through]==[r for r in second.exit_decisions if r['date']<=through]


def test_truncating_future_rows_does_not_change_completed_prefix_decisions():
    stock=100.+np.arange(280)*.5
    full,account,days=run('adaptive',paths={'1101':stock})
    end=ENTRY+10
    short,short_account,_=run('adaptive',n=end+1,paths={'1101':stock[:end+1]})
    through=str(days[end].date())
    assert short_account['daily']==[r for r in account['daily'] if r['date']<=through]
    assert short_account['trades']==[r for r in account['trades'] if r['date']<=through]
    assert short.exit_decisions==[r for r in full.exit_decisions if r['date']<=through]


@pytest.mark.parametrize('missing_between', [False,True])
def test_market_exit_requires_adjacent_observed_off_sessions_and_relative_weakness(missing_between):
    stock=np.full(280,100.); stock[ENTRY:]=70.
    market=100.+np.arange(280)*.1; market[ENTRY:]=90.
    if missing_between:
        market[ENTRY+1]=np.nan
    replay,account,days=run('market_weak',paths={'1101':stock,'0050':market})
    expected=ENTRY+(4 if missing_between else 2)
    assert {t['date'] for t in stock_sales(account)}=={str(days[expected].date())}
    assert replay.exit_states['first']['trigger_reason']=='market_weak'


def test_market_off_cannot_exit_an_individually_outperforming_stock():
    stock=np.full(280,100.)
    market=100.+np.arange(280)*.1; market[ENTRY:]=90.
    replay,account,_=run('market_weak',paths={'1101':stock,'0050':market},end_index=ENTRY+10)
    assert not stock_sales(account)
    assert replay.exit_states['first']['trigger_reason'] is None


def test_failed_entry_never_creates_a_policy_position_or_peak_state():
    path=np.full(280,100.); path[ENTRY+1:]=50.
    def mutation(quotes,days):
        quotes.loc[quotes.stock_id.eq('1101') & quotes.date.eq(days[ENTRY]),'volume']=0
    replay,account,_=run('adaptive',paths={'1101':path},mutate_quotes=mutation)
    assert not account['cohorts'] and not replay.exit_states and not replay.exit_decisions


@pytest.mark.parametrize('bad', [0.,-1.,float('inf'),float('nan')])
def test_invalid_previous_adjusted_price_cannot_trigger_a_synthetic_stop(bad):
    path=np.full(280,100.); path[ENTRY+1]=bad
    replay,account,_=run('loss12',paths={'1101':path},end_index=ENTRY+5)
    assert not stock_sales(account)
    assert replay.exit_states['first']['trigger_reason'] is None
    decision=next(r for r in replay.exit_decisions if r['held_sessions']==2)
    assert decision['has_signal'] is False and decision['signal_close'] is None


def test_signal_features_use_twenty_market_rows_and_full_ma_windows_without_filling():
    days,adjusted,_,_=fixture()
    adjusted.loc[days[ENTRY-10],'1101']=np.nan
    features=ExitSignals(adjusted,days)
    assert pd.isna(features.ma20.at[days[ENTRY],'1101'])
    # Return endpoints remain exactly 20 market sessions apart; a missing
    # intermediate session cannot silently change the observation horizon.
    expected=(adjusted.at[days[ENTRY],'1101']/adjusted.at[days[ENTRY-20],'1101']-1
              -(adjusted.at[days[ENTRY],'0050']/adjusted.at[days[ENTRY-20],'0050']-1))
    assert features.relative20.at[days[ENTRY],'1101']==pytest.approx(expected)
    adjusted.loc[days[ENTRY-20],'1101']=np.nan
    assert pd.isna(ExitSignals(adjusted,days).relative20.at[days[ENTRY],'1101'])


def test_misaligned_exit_calendar_is_rejected_before_any_account_run():
    days,adjusted,args,kwargs=fixture()
    wrong_days=days.delete(10)
    with pytest.raises(ValueError,match='calendar'):
        ScenarioExitReplay(*args,exit_signals=ExitSignals(adjusted.reindex(wrong_days),wrong_days),mode='loss12',**kwargs)


def test_missing_candidate_feature_column_is_rejected_before_any_account_run():
    days,adjusted,args,kwargs=fixture()
    with pytest.raises(ValueError,match='Candidate stock missing'):
        ScenarioExitReplay(*args,exit_signals=ExitSignals(adjusted[['0050']],days),mode='loss12',**kwargs)


def test_126_is_the_latest_exit_instruction_not_a_fabricated_forced_liquidation():
    path=100.+np.arange(280)*.5
    days=pd.bdate_range('2021-01-04',periods=280)
    replay,account,_=run('trend126',paths={'1101':path},
                         lower_limit_dates=[('1101',days[ENTRY+126])])
    assert {t['date'] for t in stock_sales(account)}=={str(days[ENTRY+127].date())}
    assert replay.exit_states['first']['target_index']==ENTRY+126
    assert replay.exit_states['first']['trigger_reason']=='hard_time126'
    held=[h for h in account['holdings'] if h['stock_id']=='1101' and h['date']==str(days[ENTRY+126].date())]
    assert held and held[0]['qty']==account['cohorts'][0]['bought_qty']
