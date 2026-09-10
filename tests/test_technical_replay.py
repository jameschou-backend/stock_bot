"""Technical decisions, sizing and adds use synthetic cached providers only."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts.replay_million import audit
from skills.scenario_exit_replay import ExitSignals, ScenarioExitReplay


ENTRY = 130
SIZE = 220


class Feeds:
    def __init__(self, quotes, blocked=(), odd_volume=100_000):
        self.quotes = quotes.set_index(['date', 'stock_id'])
        self.days = sorted(quotes.date.unique())
        self.blocked = set(blocked)
        self.odd_volume = odd_volume

    def get_limits(self, sid):
        return {str(pd.Timestamp(day).date()): dict(upper=100_000.,
            lower=float(self.quotes.loc[(day, sid), 'close'])
            if (sid, pd.Timestamp(day)) in self.blocked else .001)
            for day in self.days}

    def get_odd(self, day, sid, market):
        price = float(self.quotes.loc[(pd.Timestamp(day), sid), 'close'])
        return dict(odd_shares=self.odd_volume, odd_last=price, odd_bid=price-.01,
            odd_ask=price+.01, bid_qty=10_000, ask_qty=10_000)


class Corporate:
    def __init__(self, events=None):
        self.events = events or {}
        self.prepared = []

    def prepare(self, sid):
        self.prepared.append(sid)

    def on_date(self, sid, day):
        return deepcopy(self.events.get((sid, day), []))


def fixture(*, stock=None, entries=None, end=ENTRY+8, start=ENTRY-1,
            mutate=None, corporate=None, blocked=(), odd_volume=100_000):
    days = pd.bdate_range('2021-01-04', periods=SIZE)
    adjusted = pd.DataFrame({'0050': 100.+np.arange(SIZE)*.1,
                            '1101': np.full(SIZE, 100.)}, index=days)
    if stock is not None:
        adjusted['1101'] = stock
    quotes = pd.DataFrame([dict(date=day, stock_id=sid, open=price,
        high=price+1, low=price-1, close=price, volume=2_000_000.)
        for day in days for sid, price in [('0050', 100.), ('1101', 50.)]])
    if mutate:
        mutate(quotes, days)
    events = [dict(event_id='entry-'+str(index), members=['1101'], priority=.1,
        signal_date=str(days[index-1].date()), entry_date=str(days[index].date()))
        for index in (entries or [])]
    companies = pd.DataFrame([dict(stock_id='1101', name='1101', market='TWSE')])
    args = (quotes, companies, days, events,
            Feeds(quotes, blocked, odd_volume), corporate or Corporate())
    kwargs = dict(start=str(days[start].date()), end=str(days[end].date()))
    return days, adjusted, args, kwargs

from skills.technical_replay import (MODES, TechnicalReplay, projected_loss,
                                    risk_quantity, transaction_cash)
from skills.technical_signals import TechnicalSignals


class Signals(ExitSignals):
    """Explicit prior-row policy inputs isolate account state-machine tests."""
    def __init__(self, adjusted, days, *, support=None, breakout=None, pattern=None,
                 unavailable=()):
        super().__init__(adjusted, days)
        self.support = support or {}
        self.breakout = breakout or {}
        self.pattern = pattern or {}
        self.unavailable = set(unavailable)

    def technical_context(self, index, sid):
        prior = index-1
        price = self.price(prior, sid)
        support = self.support.get(prior, 90.)
        available = prior not in self.unavailable and support is not None
        return dict(signal_index=prior, signal_date=str(self.days[prior].date()) if prior >= 0 else None,
            adjusted_close=price, support20=support, support_available=available,
            breakout20=self.breakout.get(prior, False),
            pattern_pass=self.pattern.get(prior, True), pattern_available=prior not in self.unavailable,
            contraction10=True, volume_expansion=True,
            diagnostics=[] if available else ['support_history_incomplete'])


def run(mode='support_risk2', *, support=None, breakout=None, pattern=None,
        unavailable=(), actual_features=False, **options):
    days, adjusted, args, kwargs = fixture(**options)
    signals = (TechnicalSignals(adjusted, args[0], days) if actual_features else
               Signals(adjusted, days, support=support, breakout=breakout,
                       pattern=pattern, unavailable=unavailable))
    replay = TechnicalReplay(*args, technical_signals=signals, mode=mode, **kwargs)
    account = replay.run()
    assert all(value for value in audit(account).values() if isinstance(value, bool))
    return replay, account, days


@pytest.mark.parametrize('stock_events', [False, True])
def test_control_exact_entire_account_and_exit_trails(stock_events):
    stock = np.full(SIZE, 100.); stock[ENTRY+2:] = 87.
    days, adjusted, args, kwargs = fixture(stock=stock, entries=[ENTRY] if stock_events else [])
    baseline = ScenarioExitReplay(*args, exit_signals=ExitSignals(adjusted, days), mode='loss12', **kwargs)
    expected = baseline.run()
    replay = TechnicalReplay(*args, technical_signals=ExitSignals(adjusted, days), mode='control', **kwargs)
    assert replay.run() == expected
    assert replay.exit_decisions == baseline.exit_decisions
    assert replay.exit_states == baseline.exit_states
    assert not replay.sizing_decisions and not replay.add_decisions and not replay.pattern_decisions


@pytest.mark.parametrize('mode', MODES[1:])
def test_variant_account_independently_reconciles_with_real_technical_features(mode):
    replay, account, _ = run(mode, actual_features=True, entries=[ENTRY])
    assert account['daily'][-1]['nav'] > 0
    assert account['settings']['horizon'] == 63


def test_support_break_strict_boundary_and_next_session_only():
    stock = np.full(SIZE, 100.); stock[ENTRY+1] = 90.; stock[ENTRY+2:] = 89.
    replay, account, days = run('support20', stock=stock, entries=[ENTRY])
    sales = [t for t in account['trades'] if t['stock_id']=='1101' and t['side']=='sell']
    assert sales and {t['reason'] for t in sales} == {'support20'}
    assert {t['date'] for t in sales} == {str(days[ENTRY+3].date())}
    assert {t['signal_date'] for t in sales} == {str(days[ENTRY+2].date())}
    assert not next(d for d in replay.exit_decisions if d['date']==str(days[ENTRY+2].date()))['exit']


def test_support_ratchet_never_loosens_and_missing_update_keeps_existing_floor():
    stock = np.full(SIZE, 100.); stock[ENTRY+3:] = 94.
    support = {ENTRY:95., ENTRY+1:80., ENTRY+2:None, ENTRY+3:80.}
    replay, _, days = run('support20', stock=stock, support=support, entries=[ENTRY])
    state = replay.exit_states['entry-'+str(ENTRY)]
    assert state['support_floor'] == 95.
    assert state['target_date'] == str(days[ENTRY+4].date())
    assert all(d['support_floor']==95. for d in replay.exit_decisions)


def test_invalid_raw_signal_cannot_update_floor_from_otherwise_present_support():
    replay, _, _ = run('support20', support={ENTRY:105.}, unavailable=[ENTRY], entries=[ENTRY])
    assert replay.exit_states['entry-'+str(ENTRY)]['support_floor']==90.
    assert not replay.exit_states['entry-'+str(ENTRY)]['trigger_reason']


@pytest.mark.parametrize('bad_support', [None, 100., 110.])
@pytest.mark.parametrize('mode', ['support20','support_risk2','support_risk2_add','support_risk2_pattern'])
def test_unusable_initial_support_rejects_before_stock_funding(mode, bad_support):
    replay, account, _ = run(mode, support={ENTRY-1:bad_support}, entries=[ENTRY])
    assert not account['cohorts']
    assert replay.sizing_decisions[0]['failure']=='initial_support_missing_or_not_below_price'
    assert not any(t['reason']=='fund_stock' for t in account['trades'])


def test_risk_only_does_not_require_support_and_keeps_original_loss12_exit():
    stock = np.full(SIZE, 100.); stock[ENTRY+2:] = 87.
    replay, account, _ = run('risk2', stock=stock, support={ENTRY-1:None}, entries=[ENTRY])
    assert account['cohorts']
    assert {t['reason'] for t in account['trades'] if t['side']=='sell' and t['stock_id']=='1101'}=={'loss12'}
    assert replay.sizing_decisions[0]['planned_stop']==88.


def test_support_instruction_latches_across_price_recovery_and_blocked_exit():
    stock = np.full(SIZE, 100.); stock[ENTRY+1] = 89.
    days = pd.bdate_range('2021-01-04', periods=SIZE)
    replay, account, _ = run('support20', stock=stock, entries=[ENTRY],
                            blocked=[('1101', days[ENTRY+2])])
    sales = [t for t in account['trades'] if t['side']=='sell' and t['stock_id']=='1101']
    assert {t['date'] for t in sales} == {str(days[ENTRY+3].date())}
    assert {t['signal_date'] for t in sales} == {str(days[ENTRY+1].date())}
    assert {t['reason'] for t in sales} == {'support20'}
    assert replay.exit_states['entry-'+str(ENTRY)]['target_date']==str(days[ENTRY+2].date())


@pytest.mark.parametrize('mode', ['risk2','support_risk2'])
def test_risk_size_is_cost_inclusive_and_at_most_two_percent_prior_nav(mode):
    replay, account, _ = run(mode, entries=[ENTRY])
    row = replay.sizing_decisions[0]
    assert 0 < row['planned_risk'] <= row['prior_nav']*.02
    assert row['requested_qty'] < 6000
    assert row['planned_risk']==projected_loss(row['previous_price'], row['raw_planned_stop'], row['requested_qty'], '1101')
    assert projected_loss(row['previous_price'], row['raw_planned_stop'], row['requested_qty']+1, '1101') > row['risk_cap']
    assert row['funding_budget'] <= row['capital_cap']
    assert account['cohorts'][0]['bought_qty']==row['filled_qty']


@pytest.mark.parametrize('existing', [0, 1, 199, 999, 1000, 1234])
def test_quantity_search_matches_exhaustive_exact_fee_boundaries(existing):
    for maximum in (999, 1000, 1001, 1999, 2000, 2001, 3000):
        risks = [projected_loss(50., 49.9, q, '1101', existing) for q in range(maximum+1)]
        for budget in (risks[-1], risks[-1]-.01, 300., 500.):
            feasible = [q for q,value in enumerate(risks) if value <= budget+1e-9]
            expected = max(feasible, default=0)
            assert risk_quantity(maximum, 50., 49.9, budget, '1101', existing)==expected


def test_projected_channel_fees_include_both_buy_and_sell_minimums():
    from skills.million_replay import costs
    qty=1001
    expected=sum(-costs(50.,part,'buy','1101')['cash_change']
                 -costs(49.,part,'sell','1101')['cash_change'] for part in (1000,1))
    assert projected_loss(50.,49.,qty,'1101')==pytest.approx(expected)
    assert transaction_cash(50.,0,'buy','1101')==0


def test_signal_future_perturbation_cannot_change_entry_size_or_same_day_exit():
    stock = np.full(SIZE, 100.)
    before, old, days = run('support_risk2', stock=stock, entries=[ENTRY], end=ENTRY)
    stock[ENTRY:] = 25.
    after, new, _ = run('support_risk2', stock=stock, entries=[ENTRY], end=ENTRY)
    assert before.sizing_decisions==after.sizing_decisions
    assert old==new


@pytest.mark.parametrize('passed', [False,None])
def test_pattern_rejects_before_funding_and_records_diagnostics(passed):
    replay, account, _ = run('support_risk2_pattern', pattern={ENTRY-1:passed}, entries=[ENTRY])
    assert not account['cohorts']
    assert replay.pattern_decisions[0]['pattern_pass'] is passed
    assert replay.sizing_decisions[0]['failure']==('pattern_not_confirmed' if passed is False else 'pattern_data_missing')
    assert not any(t['reason']=='fund_stock' for t in account['trades'])


def rising_options():
    stock = np.full(SIZE, 100.); stock[ENTRY+3:] = 112.
    def mutate(quotes, days):
        mask=quotes.stock_id.eq('1101') & quotes.date.ge(days[ENTRY+3])
        for key,value in [('open',56.),('high',57.),('low',55.),('close',56.)]:
            quotes.loc[mask,key]=value
    return dict(stock=stock, mutate=mutate, entries=[ENTRY],
                support={i:108. for i in range(ENTRY+3,SIZE)},
                breakout={i:True for i in range(ENTRY+3,SIZE)})


def test_add_uses_prior_gain_and_breakout_only_once_preserving_entry_and_due_clock():
    replay, account, days = run('support_risk2_add', **rising_options())
    additions = [t for t in account['trades'] if t['reason']=='pyramid_add']
    assert additions and {t['date'] for t in additions}=={str(days[ENTRY+4].date())}
    assert {t['signal_date'] for t in additions}=={str(days[ENTRY+3].date())}
    cohort=account['cohorts'][0]
    assert cohort['entry_date']==str(days[ENTRY].date()) and cohort['due_index']==ENTRY+63
    assert replay.exit_states[cohort['event_id']]['entry_price']==100.
    assert cohort['bought_qty']==sum(t['qty'] for t in account['trades'] if t['reason']=='leader_entry')
    assert len(replay.add_states)==1
    successful=[r for r in replay.add_decisions if r['filled_qty']]
    assert len(successful)==1
    row=successful[0]
    assert row['planned_risk']<=row['risk_cap']
    assert row['funding_budget']<=.1*row['prior_nav']
    assert (row['existing_qty']+row['pending_whole_qty']+row['requested_qty'])*row['previous_price']<=row['prior_nav']/3


def test_add_partial_positive_fill_consumes_single_chance():
    options=rising_options()
    def mutate(quotes, days):
        options['mutate'](quotes,days)
        quotes.loc[quotes.stock_id.eq('1101') & quotes.date.eq(days[ENTRY+4]),'volume']=100_000.
    replay, account, _=run('support_risk2_add', **{**options,'mutate':mutate}, odd_volume=1000)
    successes=[r for r in replay.add_decisions if r['filled_qty']]
    assert len(successes)==1 and 0<successes[0]['filled_qty']<successes[0]['requested_qty']
    assert len({t['date'] for t in account['trades'] if t['reason']=='pyramid_add'})==1


def test_add_zero_fill_retries_only_when_later_signal_still_qualifies():
    options=rising_options()
    def mutate(quotes, days):
        options['mutate'](quotes,days)
        quotes.loc[quotes.stock_id.eq('1101') & quotes.date.eq(days[ENTRY+4]),'volume']=0.
    replay, account, days=run('support_risk2_add', **{**options,'mutate':mutate})
    assert {t['date'] for t in account['trades'] if t['reason']=='pyramid_add'}=={str(days[ENTRY+5].date())}
    assert next(r for r in replay.add_decisions if r['date']==str(days[ENTRY+4].date()))['filled_qty']==0
    assert len(replay.add_states)==1


def test_add_never_averages_down_even_if_breakout_input_true():
    stock=np.full(SIZE,100.);stock[ENTRY+2:]=95.
    replay, account, _=run('support_risk2_add', stock=stock, entries=[ENTRY],
                          breakout={i:True for i in range(SIZE)})
    assert not any(t['reason']=='pyramid_add' for t in account['trades'])
    assert not replay.add_states


def test_pending_stock_rights_retain_exit_latch_until_delivery():
    days=pd.bdate_range('2021-01-04',periods=SIZE)
    stock=np.full(SIZE,100.);stock[ENTRY+2:]=89.
    ex,pay=str(days[ENTRY+1].date()),str(days[ENTRY+5].date())
    corporate=Corporate({('1101',ex):[dict(action_id='rights',stock_id='1101',
        kind='stock_dividend',shares_per_share=1.,pay_date=pay,fractional_cash_per_share=0.)]})
    replay,account,_=run('support20', stock=stock,entries=[ENTRY],corporate=corporate)
    sales=[t for t in account['trades'] if t['stock_id']=='1101' and t['side']=='sell']
    assert {t['date'] for t in sales}=={str(days[ENTRY+3].date()),pay}
    assert {t['reason'] for t in sales}=={'support20'}
    assert account['cohorts'][0]['exit_date']==pay
    assert not account['receivables']


def test_invalid_mode_and_missing_signal_interface_fail_explicitly():
    days, adjusted, args, kwargs=fixture()
    with pytest.raises(ValueError,match='Unknown technical'):
        TechnicalReplay(*args,technical_signals=ExitSignals(adjusted,days),mode='unknown',**kwargs)
    with pytest.raises(ValueError,match='require lagged'):
        TechnicalReplay(*args,technical_signals=ExitSignals(adjusted,days),mode='risk2',**kwargs)


def test_entry_risk_plans_on_known_ex_day_reference_without_execution_close_sizing():
    days=pd.bdate_range('2021-01-04',periods=SIZE)
    class SplitReference(Corporate):
        def reference_price(self,sid,day,last):
            return last/2 if sid=='1101' and day==str(days[ENTRY].date()) else last
    def mutate(quotes,dates):
        mask=quotes.stock_id.eq('1101') & quotes.date.ge(dates[ENTRY])
        for field,price in [('open',25.),('high',26.),('low',24.),('close',25.)]:
            quotes.loc[mask,field]=price
    replay,account,_=run('support_risk2',entries=[ENTRY],corporate=SplitReference(),mutate=mutate)
    row=replay.sizing_decisions[0]
    assert row['previous_price']==25.
    assert row['raw_planned_stop']==22.5
    assert row['planned_risk']<=row['risk_cap']
    assert account['cohorts']


def test_add_cap_counts_pending_whole_share_rights():
    options=rising_options()
    days=pd.bdate_range('2021-01-04',periods=SIZE)
    ex,pay=str(days[ENTRY+1].date()),str(days[ENTRY+20].date())
    corporate=Corporate({('1101',ex):[dict(action_id='rights',stock_id='1101',
        kind='stock_dividend',shares_per_share=.2,pay_date=pay,fractional_cash_per_share=0.)]})
    replay,_,_=run('support_risk2_add',corporate=corporate,**options)
    attempted=[r for r in replay.add_decisions if 'pending_whole_qty' in r]
    assert attempted and all(r['pending_whole_qty']>0 for r in attempted)
    for row in attempted:
        if row['requested_qty']:
            assert (row['existing_qty']+row['pending_whole_qty']+row['requested_qty'])*row['previous_price']<=row['prior_nav']/3


def test_original_new_entries_execute_before_same_day_additions():
    options=rising_options()
    days,adjusted,args,kwargs=fixture(**{k:v for k,v in options.items() if k not in ('support','breakout')})
    quotes,companies,_,events,_,corp=args
    second=quotes[quotes.stock_id.eq('1101')].copy();second['stock_id']='1102'
    quotes=pd.concat([quotes,second],ignore_index=True)
    adjusted['1102']=100.
    companies=pd.concat([companies,pd.DataFrame([dict(stock_id='1102',name='1102',market='TWSE')])],ignore_index=True)
    events.append(dict(event_id='second',members=['1102'],priority=.1,
        signal_date=str(days[ENTRY+3].date()),entry_date=str(days[ENTRY+4].date())))
    features=Signals(adjusted,days,support=options['support'],breakout=options['breakout'])
    # The second stock needs its own feasible initial support, independent of
    # the first holding's profitable trailing support.
    original=features.technical_context
    def context(index,sid):
        result=original(index,sid)
        if sid=='1102':result['support20']=90.
        return result
    features.technical_context=context
    replay=TechnicalReplay(quotes,companies,days,events,Feeds(quotes),corp,
                           technical_signals=features,mode='support_risk2_add',**kwargs)
    account=replay.run();audit(account)
    fresh=[t for t in account['trades'] if t['stock_id']=='1102' and t['reason']=='leader_entry']
    adds=[t for t in account['trades'] if t['reason']=='pyramid_add']
    assert fresh and adds
    assert max(t['sequence'] for t in fresh)<min(t['sequence'] for t in adds)


def test_exit_latch_prevents_add_even_when_gain_and_breakout_qualify():
    options=rising_options()
    options['support']={i:113. for i in range(ENTRY+3,SIZE)}
    days=pd.bdate_range('2021-01-04',periods=SIZE)
    replay,account,_=run('support_risk2_add',blocked=[('1101',days[ENTRY+4])],**options)
    assert replay.exit_states['entry-'+str(ENTRY)]['trigger_reason']=='support20'
    assert not any(t['reason']=='pyramid_add' for t in account['trades'])


@pytest.mark.parametrize('price,stop,existing,maximum,budget', [
    (1.123456,1.123446,0,199,42.),
    (1.00000499,.9999975,999,1999,111.),
    (32.891781,32.891581,999,2001,150.),
    (50.,49.99999,1999,2001,500.),
])
def test_fractional_cent_reference_search_uses_exact_cash_even_when_risk_oscillates(
        price,stop,existing,maximum,budget):
    expected=max((q for q in range(maximum+1)
        if projected_loss(price,stop,q,'1101',existing)<=budget+1e-9),default=0)
    assert risk_quantity(maximum,price,stop,budget,'1101',existing)==expected


def test_unfilled_entry_is_visible_in_sizing_audit():
    def mutate(quotes,days):
        quotes.loc[quotes.stock_id.eq('1101') & quotes.date.eq(days[ENTRY]),'volume']=0.
    replay,account,_=run('support_risk2',entries=[ENTRY],mutate=mutate)
    row=replay.sizing_decisions[0]
    assert row['requested_qty']>0 and row['filled_qty']==0
    assert row['failure']=='partial_or_unfilled_execution'
    assert not account['cohorts']
