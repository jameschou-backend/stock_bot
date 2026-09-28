import pandas as pd
import pytest

from skills.strict_tick_inputs import restore_halt_zeroes,StrictResidualReplay,StrictResidualBenchmark
from skills.replay_market_feeds import ReplayDataUnavailable
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture,ENTRY
from test_historical_selector_replay import identities
from test_residual_tick_replay import Ticks


def setup(missing=False,halt=True,nonzero=False):
    days,adjusted,args,kwargs=fixture(entries=[ENTRY],end=ENTRY+70)
    quotes=args[0].copy();day=days[ENTRY+52]
    mask=quotes.stock_id.eq('1101') & quotes.date.eq(day)
    raw=quotes.copy()
    if not nonzero:raw.loc[mask,['open','high','low','close','volume']]=0
    source=quotes.loc[~mask].copy() if missing else quotes
    exclusion=dict(stock_id='1101',market='TWSE',start=str(day.date()),end=str(days[ENTRY+53].date()),
        kind='information_halt',source_path='official.json',source_row=[0,'1101','test','day','8:00','next','8:00'])
    identity=identities(exclusions=[exclusion] if halt else [])
    return days,adjusted,args,kwargs,source,raw,identity


def test_known_zero_halt_restores_twenty_session_liquidity_without_fake_prices():
    days,adjusted,args,kwargs,source,raw,identity=setup(missing=True)
    fixed,evidence=restore_halt_zeroes(source,[raw],identity)
    row=fixed[fixed.stock_id.eq('1101') & fixed.date.eq(days[ENTRY+52])].iloc[0]
    assert row['volume']==row['close']==0 and len(evidence)==1
    engine=StrictResidualReplay(fixed,*args[1:],**kwargs,residual_policy='release',factor_mask=0,
        identity_report=identity,liquidity_identity=identity,exit_signals=ExitSignals(adjusted,days),ticks=Ticks())
    assert engine.volume20.at[days[ENTRY+63],'1101']==1_900_000
    assert engine.raw(days[ENTRY+52],'1101') is None


def test_unknown_gap_blocks_sell_instead_of_becoming_a_zero_fill():
    days,adjusted,args,kwargs,source,raw,identity=setup(missing=True,halt=False)
    fixed,evidence=restore_halt_zeroes(source,[raw],identity)
    assert not evidence and len(fixed)==len(source)
    engine=StrictResidualReplay(fixed,*args[1:],**kwargs,residual_policy='release',factor_mask=0,
        identity_report=identity,liquidity_identity=identity,exit_signals=ExitSignals(adjusted,days),ticks=Ticks())
    with pytest.raises(ReplayDataUnavailable,match='Unknown execution inputs.*adv20'):
        engine.run()


def test_partial_day_halt_and_unverified_missing_rows_are_never_filled():
    _,_,_,_,source,raw,identity=setup(missing=True)
    identity['trading_exclusions'][0]['source_row'][4]='10:00'
    fixed,evidence=restore_halt_zeroes(source,[raw],identity)
    assert not evidence and len(fixed)==len(source)


def test_official_halt_with_nonzero_source_is_a_conflict():
    _,_,_,_,source,raw,identity=setup(missing=True,nonzero=True)
    with pytest.raises(ReplayDataUnavailable,match='conflicts'):
        restore_halt_zeroes(source,[raw],identity)


def test_missing_candidate_liquidity_cannot_silently_shrink_ranking():
    days,adjusted,args,kwargs=fixture(entries=[ENTRY])
    q=args[0].loc[~(args[0].stock_id.eq('1101') & args[0].date.eq(days[ENTRY-2]))]
    engine=StrictResidualReplay(q,*args[1:],**kwargs,residual_policy='release',factor_mask=0,
        identity_report=identities(),liquidity_identity=identities(),exit_signals=ExitSignals(adjusted,days),ticks=Ticks())
    with pytest.raises(ReplayDataUnavailable,match='Unknown execution inputs'):
        engine.run()
