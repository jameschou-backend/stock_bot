from copy import deepcopy

import pandas as pd
import pytest

from skills.diffusion_signals import build_diffusion as original
from skills.historical_diffusion_signals import build_diffusion
from skills.historical_selector_replay import eligibility_matrix, HistoricalBoardReplay
from skills.board_only_verified_replay import BoardOnlyVerifiedReplay
from skills.scenario_exit_replay import ExitSignals
from skills.replay_market_feeds import ReplayDataUnavailable
from test_diffusion_signals import market
from test_cash_allocation_replay import fixture, ENTRY


def identities(ids=('1101',), exclusions=()):
    return dict(coverage_start='2000-01-01', coverage_end='2030-01-01',
        episodes=[dict(stock_id=sid, start='2000-01-01', end=None, market='TWSE',
                       category='ETF' if sid == '0050' else '股票') for sid in (*ids, '0050')],
        trading_exclusions=list(exclusions))


def test_neutral_selector_preserves_every_decision():
    args = market()
    old = original(*args, start='2022-01-03', signal_end='2022-01-31')
    new = build_diffusion(*args, start='2022-01-03', signal_end='2022-01-31')
    old['stats'].pop('seconds'); new['stats'].pop('seconds')
    assert new == old


def test_dated_identity_masks_halt_delisting_category_and_market():
    days = pd.date_range('2022-01-03', periods=5)
    companies = pd.DataFrame({'stock_id': ['1101', '1102', '1103']})
    report = identities(('1101', '1102', '1103'), [dict(stock_id='1101', market='TWSE',
        start='2022-01-04', end='2022-01-06', kind='information_halt')])
    report['episodes'][1]['end'] = '2022-01-05'
    report['episodes'][2]['category'] = '創新板'
    mask = eligibility_matrix(report, companies, days)
    assert mask['1101'].tolist() == [True, False, False, True, True]
    assert mask['1102'].tolist() == [True, True, False, False, False]
    assert not mask['1103'].any()
    assert mask['0050'].all()
    report['trading_exclusions'][0]['market'] = 'TPEx'
    assert eligibility_matrix(report, companies, days)['1101'].all()


def test_missing_identity_does_not_silently_shrink_denominator():
    with pytest.raises(ValueError, match='unresolved: 1102'):
        eligibility_matrix(identities(), pd.DataFrame({'stock_id': ['1102']}), pd.date_range('2022-01-03', periods=2))


def test_missing_quote_company_still_counts_in_expected_coverage():
    args = list(market())
    args[-1] = pd.concat([args[-1], pd.DataFrame([{'stock_id': '1105', 'listed_date': pd.Timestamp('2000-01-01')}])])
    mask = eligibility_matrix(identities(('1101', '1102', '1103', '1104', '1105')), args[-1], args[0].index)
    result = build_diffusion(*args, start='2022-01-03', signal_end='2022-01-31', eligibility=mask)
    assert result['stats']['turnover_coverage_under_95_sessions'] == len(args[0])
    assert any(c['reason'] == 'missing_turnover_share_window' for e in result['events'] for c in e['confirmation_checks'])


def test_halted_leader_cannot_emit_signal_even_with_a_source_quote():
    args = market()
    mask = eligibility_matrix(identities(('1101', '1102', '1103', '1104'), [dict(stock_id='1101',
        market='TWSE', start='2022-01-05', end='2022-01-06', kind='information_halt')]), args[-1], args[0].index)
    result = build_diffusion(*args, start='2022-01-03', signal_end='2022-01-31', eligibility=mask)
    assert not result['entries']['leader_now']


def test_future_quote_and_eligibility_mutation_cannot_change_prior_signal():
    args = market()
    mask = eligibility_matrix(identities(('1101', '1102', '1103', '1104')), args[-1], args[0].index)
    before = build_diffusion(*args, start='2022-01-03', signal_end='2022-01-31', eligibility=mask)
    changed = tuple(frame.copy() for frame in args)
    for frame in changed[:4]:
        frame.loc['2022-01-07':] *= 3
    mask.loc['2022-01-07':, '1101'] = False
    after = build_diffusion(*changed, start='2022-01-03', signal_end='2022-01-31', eligibility=mask)
    prior = lambda r: [e for e in r['entries']['leader_now'] if e['entry_date'] <= '2022-01-06']
    assert prior(after) == prior(before) and prior(after)
    assert after['groups'][0]['clusters'] == before['groups'][0]['clusters']


@pytest.mark.parametrize('stress', ['control', 'combined'])
def test_execution_guard_preserves_neutral_full_account(stress):
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+68)
    kwargs.update(stress_mode=stress, exit_signals=ExitSignals(adjusted, days))
    old = BoardOnlyVerifiedReplay(*args, **kwargs).run()
    new = HistoricalBoardReplay(*args, **kwargs, identity_report=identities()).run()
    assert new == old


def test_known_halt_never_executes_and_preserves_existing_holdings():
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+68)
    halt = str(days[ENTRY+63].date())
    resume = str(days[ENTRY+65].date())
    # A halt across the scheduled sale delays execution, retaining ownership.
    report = identities(exclusions=[dict(stock_id='1101', market='TWSE', start=halt,
        end=resume, kind='information_halt')])
    quotes = args[0].copy()
    quotes = quotes.loc[~(quotes.stock_id.eq('1101') & quotes.date.between(halt, str(days[ENTRY+64].date()))) ]
    args = (quotes, *args[1:])
    engine = HistoricalBoardReplay(*args, **kwargs, exit_signals=ExitSignals(adjusted, days), identity_report=report)
    result = engine.run()
    assert not any(t['date'] in (halt, str(days[ENTRY+64].date())) for t in result['trades'])
    assert any(h['date'] == halt and h['qty'] > 0 for h in result['holdings'])
    assert engine.identity_decisions


def test_delisted_held_position_blocks_instead_of_creating_liquidation():
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+5)
    report = identities()
    report['episodes'][0]['end'] = str(days[ENTRY+2].date())
    with pytest.raises(ReplayDataUnavailable, match='Held stock lacks dated settlement'):
        HistoricalBoardReplay(*args, **kwargs, exit_signals=ExitSignals(adjusted, days), identity_report=report).run()


def test_unmasked_halt_quote_is_an_error_not_a_fill():
    days, adjusted, args, kwargs = fixture(entries=[ENTRY])
    report = identities(exclusions=[dict(stock_id='1101', market='TWSE', start=str(days[ENTRY].date()),
        end=str(days[ENTRY+1].date()), kind='information_halt')])
    with pytest.raises(ValueError, match='still has an executable quote'):
        HistoricalBoardReplay(*args, **kwargs, exit_signals=ExitSignals(adjusted, days), identity_report=report).run()
