"""Synthetic corporate-rights checks; no remote datasets or historical returns."""
from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from skills import replay_corporate_actions as module
from skills.million_replay import Replay, UnresolvedAction


EX = '2023-09-08'
PAY = '2023-10-13'
# Issuer 112 annual report, printed p68 / PDF p72. This is the sum of
# the two announced components, not the exchange's last-decimal rounding.
EXACT_STOCK_RATE = Decimal('99.16825873') / Decimal(1000)
EVENT_COLUMNS = ['stock_id', 'event_date', 'source', 'event_type', 'payload_json',
                 'opening_ref', 'ref_price', 'cash_increase_suspected']


@pytest.fixture(autouse=True)
def no_remote_fetch(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Corporate action tests must never request remote data')
    monkeypatch.setattr(module, 'fetch_dataset', forbidden)


def event(sid='4123', day=EX, **changes):
    result = dict(stock_id=sid, event_date=day, source='ex_rights', event_type='權',
                  payload_json=json.dumps({}), opening_ref=48., ref_price=48.,
                  cash_increase_suspected=False)
    result.update(changes)
    return result


def dividend(sid='4123', **changes):
    result = dict(stock_id=sid, date='2023-09-16', AnnouncementDate='2023-08-24',
                  CashExDividendTradingDate=EX, CashEarningsDistribution=.8,
                  CashStatutorySurplus=.2, CashDividendPaymentDate=PAY,
                  StockExDividendTradingDate='', StockEarningsDistribution=0.,
                  StockStatutorySurplus=0., TotalNumberOfCashCapitalIncrease=0.)
    result.update(changes)
    return result


def cached(tmp_path, rows, events=(), *, sid='4123', overrides=None):
    pd.DataFrame(rows).to_parquet(tmp_path / f'{sid}.parquet', index=False)
    return module.CorporateActions(pd.DataFrame(events, columns=EVENT_COLUMNS), tmp_path,
                                   offline=True, overrides=overrides)


def bare_account(corporate, *, sid='4123', qty=1234):
    """Exercise rights settlement alone, without running a strategy or returns."""
    engine = Replay.__new__(Replay)
    engine.corporate = corporate
    engine.cash = 1000.
    engine.cash_ledger = []
    engine.holdings = {sid: {'qty': qty, 'event_id': 'synthetic', 'due_index': 100}}
    engine.marks = {sid: {'price': 50.}}
    # Match the real constructor's date-indexed raw quote field. The mark is
    # separate: a historical valuation mark cannot supply ex-date evidence.
    engine.fields = {'close': pd.DataFrame({sid: [50., 50.]},
                                          index=pd.DatetimeIndex([EX, PAY]))}
    engine.receivables = []
    engine.actions = []
    engine.cohorts = [{'event_id': 'synthetic', 'due_index': 100}]
    return engine


def test_cash_uses_ex_date_not_record_or_announcement_date_and_keeps_payment(tmp_path):
    actions = cached(tmp_path, [dividend(), dividend()])
    assert actions.on_date('4123', '2023-08-24') == []
    assert actions.on_date('4123', '2023-09-16') == []
    rows = actions.on_date('4123', EX)
    assert len(rows) == 1  # Identical policy revisions do not duplicate a right.
    assert rows[0]['cash_per_share'] == 1.
    assert rows[0]['pay_date'] == PAY
    assert rows[0]['announcement_date'] == '2023-08-24'
    assert actions.manifest()['requests'] == 0


def test_cash_right_survives_sale_and_is_not_spendable_before_payment(tmp_path):
    engine = bare_account(cached(tmp_path, [dividend()]))
    engine.corporate_day(pd.Timestamp(EX))
    assert engine.cash == 1000.
    assert engine.receivables[0]['amount'] == 1234.
    engine.holdings.clear()  # Sale does not erase an already vested right.
    engine.corporate_day(pd.Timestamp('2023-10-12'))
    assert engine.cash == 1000.
    engine.corporate_day(pd.Timestamp(PAY))
    assert engine.cash == 2234.
    assert engine.receivables == []


def test_4123_cash_dividend_floors_whole_ntd_at_entitlement_and_payment(tmp_path):
    # The issuer specifies rounding the total dividend down to whole NTD.
    # 1234 shares * 0.99168258 = 1223.73670372, hence NT$1223, not 1223.74.
    actions = cached(tmp_path, [dividend(CashEarningsDistribution=.99168258,
        CashStatutorySurplus=0.)], overrides={f'4123-{EX}': {'cash_rounding': 'floor_ntd'}})
    action, = actions.on_date('4123', EX)
    assert action['cash_rounding'] == 'floor_ntd'
    engine = bare_account(actions)
    income = engine.corporate_day(pd.Timestamp(EX))
    assert income == 1223.
    assert engine.receivables[0]['amount'] == 1223.
    assert engine.actions[0]['entitlement_value'] == 1223.
    assert engine.cash == 1000.
    engine.holdings.clear()
    engine.corporate_day(pd.Timestamp(PAY))
    assert engine.cash == 2223.
    assert engine.cash_ledger[0]['cash_change'] == 1223.
    assert engine.receivables == []


def test_missing_payment_is_an_unpaid_receivable_not_immediate_cash(tmp_path):
    engine = bare_account(cached(tmp_path, [dividend(CashDividendPaymentDate='')]))
    engine.corporate_day(pd.Timestamp(EX))
    engine.corporate_day(pd.Timestamp('2023-12-29'))
    assert engine.cash == 1000.
    assert engine.receivables[0]['pay_date'] is None
    assert engine.receivables[0]['amount'] == 1234.


@pytest.mark.parametrize('kind', ['cash', 'stock'])
def test_missing_ex_date_quote_blocks_new_rights_despite_existing_mark(tmp_path, kind):
    actions = (cached(tmp_path, [dividend()]) if kind == 'cash' else
               cached(tmp_path, [stock_policy()], [stock_event()], overrides=stock_override()))
    engine = bare_account(actions)
    engine.fields['close'].at[pd.Timestamp(EX), '4123'] = np.nan
    assert engine.marks['4123']['price'] == 50.
    with pytest.raises(UnresolvedAction, match='Ex-date valuation price missing'):
        engine.corporate_day(pd.Timestamp(EX))
    assert engine.receivables == []
    assert engine.cash == 1000.
    assert engine.holdings['4123']['qty'] == 1234


@pytest.mark.parametrize('change', [dict(CashDividendPaymentDate='2023-09-07'),
                                    dict(CashEarningsDistribution=1.1)])
def test_conflicting_policy_revision_or_payment_before_ex_date_fails(tmp_path, change):
    with pytest.raises(ValueError):
        cached(tmp_path, [dividend(), dividend(**change)]).prepare('4123')


@pytest.mark.parametrize('column,value', [
    ('CashExDividendTradingDate', '2023-02-30'),
    ('CashExDividendTradingDate', 'NaT'),
    ('CashExDividendTradingDate', '2023-09-08T10:00:00'),
    ('CashDividendPaymentDate', 'not-a-date'),
    ('CashDividendPaymentDate', 'NaT'),
    ('CashDividendPaymentDate', '2023-10-13T00:00:00+08:00'),
])
def test_dividend_dates_must_be_valid_date_only_values(tmp_path, column, value):
    with pytest.raises(ValueError):
        cached(tmp_path, [dividend(**{column: value})]).prepare('4123')


@pytest.mark.parametrize('value', [-1., float('inf'), float('-inf')])
def test_invalid_distribution_amount_is_not_silently_omitted(tmp_path, value):
    with pytest.raises(ValueError):
        cached(tmp_path, [dividend(CashEarningsDistribution=value)]).prepare('4123')


def test_missing_optional_cash_component_does_not_erase_known_cash(tmp_path):
    actions = cached(tmp_path, [dividend(CashStatutorySurplus=np.nan)])
    rows = actions.on_date('4123', EX)
    assert len(rows) == 1
    assert rows[0]['cash_per_share'] == .8


def test_official_cash_disagreement_is_explicit_unresolved_action(tmp_path):
    actions = cached(tmp_path, [dividend()], [event(
        event_type='息', payload_json=json.dumps({'cash_dividend': 2.}))])
    assert 'unresolved_cash_dividend' in {r['kind'] for r in actions.on_date('4123', EX)}


def stock_override(**changes):
    result = dict(shares_per_share=float(EXACT_STOCK_RATE), pay_date=PAY,
                  fractional_cash_per_share=0., source='Synthetic issuer-verified book-entry terms')
    result.update(changes)
    return {f'4123-{EX}': result}


def stock_policy():
    return dividend(CashEarningsDistribution=0., CashStatutorySurplus=0.,
                    StockExDividendTradingDate=EX, StockEarningsDistribution=.545425423,
                    StockStatutorySurplus=.4462571643)


def stock_event():
    return event(payload_json=json.dumps({'stock_dividend_per_1000': 99.16825874}))


def test_4123_precise_issuer_override_wins_over_exchange_rounded_total(tmp_path):
    actions = cached(tmp_path, [stock_policy()], [stock_event()], overrides=stock_override())
    row, = actions.on_date('4123', EX)
    assert row['kind'] == 'stock_dividend'
    assert Decimal(str(row['shares_per_share'])) == EXACT_STOCK_RATE
    assert row['pay_date'] == PAY
    assert row['fractional_cash_per_share'] == 0.


def test_4123_delivery_is_later_and_book_entry_fraction_does_not_create_cash(tmp_path):
    actions = cached(tmp_path, [stock_policy()], [stock_event()], overrides=stock_override())
    engine = bare_account(actions)
    new_whole = int((Decimal(1234) * EXACT_STOCK_RATE).to_integral_value(rounding=ROUND_FLOOR))
    engine.corporate_day(pd.Timestamp(EX))
    assert engine.holdings['4123']['qty'] == 1234
    assert engine.receivables[0]['qty'] == new_whole
    assert engine.receivables[0]['fraction'] > 0
    engine.corporate_day(pd.Timestamp('2023-10-12'))
    assert engine.holdings['4123']['qty'] == 1234
    engine.corporate_day(pd.Timestamp(PAY))
    assert engine.holdings['4123']['qty'] == 1234 + new_whole
    assert engine.cash == 1000.
    assert engine.receivables == []
    assert not any(r['cash_change'] > 0 for r in engine.cash_ledger)


def test_stock_dividend_without_delivery_terms_remains_blocking(tmp_path):
    actions = cached(tmp_path, [stock_policy()], [stock_event()])
    engine = bare_account(actions)
    with pytest.raises(UnresolvedAction, match='delivery'):
        engine.corporate_day(pd.Timestamp(EX))


def test_stock_policy_without_official_event_is_not_silently_discarded(tmp_path):
    actions = cached(tmp_path, [stock_policy()])
    engine = bare_account(actions)
    with pytest.raises((ValueError, UnresolvedAction)):
        engine.corporate_day(pd.Timestamp(EX))


def test_stock_delivery_before_ex_date_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        cached(tmp_path, [stock_policy()], [stock_event()],
               overrides=stock_override(pay_date='2023-09-07')).prepare('4123')


def test_2486_cash_subscription_waiver_creates_neither_cash_cost_nor_free_shares(tmp_path):
    sid = '2486'
    actions = cached(tmp_path, [dividend(sid, CashEarningsDistribution=0., CashStatutorySurplus=0.,
        StockExDividendTradingDate=EX, TotalNumberOfCashCapitalIncrease=12_000_000)],
        [event(sid)], sid=sid)
    row, = actions.on_date(sid, EX)
    assert row['kind'] == 'waive_subscription'
    engine = bare_account(actions, sid=sid)
    engine.corporate_day(pd.Timestamp(EX))
    assert engine.holdings[sid]['qty'] == 1234
    assert engine.cash == 1000.
    assert engine.cash_ledger == [] and engine.receivables == []
    assert engine.actions[0]['kind'] == 'waive_subscription'


def test_missing_cash_increase_flag_does_not_invent_a_waiver(tmp_path):
    actions = cached(tmp_path, [dividend()], [event(cash_increase_suspected=np.nan)])
    assert 'waive_subscription' not in {r['kind'] for r in actions.on_date('4123', EX)}


def test_0050_split_is_exact_four_and_does_not_duplicate_existing_cash_right(tmp_path):
    actions = cached(tmp_path, [], sid='0050')
    engine = bare_account(actions, sid='0050', qty=1001)
    engine.receivables = [dict(stock_id='0050', amount=250., kind='cash', pay_date=None)]
    engine.corporate_day(pd.Timestamp('2025-06-18'))
    assert engine.holdings['0050']['qty'] == 4004
    assert engine.marks['0050']['price'] == 12.5
    assert engine.receivables[0]['amount'] == 250.
    assert engine.cash == 1000.
    assert actions.reference_price('0050', '2025-06-18', 188.65) == 188.65 / 4


def test_offline_missing_source_fails_without_fetch_and_wrong_stock_is_rejected(tmp_path):
    actions = module.CorporateActions(pd.DataFrame(columns=EVENT_COLUMNS), tmp_path, offline=True)
    with pytest.raises(ValueError, match='missing'):
        actions.prepare('4123')
    with pytest.raises(ValueError, match='Wrong stock'):
        cached(tmp_path, [dividend('2486')]).prepare('4123')


def test_missing_fetch_dependency_is_visible_and_does_not_cache_empty_success(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ImportError('Install the missing provider dependency')
    monkeypatch.setattr(module, 'fetch_dataset', unavailable)
    actions = module.CorporateActions(pd.DataFrame(columns=EVENT_COLUMNS), tmp_path)
    with pytest.raises(ImportError, match='Install'):
        actions.prepare('4123')
    assert actions.loaded == {}
    assert not (tmp_path / '4123.parquet').exists()


def test_missing_parquet_dependency_never_falls_back_to_network(tmp_path, monkeypatch):
    actions = cached(tmp_path, [dividend()])
    def unavailable(*args, **kwargs):
        raise ImportError('Install pyarrow to read the frozen source')
    monkeypatch.setattr(module.pd, 'read_parquet', unavailable)
    with pytest.raises(ImportError, match='pyarrow'):
        actions.prepare('4123')
    assert actions.loaded == {}


def test_reference_uses_official_opening_then_reference_and_no_future_feed(tmp_path):
    actions = cached(tmp_path, [], [event(), event(day='2023-09-11', opening_ref=np.nan, ref_price=47.)])
    assert actions.reference_price('4123', EX, 52.) == 48.
    assert actions.reference_price('4123', '2023-09-11', 52.) == 47.
    assert actions.reference_price('4123', '2023-09-12', 52.) == 52.


@pytest.mark.parametrize('reference', [None, np.nan, 0., -1., float('inf')])
def test_company_action_reference_must_be_finite_positive_or_execution_blocks(tmp_path, reference):
    actions = cached(tmp_path, [], [event(opening_ref=reference, ref_price=reference)])
    with pytest.raises(UnresolvedAction):
        actions.reference_price('4123', EX, 52.)


def test_multiple_opening_references_are_not_arbitrarily_selected(tmp_path):
    actions = cached(tmp_path, [], [event(), event()])
    with pytest.raises(UnresolvedAction, match='Multiple'):
        actions.reference_price('4123', EX, 52.)


def test_manifest_records_actual_frozen_file_and_actions_are_returned_by_copy(tmp_path):
    actions = cached(tmp_path, [dividend()])
    expected = hashlib.sha256((tmp_path / '4123.parquet').read_bytes()).hexdigest()
    original = deepcopy(actions.on_date('4123', EX))
    returned = actions.on_date('4123', EX)
    returned[0]['cash_per_share'] = 999.
    assert actions.on_date('4123', EX) == original
    assert actions.manifest()['files_sha256']['4123.parquet'] == expected


def test_manifest_detects_frozen_source_changed_after_prepare(tmp_path):
    actions = cached(tmp_path, [dividend()])
    actions.prepare('4123')
    pd.DataFrame([dividend(CashEarningsDistribution=10.)]).to_parquet(tmp_path / '4123.parquet', index=False)
    with pytest.raises(ValueError, match='(?i)hash|changed|modified|source'):
        actions.manifest()
