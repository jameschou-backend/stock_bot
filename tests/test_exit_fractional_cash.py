"""Synthetic fractional rights: integer new shares and cash settle separately.

Reuse the zero-network scenario fixture. The independent original account
audit reconstructs cash, whole shares, trade costs and every daily NAV.
"""
from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.replay_million import audit
from skills.exit_policy import MODES
from skills.million_replay import Replay, UnresolvedAction
from skills.scenario_exit_replay import ExitSignals, FractionalCashActions, ScenarioExitReplay
from test_scenario_exit_replay import Corporate, ENTRY, Feeds, fixture


class SharedCorporate(Corporate):
    """Return provider-owned rows so accidental cross-case mutation is visible."""

    def __init__(self, events, overrides):
        super().__init__(events)
        self.overrides = overrides

    def on_date(self, sid, day):
        return self.events.get((sid, day), [])


def case(*, payment_index=None, ex_index=ENTRY+2, delivery_index=ENTRY+8,
         marked=True, n=ENTRY+14, rate=.05):
    days, adjusted, args, settings = fixture(n=n)
    ex, delivery = str(days[ex_index].date()), str(days[delivery_index].date())
    payment = None if payment_index is None else str(days[payment_index].date())
    row = dict(action_id='synthetic-stock-right', stock_id='1101', kind='stock_dividend',
               shares_per_share=rate, pay_date=delivery, fractional_cash_per_share=10.,
               source={'title': 'Synthetic issuer notice', 'pages': [2]})
    terms = dict(fractional_cash_rounding='floor_ntd', fractional_cash_pay_date=payment)
    provider = SharedCorporate({('1101', ex): [row]}, {f'1101-{ex}': terms} if marked else {})
    return days, adjusted, (*args[:-1], provider), settings, ex, delivery, payment


def replay_case(bundle, mode='fixed63'):
    days, adjusted, args, settings, *_ = bundle
    engine = ScenarioExitReplay(*args, exit_signals=ExitSignals(adjusted, days), mode=mode, **settings)
    result = engine.run()
    checks = audit(result)
    assert checks['all_daily_nav_reconciled'] and checks['cash_ledger_reconciled']
    assert checks['fees_recomputed'] and checks['daily_shares_reconstructed']
    return result


def cash_entitlements(account):
    return [row for row in account['corporate_actions']
            if row.get('distribution_type') == 'fractional_share_cash']


def cash_payments(account):
    return [row for row in account['cash_ledger']
            if row['kind'] in ('dividend_payment', 'fractional_share_payment')
            and row['cash_change'] > 0]


def entitlement_from_bought(account):
    # Compute independently from integer opening ownership, not adapter fields.
    quantity = account['cohorts'][0]['bought_qty']
    exact = Decimal(quantity) * Decimal('.05')
    whole = int(exact.to_integral_value(rounding=ROUND_FLOOR))
    fraction = exact-whole
    gross = int((fraction*10).to_integral_value(rounding=ROUND_FLOOR))
    return quantity, whole, fraction, gross


def test_unknown_payment_stays_cash_receivable_after_integer_stock_delivery():
    bundle = case()
    account = replay_case(bundle)
    _, _, _, _, ex, delivery, _ = bundle
    quantity, whole, fraction, gross = entitlement_from_bought(account)
    assert gross > 0
    action, = cash_entitlements(account)
    stock, = [row for row in account['corporate_actions'] if row['kind']=='stock_dividend']
    assert action['date']==ex and action['entitled_qty']==quantity
    assert action['entitlement_value']==action['gross_cash_amount']==gross
    assert action['pay_date'] is None
    assert stock['whole_new_shares']==whole and stock['fractional_cash_per_share']==0.
    assert stock['fractional_right']==float(fraction)
    assert not cash_payments(account)
    pending, = account['receivables']
    assert pending['kind']=='cash' and pending['amount']==gross
    assert pending['pay_date'] is None and pending['action_id']==action['action_id']
    daily = {row['date']: row for row in account['daily']}
    for day, row in daily.items():
        expected = 0 if day<ex else whole*50+gross if day<delivery else gross
        assert row['receivable']==expected
    delivered, = [row for row in account['corporate_actions'] if row['kind']=='share_delivery']
    assert delivered['date']==delivery and delivered['qty']==whole
    assert daily[delivery]['dividend_entitlement']==0


@pytest.mark.parametrize('payment_index', [ENTRY+2, ENTRY+5, ENTRY+8, ENTRY+10])
def test_only_explicit_cash_payment_date_moves_gross_cash_and_delivery_cannot_pay_twice(payment_index):
    bundle = case(payment_index=payment_index)
    account = replay_case(bundle)
    _, _, _, _, ex, delivery, payment = bundle
    quantity, whole, _, gross = entitlement_from_bought(account)
    positive, = cash_payments(account)
    assert positive['date']==payment and positive['cash_change']==gross
    assert positive['action_id']=='synthetic-stock-right-fractional-cash'
    assert not account['receivables']
    action, = cash_entitlements(account)
    assert action['fractional_settlement']['payment_date_verified'] is True
    assert action['fractional_settlement']['processing_or_remittance_fees_verified'] is False
    assert action['entitled_qty']==quantity
    for row in account['daily']:
        expected = (whole*50 if ex<=row['date']<delivery else 0)
        expected += gross if ex<=row['date']<payment else 0
        assert row['receivable']==expected
    # Cash consideration migrates from a locked entitlement; it is not income
    # for a second time on a later delivery/payment date.
    assert sum(row['dividend_entitlement'] for row in account['daily'])==gross
    assert sum(row['cash_change'] for row in cash_payments(account))==gross


def test_payment_after_account_end_remains_receivable_even_when_date_is_verified():
    bundle = case(n=ENTRY+30, payment_index=ENTRY+20)
    bundle[3]['end'] = str(bundle[0][ENTRY+12].date())
    account = replay_case(bundle)
    assert not cash_payments(account)
    pending, = account['receivables']
    assert pending['pay_date']==bundle[6]
    assert cash_entitlements(account)[0]['fractional_settlement']['payment_date_verified'] is True


def test_cash_payment_cannot_precede_the_ex_date():
    with pytest.raises(UnresolvedAction, match='Invalid fractional payment date'):
        replay_case(case(payment_index=ENTRY+1))


@pytest.mark.parametrize('bad', ['NaT', '', 'not-a-date', '2021-02-30',
                               '2021-07-07T00:00:00', '2021-07-07+08:00',
                               pd.Timestamp('2021-07-07'), 0, True])
def test_malformed_payment_dates_fail_with_the_explicit_source_error(bad):
    bundle = case()
    bundle[2][-1].overrides[f'1101-{bundle[4]}']['fractional_cash_pay_date'] = bad
    with pytest.raises(UnresolvedAction, match='Invalid fractional payment date'):
        replay_case(bundle)


@pytest.mark.parametrize('qty, fraction, gross', [(100, 0., 0), (101, .05, 0),
                                                (102, .1, 1), (119, .95, 9)])
def test_sub_one_ntd_fraction_does_not_create_a_positive_cash_right(qty, fraction, gross):
    bundle = case()
    provider = bundle[2][-1]
    owner = SimpleNamespace(holdings={'1101': {'qty': qty}})
    rows = FractionalCashActions(provider, owner).on_date('1101', bundle[4])
    stock, = [r for r in rows if r['kind']=='stock_dividend']
    assert stock['fractional_settlement']['fraction']==fraction
    assert stock['fractional_settlement']['gross_cash_amount']==gross
    cash = [r for r in rows if r.get('distribution_type')=='fractional_share_cash']
    assert len(cash)==int(gross>0)
    if cash:
        assert cash[0]['gross_cash_amount']==gross
        assert qty*cash[0]['cash_per_share']==pytest.approx(gross)


@pytest.mark.parametrize('face', [None, True, -10, np.nan, np.inf, '10'])
def test_missing_or_invalid_face_is_not_silently_interpreted_as_zero(face):
    bundle = case()
    bundle[2][-1].events[('1101', bundle[4])][0]['fractional_cash_per_share'] = face
    with pytest.raises(UnresolvedAction, match='Fractional face value missing'):
        replay_case(bundle)


def test_unknown_rounding_cannot_silently_use_floor():
    bundle = case()
    bundle[2][-1].overrides[f'1101-{bundle[4]}']['fractional_cash_rounding']='round_half_up'
    with pytest.raises(UnresolvedAction, match='Unsupported fractional rounding'):
        replay_case(bundle)


def test_purchase_on_the_ex_date_gets_neither_new_shares_nor_fractional_cash():
    bundle = case(ex_index=ENTRY)
    account = replay_case(bundle)
    assert account['cohorts'][0]['entry_date']==bundle[4]
    assert not account['corporate_actions'] and not account['receivables']
    assert not cash_payments(account)
    assert all(row['receivable']==0 for row in account['daily'])


def test_synthetic_cash_journal_labels_face_and_gross_instead_of_claiming_an_issuer_dividend_rate():
    account = replay_case(case())
    action, = cash_entitlements(account)
    _, _, fraction, gross = entitlement_from_bought(account)
    assert action['kind']=='cash_dividend'  # Compatibility with old journal algebra.
    assert action['distribution_type']=='fractional_share_cash'
    assert action['cash_per_share']!=10. and action['cash_per_share']!=.5
    assert action['gross_cash_amount']==action['entitlement_value']==gross
    assert action['fractional_settlement']==dict(
        fraction=float(fraction), face_value=10., gross_cash_amount=float(gross),
        rounding='floor_ntd', payment_date=None, payment_date_verified=False,
        processing_or_remittance_fees_verified=False)
    assert action['source']=={'title': 'Synthetic issuer notice', 'pages': [2]}


def test_all_seven_cases_can_share_a_provider_without_mutating_original_stock_terms():
    bundle = case(n=ENTRY+140)
    bundle[1].loc[bundle[0][ENTRY+4]:, '1101']=80.
    provider = bundle[2][-1]
    before = deepcopy((provider.events, provider.overrides))
    for mode in MODES:
        account = replay_case(bundle, mode)
        action, = cash_entitlements(account)
        assert action['gross_cash_amount']==3.
        assert account['receivables']==[dict(stock_id='1101', amount=3., kind='cash',
            pay_date=None, action_id='synthetic-stock-right-fractional-cash',
            event_id='first', ex_date=bundle[4])]
        assert (provider.events, provider.overrides)==before
        assert not cash_payments(account)


@pytest.mark.parametrize('rate', [.05, 1.])
def test_fixed63_without_marker_matches_original_full_account_even_with_stock_actions(rate):
    bundle = case(marked=False, n=ENTRY+70, rate=rate)
    _, _, args, settings, *_ = bundle
    original = Replay(*args, **settings).run()
    assert audit(original)['all_daily_nav_reconciled']
    assert replay_case(bundle)==original


def test_adapter_preserves_unmarked_provider_methods_and_rows():
    bundle = case(marked=False)
    provider = bundle[2][-1]
    provider.reference_price=lambda sid, day, prior: prior-2.
    wrapped = FractionalCashActions(provider, SimpleNamespace(holdings={}))
    assert wrapped.on_date('1101', bundle[4]) is provider.events[('1101', bundle[4])]
    assert wrapped.reference_price('1101', bundle[4], 50.)==48.
    assert wrapped.prepare('1101') is None


def test_certificate_metadata_survives_journal_while_ordinary_delivery_and_slot_wait_until_september_30():
    from scripts.research_exit_scenarios import exit_statistics

    # Dates match the lifecycle being modelled; every price, volume, company
    # and entry below is synthetic, not a replay of the issuer's history.
    days = pd.bdate_range('2024-01-02', periods=210)
    old_days, adjusted, args, settings = fixture(n=len(days), slots=1, events=[],
        paths={'1102': np.full(len(days), 100.)})
    quotes = args[0].copy()
    quotes['date'] = quotes.date.map(dict(zip(old_days, days)))
    adjusted.index = days
    ex, certificate, ordinary = '2024-08-13', '2024-08-30', '2024-09-30'
    ex_index = days.get_loc(ex)
    adjusted.loc[ex:, '1101'] = 80.
    entries = []
    for index, sid, identity in [(ex_index-2, '1101', 'certificate-owner'),
                                 (days.get_loc(certificate), '1102', 'slot-blocked'),
                                 (days.get_loc(ordinary), '1102', 'after-conversion')]:
        entries.append(dict(event_id=identity, members=[sid], priority=.1,
            signal_date=str(days[index-1].date()), entry_date=str(days[index].date())))
    restriction = dict(certificate_delivery_date=certificate,
        ordinary_share_available_date=ordinary, valuation_basis='ordinary_share_close_proxy',
        certificate_trading_modeled=False)
    corporate = SharedCorporate({('1101', ex): [dict(stock_id='1101',
        action_id='synthetic-certificate', kind='stock_dividend', shares_per_share=.01,
        pay_date=ordinary, fractional_cash_per_share=10.)]},
        {f'1101-{ex}': dict(fractional_cash_rounding='floor_ntd',
                            fractional_cash_pay_date=None, **restriction)})
    before = deepcopy((corporate.events, corporate.overrides))
    engine = ScenarioExitReplay(quotes, args[1], days, entries, Feeds(quotes), corporate,
        start=str(days[ENTRY-1].date()), end=str(days[-1].date()), slots=1,
        exit_signals=ExitSignals(adjusted, days), mode='loss12')
    account = engine.run()
    assert audit(account)['all_daily_nav_reconciled']
    stock_action, = [a for a in account['corporate_actions'] if a['kind']=='stock_dividend']
    assert stock_action['certificate_restriction']==restriction
    assert stock_action['date']==ex and stock_action['pay_date']==ordinary
    summary = exit_statistics(account, engine.exit_states, days)
    assert summary['restricted_certificate_actions']==[stock_action]
    delivery, = [a for a in account['corporate_actions'] if a['kind']=='share_delivery']
    assert delivery['date']==ordinary
    sales = [t for t in account['trades'] if t['stock_id']=='1101' and t['side']=='sell']
    assert {t['date'] for t in sales}=={'2024-08-14', ordinary}
    assert sum(t['qty'] for t in sales if t['date']==ordinary)==stock_action['whole_new_shares']
    assert any(o['event_id']=='slot-blocked' and o['failure']=='slots_full' for o in account['orders'])
    assert [c['event_id'] for c in account['cohorts']]==['certificate-owner', 'after-conversion']
    assert account['cohorts'][0]['exit_date']==ordinary
    cash_action, = cash_entitlements(account)
    for row in account['daily']:
        if certificate<=row['date']<ordinary:
            assert row['receivable']==stock_action['whole_new_shares']*50+cash_action['gross_cash_amount']
    assert not cash_payments(account)
    assert (corporate.events, corporate.overrides)==before
