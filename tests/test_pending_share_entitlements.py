"""Undated rights must affect NAV without becoming cash or tradable stock."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, ROUND_FLOOR
import hashlib

import pandas as pd
import pytest

from skills import pending_share_entitlements as pending
from skills import sector_account_replay as research
from scripts.research_exit_scenarios import TrackedCorporateActions
from test_sector_account_replay import account_data, Corporate, Feeds, configuration


@pytest.fixture
def setup_case(tmp_path, monkeypatch):
    data = account_data(end=145)
    ex = str(data.days[140].date())
    bound = str(data.days[146].date())
    proof = tmp_path / 'primary.html'
    proof.write_text('Synthetic official record-date evidence for testing only')
    terms = dict(pending_only=True, ordinary_share_delivery_status='unannounced',
        shares_per_share=.1, pay_date=None, ordinary_share_available_date=None,
        pending_delivery_not_before=bound, record_date=bound,
        pending_not_before_basis='record_date', entitlement_announcement_date=data.start,
        fractional_cash_per_share=10., fractional_cash_rounding='floor_ntd',
        fractional_cash_pay_date=None, certificate_trading_modeled=False,
        certificate_delivery_date=str(data.days[160].date()), valuation_basis='ordinary_share_close_proxy',
        evidence_files=['primary.html'], pending_delivery_evidence=dict(path='primary.html',
            url='https://mopsov.twse.com.tw/mops/web/example',
            sha256=hashlib.sha256(proof.read_bytes()).hexdigest()))
    action = dict(stock_id='1101', date=ex, action_id='1101-stock-'+ex,
        kind='stock_dividend', shares_per_share=.1, pay_date=None, fractional_cash_per_share=10.)
    monkeypatch.setattr(research, 'install_pending_share_rights',
        lambda engine: pending.install_pending_share_rights(engine, source_root=tmp_path))
    return data, ex, terms, action, tmp_path


def run(data, ex, terms, action, *, board=False, stress='control', corporate=None):
    source = corporate or Corporate({('1101', ex): [action]})
    source.overrides = {'1101-'+ex: deepcopy(terms)}
    return research.run_case(data, configuration(board=board, stress=stress), '.', source.overrides,
        feeds=Feeds(data.quotes), corporate=source)


@pytest.mark.parametrize('board', [False, True])
def test_pending_whole_and_fraction_keep_cash_shares_and_daily_nav_reconciled(setup_case, board):
    data, ex, terms, action, _ = setup_case
    result = run(data, ex, terms, action, board=board)
    assert result['completed'], result.get('reason')
    account = result['account']
    share_action = next(row for row in account['corporate_actions'] if row['kind']=='stock_dividend')
    qty = share_action['entitled_qty']
    new = Decimal(qty) * Decimal('.1')
    whole = int(new.to_integral_value(rounding=ROUND_FLOOR))
    fractional = float(((new-whole)*10).to_integral_value(rounding=ROUND_FLOOR))
    right = next(row for row in account['receivables'] if row['kind']=='shares')
    assert right['qty']==whole and right['pay_date'] is None and right['tradable'] is False
    assert right['delivery_status']=='pending_unannounced'
    last = account['daily'][-1]
    assert last['receivable']==pytest.approx(whole*50+fractional)
    before = next(row for row in account['daily'] if row['date']==str(data.days[139].date()))
    assert last['cash']==before['cash']
    assert last['market_value']==before['market_value']
    assert not any(row['kind'] in ('dividend_payment','fractional_share_payment') for row in account['cash_ledger'])
    assert not any(row['kind']=='share_delivery' for row in account['corporate_actions'])
    for row in account['daily']:
        assert row['nav']==pytest.approx(row['cash']+row['market_value']+row['receivable'])
        assert row['nav']==pytest.approx(row['opening_nav']+row['market_pnl']+
            row['dividend_entitlement']+row['execution_basis_pnl']-row['cost'], abs=.06)


@pytest.mark.parametrize('end_index', [146, 147])
def test_equal_or_later_than_evidenced_boundary_blocks(setup_case, end_index):
    data, ex, terms, action, _ = setup_case
    result = run(replace(data, end=str(data.days[end_index].date())), ex, terms, action)
    assert not result['completed'] and 'boundary' in result['reason']
    assert 'summary' not in result


@pytest.mark.parametrize('mutation', ['missing_source','changed_source','missing_bound','fake_pay_date','invalid_rate'])
def test_unverified_or_changed_terms_fail_closed(setup_case, mutation):
    data, ex, terms, action, root = setup_case
    if mutation=='missing_source': terms.pop('pending_delivery_evidence')
    if mutation=='changed_source': (root/'primary.html').write_text('changed')
    if mutation=='missing_bound': terms.pop('pending_delivery_not_before')
    if mutation=='fake_pay_date': terms['pay_date']='2099-01-01'
    if mutation=='invalid_rate': action['shares_per_share']=float('nan')
    result = run(data, ex, terms, action)
    assert not result['completed'] and 'Pending share' in result['reason']
    assert not result['partial_account']['receivables']


@pytest.mark.parametrize('arm', research.ARMS)
@pytest.mark.parametrize('board', [False, True])
@pytest.mark.parametrize('stress', ['control','combined'])
def test_all_eight_unaffected_strategy_accounts_are_exact(setup_case, monkeypatch, arm, board, stress):
    data, ex, terms, _, _ = setup_case
    config = configuration(arm=arm, board=board, stress=stress)
    source = Corporate()
    # Invalid evidence on an asset not held must not influence the account.
    source.overrides = {'2999-'+ex: dict(terms, pending_delivery_evidence={})}
    with_adapter = research.run_case(data, config, '.', source.overrides, feeds=Feeds(data.quotes), corporate=source)
    monkeypatch.setattr(research, 'install_pending_share_rights', lambda engine: None)
    original = research.run_case(data, config, '.', source.overrides, feeds=Feeds(data.quotes), corporate=source)
    assert original['completed'] and with_adapter == original


def test_pending_null_date_preserves_tracked_loader_and_observer(setup_case):
    data, ex, terms, _, root = setup_case
    frame = pd.DataFrame([dict(stock_id='1101', StockExDividendTradingDate=ex,
                               StockEarningsDistribution=1.)])
    frame.to_parquet(root/'1101.parquet', index=False)
    pd.DataFrame().to_parquet(root/'0050.parquet', index=False)
    events = pd.DataFrame([dict(stock_id='1101',event_date=ex,source='ex_rights',
        event_type='權',payload_json='{"stock_dividend_per_1000":100}',cash_increase_suspected=False)])
    provider = TrackedCorporateActions(events,root,offline=True,overrides={'1101-'+ex:terms})
    with pytest.raises(pending.UnresolvedAction, match='pay_date'):
        provider.on_date('1101', ex)
    wrapper = pending.PendingOnlySource(provider, data.end, root)
    rows = wrapper.on_date('1101', ex)
    assert rows[0]['pay_date'] is None
    assert provider.manifest()['files_sha256']['1101.parquet']
    assert provider.requests == 0


def test_without_explicit_pending_terms_old_missing_delivery_still_blocks(setup_case):
    data, ex, terms, action, _ = setup_case
    terms.pop('pending_only')
    result=run(data, ex, terms, action)
    assert not result['completed'] and 'Stock delivery date missing' in result['reason']
