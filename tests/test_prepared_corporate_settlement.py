from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, ROUND_FLOOR

import pandas as pd
import pytest

from skills.account_source_preflight import prepared_case, digest
from skills.corporate_account_audit import audit_corporate_account, audit_capital_cash
from skills.prepared_corporate_settlement import validate_delivery_terms
from skills.million_replay import UnresolvedAction
from test_sector_account_replay import account_data, configuration, Corporate, Feeds


def capital_case(tmp_path, *, end=160, ratio=.8, payment_index=150, missing_quote=False):
    data = account_data(end=end)
    mask = data.quotes.stock_id.eq('1101')
    data.quotes.loc[mask, ['open','high','low','close']] += 1
    day = str(data.days[140].date())
    ref = str(data.days[139].date())
    action = dict(action_id='test-capital', stock_id='1101', date=day, kind='capital_reduction',
        multiplier=ratio, cash_per_share=2., cash_rounding='floor_ntd', known_date=data.start,
        pay_date=str(data.days[payment_index].date()), fractional_reference_date=ref,
        fractional_policy='historical_close_gross_receivable', fractional_cash_rounding='floor_ntd',
        fractional_cash_pay_date=None, evidence_files=['synthetic-official-notice'])
    if missing_quote:
        data.quotes.loc[data.quotes.stock_id.eq('1101') & data.quotes.date.eq(ref), 'close'] = float('nan')
    source = Corporate({('1101', day): [action]})
    source.overrides = {'1101-'+day: action}
    return prepared_case(data, configuration(), tmp_path, source.overrides,
        feeds=Feeds(data.quotes), corporate=source), data, action


def test_fractional_exchange_keeps_gross_right_separate_and_reconciles(tmp_path):
    result, data, terms = capital_case(tmp_path)
    assert result['completed'], result.get('reason')
    account = result['account']
    action = next(a for a in account['corporate_actions'] if a['kind']=='capital_reduction')
    expected = Decimal(action['entitled_qty'])*Decimal('.8')
    assert expected != expected.to_integral_value()  # actually exercise the old failure
    whole = int(expected.to_integral_value(rounding=ROUND_FLOOR))
    assert action['qty_after'] == whole
    assert action['capital_cash_amount'] == action['entitled_qty']*2
    residual = next(r for r in account['receivables'] if r['action_id'].endswith('-fractional-cash'))
    assert residual['amount'] == int((expected-whole)*51)
    assert residual['pay_date'] is None and residual['net_amount_verified'] is False
    paid = [r for r in account['cash_ledger'] if r.get('action_id')=='test-capital-capital-cash']
    assert len(paid)==1 and paid[0]['date']==terms['pay_date']
    assert paid[0]['cash_change']==action['capital_cash_amount']
    assert paid[0]['cash_flow_nature']=='capital_return'
    assert not any(r.get('action_id','').endswith('-fractional-cash') for r in account['cash_ledger'])
    assert audit_corporate_account(account)['capital_reductions_reconciled']


def test_return_unpaid_before_its_own_date(tmp_path):
    result, _, _ = capital_case(tmp_path, end=145)
    assert result['completed']
    account = result['account']
    assert len([r for r in account['receivables'] if r['action_id'].startswith('test-capital')]) == 2
    assert not any(r.get('action_id','').startswith('test-capital') for r in account['cash_ledger'])


def test_exchange_to_zero_whole_shares_closes_slot_but_keeps_cash_rights(tmp_path):
    result, _, terms = capital_case(tmp_path, ratio=.00001)
    assert result['completed'], result.get('reason')
    account = result['account']
    action = next(a for a in account['corporate_actions'] if a['kind']=='capital_reduction')
    assert action['qty_after'] == 0
    assert not any(h['stock_id']=='1101' and h['date'] >= terms['date'] for h in account['holdings'])
    assert any(r['action_id']=='test-capital-fractional-cash' for r in account['receivables'])
    assert audit_corporate_account(account)['capital_reductions_reconciled']


@pytest.mark.parametrize('field', ['qty_after', 'capital_cash_amount', 'fractional_gross_amount'])
def test_auditor_rejects_tampered_exchange_or_cash(tmp_path, field):
    result, _, _ = capital_case(tmp_path)
    account = deepcopy(result['account'])
    action = next(a for a in account['corporate_actions'] if a['kind']=='capital_reduction')
    action[field] += 1
    with pytest.raises(ValueError):
        audit_corporate_account(account)


def test_auditor_rejects_premature_fraction_payment(tmp_path):
    result, _, _ = capital_case(tmp_path)
    account = deepcopy(result['account'])
    account['cash_ledger'].append(dict(action_id='test-capital-fractional-cash'))
    with pytest.raises(ValueError, match='paid early'):
        audit_capital_cash(account)


def test_missing_fraction_reference_still_blocks(tmp_path):
    result, _, _ = capital_case(tmp_path, missing_quote=True)
    assert not result['completed'] and 'quote missing' in result['reason']
    assert not result['partial_account']['corporate_actions']


def pending_terms(tmp_path):
    p=tmp_path/'qa.html';p.write_text('Synthetic agent QA for testing')
    proof=dict(path=p.name,sha256=digest(p),url='https://www.yuanta.com.tw/example')
    return dict(pending_only=True,ordinary_share_delivery_status='unannounced',pay_date=None,
        ordinary_share_available_date=None,use_scope='account_settlement_only_not_selection',
        pending_confirmation_date='2026-09-09',pending_confirmed_through='2026-09-09',
        pending_delivery_not_before='2026-09-10',record_date='2026-09-08',
        entitlement_announcement_date='2026-08-14',pending_not_before_basis='dated_agent_pending_confirmation',
        pending_delivery_evidence=proof,pending_confirmation_index_evidence=proof,evidence_files=[p.name])


def test_dated_pending_evidence_is_bounded_and_not_a_delivery_date(tmp_path):
    terms=pending_terms(tmp_path)
    assert validate_delivery_terms(terms,'6669','2026-09-02','2026-09-09',tmp_path)=='2026-09-10'
    assert terms['pay_date'] is None
    with pytest.raises(UnresolvedAction,match='does not cover'):
        validate_delivery_terms(terms,'6669','2026-09-02','2026-09-10',tmp_path)


@pytest.mark.parametrize('change', ['hash','index','payment','confirmation'])
def test_pending_status_cannot_be_forged_or_promoted_to_cash(tmp_path, change):
    terms=pending_terms(tmp_path)
    if change=='hash':(tmp_path/'qa.html').write_text('changed')
    if change=='index':terms.pop('pending_confirmation_index_evidence')
    if change=='payment':terms['pay_date']='2026-09-09'
    if change=='confirmation':terms['pending_confirmation_date']='2026-09-08'
    with pytest.raises(UnresolvedAction):
        validate_delivery_terms(terms,'6669','2026-09-02','2026-09-09',tmp_path)


def test_dated_confirmation_creates_untradable_rights_through_account_end(tmp_path, monkeypatch):
    from skills import sector_account_replay, pending_share_entitlements
    data=account_data(end=145)
    ex=str(data.days[140].date())
    terms=pending_terms(tmp_path)
    terms.update(shares_per_share=1.9827946,fractional_cash_per_share=0.,
        fractional_cash_rounding='floor_ntd',fractional_cash_pay_date=None,
        entitlement_announcement_date=data.start,record_date=str(data.days[144].date()),
        pending_confirmation_date=data.end,pending_confirmed_through=data.end,
        pending_delivery_not_before=str((pd.Timestamp(data.end)+pd.Timedelta(days=1)).date()))
    action=dict(stock_id='1101',date=ex,action_id='pending-shares',kind='stock_dividend',
        shares_per_share=terms['shares_per_share'],pay_date=None,fractional_cash_per_share=0.)
    source=Corporate({('1101',ex):[action]});source.overrides={'1101-'+ex:terms}
    monkeypatch.setattr(sector_account_replay,'install_pending_share_rights',
        lambda account:pending_share_entitlements.install_pending_share_rights(account,tmp_path))
    result=prepared_case(data,configuration(),tmp_path,source.overrides,feeds=Feeds(data.quotes),corporate=source)
    assert result['completed'],result.get('reason')
    account=result['account']
    rights=[r for r in account['receivables'] if r['kind']=='shares']
    assert len(rights)==1 and rights[0]['pay_date'] is None and rights[0]['tradable'] is False
    original_qty=sum(t['qty'] for t in account['trades'] if t['side']=='buy' and t['stock_id']=='1101')
    assert rights[0]['qty']==int(Decimal(original_qty)*Decimal('1.9827946'))
    assert account['holdings'][-1]['qty']==original_qty
    assert not any(r['kind']=='share_delivery' for r in account['corporate_actions'])


@pytest.mark.parametrize('stress',['control','combined'])
def test_unaffected_account_cash_trades_and_holdings_remain_exact(tmp_path, stress):
    from skills import sector_account_replay
    data=account_data();config=configuration(stress=stress)
    old=sector_account_replay.run_case(data,config,tmp_path,{},feeds=Feeds(data.quotes),corporate=Corporate())
    new=prepared_case(data,config,tmp_path,{},feeds=Feeds(data.quotes),corporate=Corporate())
    assert old['completed'] and new['completed']
    assert old['account']==new['account']
