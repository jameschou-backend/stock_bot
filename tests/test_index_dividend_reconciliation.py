from copy import deepcopy
from pathlib import Path
import pytest
from skills.index_dividend_reconciliation import issuer_rows,audit_account,URL
from scripts.research_exit_scenarios import write,sha


def table():
    rows=[[f'{y}/{m:02d}',f'{y}/{m:02d}/10',f'{y}/{m:02d}/20','1','0%','0%']
          for y in range(2026,2011,-1) for m in (7,1)]
    return dict(schema='issuer_visible_dividend_tables_v1',source_url=URL,security_id='0050',
        fund_heading='0050 元大台灣卓越50基金',columns=['配息年月','除息日','發放日','每單位配息金額'],
        pages=[dict(page=i+1,rows=rows[i*10:(i+1)*10]) for i in range(3)])


@pytest.mark.parametrize('bad',['fund','page','amount','date','duplicate','order','columns'])
def test_malformed_issuer_history_is_not_accepted(bad):
    t=table();assert len(issuer_rows(t))==30
    if bad=='fund':t['security_id']='0056'
    if bad=='page':t['pages'].pop()
    if bad=='amount':t['pages'][0]['rows'][0][3]='NaN'
    if bad=='date':t['pages'][0]['rows'][0][2]='2026/07/09'
    if bad=='duplicate':t['pages'][1]['rows'][0]=deepcopy(t['pages'][0]['rows'][0])
    if bad=='order':t['pages'][0]['rows'].reverse()
    if bad=='columns':t['columns'][1:3]=reversed(t['columns'][1:3])
    with pytest.raises(ValueError):issuer_rows(t)


def account():
    action=dict(kind='cash_dividend',action_id='div-2026',stock_id='0050',date='2026-01-10',
                pay_date='2026-01-20',cash_per_share=1.,entitled_qty=1000,entitlement_value=1000.)
    payment=dict(kind='payment',action_id='div-2026',stock_id='0050',date='2026-01-20',
                 pay_date='2026-01-20',ex_date='2026-01-10',amount=1000.)
    ledger=dict(kind='dividend_payment',action_id='div-2026',stock_id='0050',date='2026-01-20',cash_change=1000.)
    return dict(corporate_actions=[action,payment],cash_ledger=[ledger])


def test_cash_is_recognized_on_issuer_payment_date_without_mutating_account():
    a=account();before=deepcopy(a)
    result=audit_account(a,issuer_rows(table()),'2026-01-01','2026-06-30')
    assert result['matched_dividends']==1 and result['payments_on_issuer_dates']
    assert result['rows'][0]['cash_received']==1000. and a==before


@pytest.mark.parametrize('bad',['early_cash','late_cash','amount','entitlement','security','duplicate','missing','event_date','wrong_ex'])
def test_cash_date_amount_identity_or_missing_receipts_fail(bad):
    a=account()
    if bad=='early_cash':a['cash_ledger'][0]['date']='2026-01-10'
    if bad=='late_cash':a['cash_ledger'][0]['date']='2026-01-21'
    if bad=='amount':a['cash_ledger'][0]['cash_change']=999
    if bad=='entitlement':a['corporate_actions'][0]['entitlement_value']=999
    if bad=='security':a['corporate_actions'][1]['stock_id']='0056'
    if bad=='duplicate':a['corporate_actions'].append(deepcopy(a['corporate_actions'][1]))
    if bad=='missing':a['cash_ledger'].clear()
    if bad=='event_date':a['corporate_actions'][1]['date']='2026-01-10'
    if bad=='wrong_ex':a['corporate_actions'][1]['ex_date']='2025-01-10'
    with pytest.raises(ValueError):audit_account(a,issuer_rows(table()),'2026-01-01','2026-06-30')


def test_additive_ui_recomputes_evidence_and_keeps_live_false(tmp_path,monkeypatch):
    from app import index_dividend_ui as ui
    from skills.backtest_case_cache import file_identities
    code=tmp_path/'audit.py';code.write_text('fixed audit')
    core=dict(passed=True,strict_data_ready=False,live_qualified=False,unseen_validation=False,
              changed_accounts=0,new_backtests=0,unique_period_dividends=21)
    monkeypatch.setattr(ui,'reconcile',lambda root:deepcopy(core))
    p=tmp_path/ui.PUBLICATION
    value={**core,'audit_code':file_identities([code],tmp_path)}
    def publish():write(p,value);p.with_suffix('.sha256').write_text(sha(p))
    publish();assert ui.load(tmp_path)['live_qualified'] is False
    value['unique_period_dividends']=22;publish()
    with pytest.raises(ValueError):ui.load(tmp_path)
    value['unique_period_dividends']=21;publish();code.write_text('changed audit')
    with pytest.raises(ValueError):ui.load(tmp_path)
