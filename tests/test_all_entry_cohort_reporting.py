from copy import deepcopy

import pandas as pd
import pytest

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest
from skills.all_entry_cohort_reporting import describe_cohorts


def fixture(tmp_path):
    p=tmp_path/'inputs/raw-close.parquet';p.parent.mkdir();pd.DataFrame([{'date':'2026-10-02','9941':76.7}]).to_parquet(p)
    action=dict(action_id='a',stock_id='9941',event_id='e',kind='stock_dividend',date='2026-09-01',
                pay_date='2026-10-16',entitled_qty=1000,shares_per_share=.02,whole_new_shares=20)
    right=dict(action_id='a',stock_id='9941',event_id='e',kind='shares',ex_date='2026-09-01',
               pay_date='2026-10-16',qty=20,fractional_cash_per_share=0)
    case=dict(account=dict(cohorts=[dict(event_id='e',stock_id='9941',name='A',entry_date='2026-08-03')],
        corporate_actions=[action],cash_ledger=[dict(kind='initial_deposit',cash_change=100000.),
            dict(kind='buy',event_id='e',stock_id='9941',cash_change=-50000.),
            dict(kind='sell',event_id='e',stock_id='9941',cash_change=60000.)],receivables=[right]),
        summary=dict(end='2026-10-02',initial_cash=100000.,cash=110000.,market_value=0.,
            final_holdings=[],final_receivables=[right],receivable=1534.,profit=11534.))
    report=dict(input_bundle='inputs',source_sha256={'inputs/raw-close.parquet':digest(p)})
    return case,report,Sources(tmp_path),p


def test_pending_share_value_reconciles_but_remains_unsettled(tmp_path):
    case,report,sources,_=fixture(tmp_path);before=deepcopy(case)
    result=describe_cohorts(case,report,sources)
    assert result['funded']==result['unresolved']==1 and result['settled']==0 and result['win_rate'] is None
    proof=result['noncash_rights_valuation'][0]
    assert proof['value']==1534 and proof['spendable_cash']==0 and proof['settled'] is False
    assert case==before and case['account']['receivables'][0]['kind']=='shares'


@pytest.mark.parametrize('change',['unbound','wrong_quote','early_delivery','duplicate','inflated','wrong_cohort','spent_fraction','missing_date'])
def test_inconsistent_pending_valuation_is_rejected(tmp_path,change):
    case,report,sources,path=fixture(tmp_path)
    if change=='unbound':report['source_sha256']={}
    elif change in ('wrong_quote','missing_date'):
        pd.DataFrame([{'date':'2026-10-01' if change=='missing_date' else '2026-10-02','9941':77.}]).to_parquet(path)
        report['source_sha256']['inputs/raw-close.parquet']=digest(path)
    elif change=='early_delivery':case['account']['corporate_actions'].append(dict(case['account']['corporate_actions'][0],kind='share_delivery'))
    elif change=='duplicate':case['account']['corporate_actions'].append(deepcopy(case['account']['corporate_actions'][0]))
    elif change=='inflated':case['account']['receivables'][0]['qty']=21
    elif change=='wrong_cohort':case['account']['receivables'][0]['event_id']='other'
    else:case['account']['receivables'][0]['fractional_cash_per_share']=10
    with pytest.raises(ValueError):describe_cohorts(case,report,sources)
