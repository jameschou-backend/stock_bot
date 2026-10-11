from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.research_exit_scenarios import TrackedCorporateActions
from skills.million_replay import UnresolvedAction
from skills.rescheduled_corporate import CorporateCalendar
from skills.scenario_exit_replay import FractionalCashActions
import skills.all_entry_rescheduled_distributions as subject


def inputs():
    rows={sid+'-2024-07-26':dict(shares_per_share=rate,pay_date=pay,
        fractional_cash_per_share=10,fractional_cash_rounding='floor_ntd',fractional_cash_pay_date=None)
        for sid,rate,pay in [('1519',.1,'2024-08-30'),('2884',.020033197,'2024-08-29')]}
    corrections=[dict(stock_id=sid,scheduled_date='2024-07-25',effective_date='2024-07-26',
        lower=25,reference_price=28,upper=30,source='fixture',
        official_url='https://www.twse.com.tw/rwd/zh/exRight/TWT49U',
        official_row=['113年07月25日',sid]) for sid in subject.EVENTS]
    return rows,corrections


@pytest.mark.parametrize('sid',['1519','2884'])
def test_original_date_preparation_then_calendar_move_produces_one_entitlement(tmp_path,sid):
    rows,corrections=inputs()
    calendar=CorporateCalendar(pd.to_datetime(['2024-07-23','2024-07-26','2024-08-29','2024-08-30']),corrections)
    frame=pd.DataFrame([dict(stock_id=sid,StockExDividendTradingDate='2024-07-25',
        StockEarningsDistribution=1,CashExDividendTradingDate='2024-07-25',
        CashDividendPaymentDate='2024-08-30',CashEarningsDistribution=.5)])
    frame.to_parquet(tmp_path/(sid+'.parquet'),index=False)
    events=pd.DataFrame([dict(stock_id=sid,event_date='2024-07-25',source='ex_rights',
        event_type='權息',payload_json=json.dumps(dict(cash_dividend=.5)))])
    class Corp(TrackedCorporateActions):
        def prepare(self, stock):
            super().prepare(stock)
            self.loaded[stock]=calendar.actions(stock,self.loaded[stock])
    with pytest.raises(UnresolvedAction,match='Stock dividend data missing'):
        Corp(events,tmp_path,offline=True,overrides=rows).on_date(sid,'2024-07-26')
    fixed=subject.bind_scheduled_aliases(rows,rows,corrections)
    assert rows==inputs()[0]
    corp=Corp(events,tmp_path,offline=True,overrides=fixed)
    assert corp.on_date(sid,'2024-07-25')==[]
    actions=corp.on_date(sid,'2024-07-26')
    stocks=[r for r in actions if r['kind']=='stock_dividend']
    assert len(stocks)==1 and stocks[0]['action_id']==sid+'-2024-07-25-stock'
    assert stocks[0]['scheduled_ex_date']=='2024-07-25'
    assert stocks[0]['shares_per_share']==rows[sid+'-2024-07-26']['shares_per_share']
    assert stocks[0]['pay_date']==rows[sid+'-2024-07-26']['pay_date']
    cash=[r for r in actions if r['kind']=='cash_dividend']
    assert len(cash)==1 and cash[0]['cash_per_share']==.5 and cash[0]['pay_date']=='2024-08-30'
    rounded=FractionalCashActions(corp,SimpleNamespace(holdings={sid:dict(qty=1143)}))
    result=rounded.on_date(sid,'2024-07-26')
    fractional=[r for r in result if r.get('fractional_cash_gross')]
    assert any(r['kind']=='cash_dividend' and r['pay_date'] is None for r in result)
    assert next(r for r in result if r['kind']=='stock_dividend')['fractional_cash_per_share']==0


@pytest.mark.parametrize('change',['missing_move','changed_date','changed_terms','conflict_alias'])
def test_aliases_require_exact_prior_terms_and_calendar(change):
    rows,corrections=inputs();existing=deepcopy(rows)
    if change=='missing_move':corrections=corrections[1:]
    elif change=='changed_date':corrections[0]['effective_date']='2024-07-29'
    elif change=='changed_terms':existing['1519-2024-07-26']['pay_date']='2024-07-26'
    else:existing['1519-2024-07-25']=dict(rows['1519-2024-07-26'],shares_per_share=9)
    with pytest.raises(ValueError):subject.bind_scheduled_aliases(existing,rows,corrections)


def test_reapplying_aliases_is_idempotent_and_keeps_unrelated_terms():
    rows,corrections=inputs();existing=dict(rows,unrelated=dict(value=42))
    fixed=subject.bind_scheduled_aliases(existing,rows,corrections)
    assert fixed['unrelated']==existing['unrelated']
    assert subject.bind_scheduled_aliases(fixed,rows,corrections)==fixed
    fixed['1519-2024-07-25']['shares_per_share']=7
    assert fixed['1519-2024-07-26']==rows['1519-2024-07-26']


def test_real_archived_sources():
    root=Path(__file__).resolve().parents[1]
    if not (root/'.cache/all-entry-accounts-20261011/corporate-evidence/2884/agm-2024-minutes.pdf').exists():
        pytest.skip('Archived primary evidence unavailable')
    banks,_=subject.load_banks(root,root/list(subject.DOCUMENTS)[0])
    issuers,_=subject.load_issuers(root,root/list(subject.DOCUMENTS)[1])
    provider=SimpleNamespace(corporate_overrides={**banks,**issuers})
    refs=subject.install_distributions(provider,root)
    assert len(refs)>30
    for sid in subject.EVENTS:
        assert provider.corporate_overrides[sid+'-2024-07-25']==provider.corporate_overrides[sid+'-2024-07-26']
