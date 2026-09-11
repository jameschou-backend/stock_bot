from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from app import forward_corporate_audit as a, forward_journal as j, forward_portfolio as p
from app.finmind import FinMindQuotaError
from test_forward_portfolio import setup, buy, fill, close, clock


def account(tmp_path, filled=True):
    path = tmp_path / 'account'
    with j.connection(path) as con:
        setup(con)
        order = buy(con)
        if filled:
            fill(con, order, 1000, '100')
            close(con, '2026-09-14', {'2492':'100', '0050':'100'})
    return path


def cash(ex='2026-09-15', amount='10', pay='2026-09-17'):
    return dict(date='2026-09-15',stock_id='2492',CashExDividendTradingDate=ex,
                CashDividendPaymentDate=pay,CashEarningsDistribution=amount,CashStatutorySurplus=0,
                StockExDividendTradingDate='',StockEarningsDistribution=0,StockStatutorySurplus=0,
                TotalNumberOfCashCapitalIncrease=0)


def sources(path, evidence, data=None, at=None):
    at = at or clock('2026-09-15')
    sc = a.scope(a.account_rows(path, at), at)
    with j.connection(evidence) as con:
        for query in a.specs(sc):
            body = dict(version=a.VERSION,query=list(query),status='ok',retrieved_at=at().isoformat(),
                        rows=(data or {}).get(query[0],[]),error=None)
            j.append(con,'source:'+j.digest(body),'corporate_source',body,at)


def pair(amount='10'):
    return {a.POLICY:[cash()], a.RESULT:[dict(stock_id='2492',date='2026-09-15',
                    stock_and_cache_dividend=amount,stock_or_cache_dividend='息')]}


def codes(report):
    return {i['code'] for i in report['issues']}


def test_empty_is_not_coverage_and_missing_stale_future_sources_block(tmp_path):
    path=account(tmp_path); evidence=tmp_path/'sources'
    assert 'source_unavailable' in codes(a.inspect(path,evidence,clock('2026-09-15')))
    sources(path,evidence)
    result=a.inspect(path,evidence,clock('2026-09-15'))
    assert not result['blocked'] and result['manual_review_required'] and not result['coverage_complete']
    assert a.inspect(path,evidence,clock('2026-09-15',22))['blocked']
    assert a.inspect(path,evidence,clock('2026-09-15',19))['blocked']


def test_missing_dividend_blocks_without_mutating_account(tmp_path):
    path=account(tmp_path); evidence=tmp_path/'sources'; before=path.read_bytes()
    sources(path,evidence,pair())
    result=a.inspect(path,evidence,clock('2026-09-15'))
    assert 'missing_entitlement' in codes(result)
    assert result['events'][0]['eligible_qty']==1000
    assert path.read_bytes()==before


def test_policy_result_conflict_and_conflicting_revisions(tmp_path):
    path=account(tmp_path); evidence=tmp_path/'sources'
    data=pair('9');data[a.POLICY].append(cash(amount='11'))
    sources(path,evidence,data)
    result=a.inspect(path,evidence,clock('2026-09-15'))
    assert {'amount_conflict','policy_revision'} <= codes(result)


def test_paid_day_never_creates_cash_and_wrong_ledger_terms_block(tmp_path):
    path=account(tmp_path);evidence=tmp_path/'sources'
    with j.connection(path) as con:
        p.submit(con,dict(kind='entitlement',id='cash',action_id='cash',action_type='cash',stock_id='2492',
                 ex_date='2026-09-15',delivery_date='2026-09-17',eligible_qty=1000,
                 cash_per_share='10',amount='10000',evidence='official fixture'),clock('2026-09-15',8))
    sources(path,evidence,pair(),clock('2026-09-17'))
    before=p.summary(path,clock('2026-09-17'))
    result=a.inspect(path,evidence,clock('2026-09-17'))
    assert not result['blocked'] and 'payment_due' in codes(result)
    assert p.summary(path,clock('2026-09-17'))==before and before['cash']=='899980'
    data=pair();data[a.POLICY][0]['CashDividendPaymentDate']='2026-09-18'
    sources(path,evidence,data,clock('2026-09-17',21))
    assert 'policy_revision' in codes(a.inspect(path,evidence,clock('2026-09-17',21)))
    changed=tmp_path/'only-new-source'
    sources(path,changed,data,clock('2026-09-17',21))
    assert 'ledger_conflict' in codes(a.inspect(path,changed,clock('2026-09-17',21)))


@pytest.mark.parametrize('dataset', [a.SPLIT,a.REDUCTION,a.PAR])
def test_structural_actions_never_infer_ratio_or_payment(dataset,tmp_path):
    path=account(tmp_path); evidence=tmp_path/'sources'
    sources(path,evidence,{dataset:[dict(stock_id='2492',date='2026-09-15',before_price=100,after_price=25)]})
    assert 'structural_action' in codes(a.inspect(path,evidence,clock('2026-09-15')))
    assert p.summary(path,clock('2026-09-15'))['holdings'][0]['qty']==1000


def test_wrong_stock_invalid_dates_and_missing_numbers_fail_closed(tmp_path):
    path=account(tmp_path); evidence=tmp_path/'sources'
    data=pair();data[a.POLICY][0]['stock_id']='0050'
    data[a.RESULT][0]['date']='2026-09-16'
    sources(path,evidence,data)
    assert 'invalid_schema' in codes(a.inspect(path,evidence,clock('2026-09-15')))
    data=pair();data[a.POLICY][0]['CashEarningsDistribution']=None
    sources(path,evidence,data,clock('2026-09-15',21))
    assert 'invalid_terms' in codes(a.inspect(path,evidence,clock('2026-09-15',21)))


def test_ex_date_buyer_not_entitled(tmp_path):
    path=account(tmp_path); evidence=tmp_path/'sources'
    data=pair();data[a.POLICY][0]['CashExDividendTradingDate']='2026-09-14';data[a.RESULT][0]['date']='2026-09-14'
    sources(path,evidence,data)
    result=a.inspect(path,evidence,clock('2026-09-15'))
    assert result['events'][0]['eligible_qty']==0 and 'missing_entitlement' not in codes(result)


def test_refresh_bounded_resumable_empty_reused_and_global_par_query(tmp_path,monkeypatch):
    path=account(tmp_path);evidence=tmp_path/'sources';calls=[]
    monkeypatch.setattr('app.config.load_config',lambda:SimpleNamespace(finmind_token='fixture'))
    def fetch(ds,start,end,**kwargs):
        calls.append((ds,kwargs))
        df=pd.DataFrame();df.attrs['retrieved_at']=clock('2026-09-15')().timestamp()
        return df
    monkeypatch.setattr(a,'fetch_dataset',fetch)
    r=a.refresh(path,evidence,clock('2026-09-15'),request_budget=2)
    assert r['calls']==2 and r['report']['blocked']
    r=a.refresh(path,evidence,clock('2026-09-15'))
    assert r['calls']==3 and r['reused']==2 and not r['report']['blocked']
    assert calls[-1][0]==a.PAR and calls[-1][1]['data_id'] is None
    assert all(k['max_retries']==0 for _,k in calls)
    assert a.refresh(path,evidence,clock('2026-09-15'))['calls']==0


def test_quota_stops_immediately_and_keeps_progress(tmp_path,monkeypatch):
    path=account(tmp_path);evidence=tmp_path/'sources';calls=[]
    monkeypatch.setattr('app.config.load_config',lambda:SimpleNamespace(finmind_token='fixture'))
    def fetch(*args,**kwargs):
        calls.append(args)
        if len(calls)==2: raise FinMindQuotaError(900)
        df=pd.DataFrame();df.attrs['retrieved_at']=clock('2026-09-15')().timestamp()
        return df
    monkeypatch.setattr(a,'fetch_dataset',fetch)
    with pytest.raises(FinMindQuotaError): a.refresh(path,evidence,clock('2026-09-15'))
    assert len(calls)==2 and len(a.latest(evidence))==2


def test_close_checkbox_cannot_bypass_conflict_then_verified_close_seals_receipt(tmp_path,monkeypatch):
    path=account(tmp_path); evidence=tmp_path/'sources'
    monkeypatch.setattr('app.workbench_service.data_status',lambda:dict(data_ready=True,price_date='2026-09-15'))
    sources(path,evidence,pair())
    before=path.read_bytes()
    with pytest.raises(ValueError,match='漏登'): a.capture_close(path,True,clock('2026-09-15'),evidence)
    assert path.read_bytes()==before
    evidence=tmp_path/'clean-sources'
    sources(path,evidence,{},clock('2026-09-15',21))
    with pytest.raises(ValueError,match='官方'): a.capture_close(path,False,clock('2026-09-15',21),evidence)
    class DB:
        def scalars(self,q): return ['2026-09-11','2026-09-14','2026-09-15','2026-09-16','2026-09-17','2026-09-18']
        def execute(self,q): return [('2492',100),('0050',100)]
    @contextmanager
    def session(): yield DB()
    monkeypatch.setattr('app.db.get_session',session)
    result=a.capture_close(path,True,clock('2026-09-15',21),evidence)
    assert result['body']['nav']=='999980'
    rows=a.account_rows(path,clock('2026-09-15',21))
    assert rows[-2]['kind']=='corporate_review'
    assert a.capture_close(path,True,clock('2026-09-15',22),evidence)==result


def test_ui_render_never_fetches_and_shows_manual_limit(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    path=account(tmp_path); evidence=tmp_path/'sources';sources(path,evidence)
    report=a.inspect(path,evidence,clock('2026-09-15'))
    monkeypatch.setattr(a,'inspect',lambda *args:report)
    monkeypatch.setattr(a,'refresh',lambda *args:pytest.fail('render must not fetch'))
    app=AppTest.from_string('from app.forward_corporate_ui import render\nrender("unused", "fixture")').run()
    assert not app.exception
    assert '仍需人工核對' in app.info[0].value


def test_deleted_or_revised_source_does_not_erase_earlier_observation(tmp_path):
    path=account(tmp_path); evidence=tmp_path/'sources'
    sources(path,evidence,pair())
    sources(path,evidence,{},clock('2026-09-15',21))
    assert 'missing_entitlement' in codes(a.inspect(path,evidence,clock('2026-09-15',21)))
    data=pair(); data[a.POLICY][0]['CashEarningsDistribution']='9'
    sources(path,evidence,data,clock('2026-09-15',22))
    assert 'policy_revision' in codes(a.inspect(path,evidence,clock('2026-09-15',22)))


def test_selling_on_ex_date_still_requires_full_prior_shares(tmp_path):
    path=tmp_path/'account';evidence=tmp_path/'sources'
    with j.connection(path) as con:
        setup(con);order=buy(con);fill(con,order,1000,'100')
        close(con,'2026-09-14',{'2492':'87','0050':'100'})
        decision=p.submit(con,dict(kind='decision',id='stop',stock_id='2492',date='2026-09-14',reason='stop12_close'),clock('2026-09-14'))
        sell=p.submit(con,dict(kind='order',id='sell',order_id='sell',stock_id='2492',side='sell',channel='board',qty=1000,
            limit_price='87',session='2026-09-15',reason=decision['hash']),clock('2026-09-14'))
        fill(con,sell,1000,'90',day='2026-09-15',key='exit',tax='270')
    sources(path,evidence,pair())
    report=a.inspect(path,evidence,clock('2026-09-15'))
    assert report['events'][0]['eligible_qty']==1000
    assert 'missing_entitlement' in codes(report)
    assert not p.summary(path,clock('2026-09-15'))['holdings']


def test_eligibility_includes_previously_delivered_split(tmp_path):
    path=account(tmp_path)
    with j.connection(path) as con:
        p.submit(con,dict(kind='entitlement',id='split',action_id='split',action_type='split',stock_id='2492',
            ex_date='2026-09-15',delivery_date='2026-09-15',eligible_qty=1000,ratio='4',result_qty=4000,evidence='official'),clock('2026-09-15',8))
        p.submit(con,dict(kind='delivery',id='delivery',action_id='split',date='2026-09-15',evidence='report'),clock('2026-09-15',8))
        rows=j.read_events(con)
    assert a.before_ex(rows,'2026-09-16')['holdings']['2492']['qty']==4000
    assert a.before_ex(rows,'2026-09-15')['holdings']['2492']['qty']==1000
