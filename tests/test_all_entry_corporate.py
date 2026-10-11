import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from skills.all_entry_accounts import digest
from skills.all_entry_corporate import load_terms
from skills.million_replay import Replay
from skills.scenario_exit_replay import FractionalCashActions


def fixture(root):
    # This test is independent of downloaded/private cache files.
    document = json.loads((Path(__file__).resolve().parents[1] /
        'docs/all_entry_corporate_terms_v1_20261011.json').read_text())
    for notice in document['evidence']['4763-2024-09-12']['sources']:
        path = root/notice['path']; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Synthetic test notice: ' + ' '.join(notice['required_text']))
        notice['sha256'] = digest(path)
        day = pd.Timestamp(notice['announcement_date'])
        receipt = dict(url=notice['url'],http_status=200,redirected=False,sha256=digest(path),
            announcement_row=['4763','synthetic',f'{day.year-1911:03d}/{day.month:02d}/{day.day:02d}'])
        (root/notice['receipt_path']).write_text(json.dumps(receipt))
        notice['receipt_sha256'] = digest(root/notice['receipt_path'])
    return document


def save(root, document):
    path=root/'terms.json';path.write_text(json.dumps(document));return path


@pytest.mark.parametrize('field,value', [
    ('shares_per_share', .15), ('fractional_cash_per_share', 1),
    ('pay_date','2024-09-12'), ('ordinary_share_available_date','2024-09-12'),
    ('fractional_cash_available_for_trading',True), ('fractional_cash_pay_date','2024-10-09')])
def test_no_guessed_rate_premature_delivery_or_fractional_cash(tmp_path,field,value):
    document=fixture(tmp_path)
    document['overrides']['4763-2024-09-12'][field]=value
    with pytest.raises(ValueError):load_terms(tmp_path,save(tmp_path,document))


def test_primary_evidence_is_checked_not_just_the_document(tmp_path):
    document=fixture(tmp_path);path=save(tmp_path,document)
    terms,refs=load_terms(tmp_path,path)
    assert len(refs)==7 and terms['4763-2024-09-12']['shares_per_share']==.14978205387
    notice=document['evidence']['4763-2024-09-12']['sources'][0]
    (tmp_path/notice['path']).write_text('changed')
    with pytest.raises(ValueError,match='hash'):load_terms(tmp_path,path)


def test_actual_388_share_exposure_delivers_58_shares_but_no_guessed_net_cash(tmp_path):
    terms,_=load_terms(tmp_path,save(tmp_path,fixture(tmp_path)))
    row=terms['4763-2024-09-12']
    action=dict(action_id='4763-2024-09-12-stock',stock_id='4763',kind='stock_dividend',
        shares_per_share=row['shares_per_share'],pay_date=row['pay_date'],fractional_cash_per_share=10.)
    provider=SimpleNamespace(overrides=terms,on_date=lambda sid,day:[action] if day=='2024-09-12' else [])
    account=Replay.__new__(Replay)
    account.holdings={'4763':dict(qty=388,event_id='poc-4763',due_index=9)}
    account.cohorts=[dict(event_id='poc-4763',stock_id='4763',entry_date='2024-09-03',due_index=9)]
    account.receivables=[];account.actions=[];account.cash_ledger=[];account.cash=1000.
    account.marks={'4763':dict(price=800.)};account.raw=lambda day,sid:dict(close=800.)
    account.corporate=FractionalCashActions(provider,account)
    account.corporate_day(pd.Timestamp('2024-09-12'))
    shares,=[r for r in account.receivables if r['kind']=='shares']
    assert shares['qty']==58 and shares['pay_date']=='2024-10-09'
    account.holdings={}
    account.corporate_day(pd.Timestamp('2024-10-08'))
    assert not account.holdings
    account.corporate_day(pd.Timestamp('2024-10-09'))
    assert account.holdings['4763']['qty']==58 and account.holdings['4763']['event_id']=='poc-4763'
    assert account.cash==1000. and account.receivables[0]['amount']==1.
    assert account.receivables[0]['pay_date'] is None
