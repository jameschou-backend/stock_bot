from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from skills.all_entry_accounts import digest
from skills.all_entry_settlement_completion import load_completion, install_completion
from skills.million_replay import Replay
from skills.scenario_exit_replay import FractionalCashActions


def fixture(root):
    repo = Path(__file__).resolve().parents[1]
    document = json.loads((repo/'docs/all_entry_6669_delivery_20261011.json').read_text())
    previous = document['previous_document']
    old = root/previous['path'];old.parent.mkdir(parents=True,exist_ok=True)
    old.write_bytes((repo/previous['path']).read_bytes())
    for notice in document['sources']:
        path = root/notice['path'];path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('Synthetic test notice: '+' '.join(notice['required_text']))
        notice['sha256'] = digest(path)
        day = pd.Timestamp(notice['announcement_date'])
        receipt = dict(url=notice['url'],http_status=200,redirected=False,sha256=digest(path),
            announcement_row=['6669','synthetic',f'{day.year-1911:03d}/{day.month:02d}/{day.day:02d}'])
        (root/notice['receipt_path']).write_text(json.dumps(receipt))
        notice['receipt_sha256'] = digest(root/notice['receipt_path'])
    return document


def save(root, document):
    path=root/'completion.json';path.write_text(json.dumps(document));return path


def test_october_delivery_keeps_all_new_shares_unavailable_at_account_endpoint(tmp_path):
    key,old,row,_=load_completion(tmp_path,save(tmp_path,fixture(tmp_path)))
    assert old['pending_only'] is True and 'pending_only' not in row
    action=dict(action_id=key+'-stock',stock_id='6669',kind='stock_dividend',
        shares_per_share=row['shares_per_share'],pay_date=row['pay_date'],fractional_cash_per_share=0.)
    provider=SimpleNamespace(overrides={key:row},on_date=lambda sid,day:[action] if day=='2026-09-02' else [])
    account=Replay.__new__(Replay)
    account.holdings={'6669':dict(qty=42,event_id='near-high-6669',due_index=9)}
    account.cohorts=[dict(event_id='near-high-6669',stock_id='6669',entry_date='2026-08-20',due_index=9)]
    account.receivables=[];account.actions=[];account.cash_ledger=[];account.cash=1000.
    account.marks={'6669':dict(price=1000.)};account.raw=lambda day,sid:dict(close=1000.)
    account.corporate=FractionalCashActions(provider,account)
    account.corporate_day(pd.Timestamp('2026-09-02'))
    shares,=account.receivables
    assert shares['qty']==83 and shares['pay_date']=='2026-10-14'
    account.holdings={}
    account.corporate_day(pd.Timestamp('2026-10-02'))
    assert not account.holdings and account.receivables==[shares]
    assert account.cash==1000.
    account.corporate_day(pd.Timestamp('2026-10-14'))
    assert account.holdings['6669']['qty']==83 and not account.receivables
    assert account.cash==1000.


@pytest.mark.parametrize('field,value', [('pay_date','2026-09-08'),('shares_per_share',2.),
    ('fractional_cash_per_share',10.),('pending_only',True),('fractional_net_zero_verified',False)])
def test_no_earlier_shares_or_changed_entitlements(tmp_path,field,value):
    document=fixture(tmp_path);document['replacement'][field]=value
    with pytest.raises(ValueError):load_completion(tmp_path,save(tmp_path,document))


def test_only_exact_old_pending_action_can_be_replaced_without_global_mutation(tmp_path):
    path=save(tmp_path,fixture(tmp_path));key,old,new,_=load_completion(tmp_path,path)
    def original(base,additions):
        result=deepcopy(base)
        for k,v in additions.items():
            if k in result and result[k]!=v:raise ValueError('conflicting ordinary terms')
            result[k]=deepcopy(v)
        return result
    runner=SimpleNamespace(merge_overrides=original)
    provider=SimpleNamespace(corporate_overrides={})
    install_completion(runner,provider,path,tmp_path)
    assert runner.merge_overrides({key:old,'2330':{'pay_date':'2024-01-02'}},provider.corporate_overrides)=={
        key:new,'2330':{'pay_date':'2024-01-02'}}
    altered=dict(old,pending_confirmed_through='2026-09-10')
    with pytest.raises(ValueError,match='Pending terms changed'):
        runner.merge_overrides({key:altered},provider.corporate_overrides)
    with pytest.raises(ValueError,match='conflicting ordinary terms'):
        original({key:old},provider.corporate_overrides)
