from copy import deepcopy
import json

import pytest

from skills.board_tape_reconciliation_v2 import basket_requests,basket_key,parse_basket_detail,parse_twse_complete,verify_report
from skills.board_tape_reconciliation import digest
from test_board_tape_reconciliation import twse,DAY
from scripts.audit_twse_board_reconciliation_v2 import detail_item


def parent():
    return dict(stat='OK',date='20220104',selectType='M',title='111年01月04日 鉅額交易日成交資訊-股票組合',
        fields=['序號','資料內容','交易別','股票種數','成交總股數','成交總金額'],
        data=[[1,dict(sub=1,stockType='2',buyNo='ABC123',date='20220104'),'配對交易',2,'1500','20000'],
              ['總計','','','','1500','20000']])


def detail():
    return dict(stat='OK',date='20220104',sub='1',stockType='2',buyNo='ABC123',fields=['證券代號','成交股數','成交金額'],
                data=[['1560','1000','10000'],['2330','500','10000'],['總計','1500','20000']])


def test_server_basket_identity_and_parent_totals():
    rows=basket_requests(parent(),DAY)
    assert len(rows)==1 and rows[0]['security_count']==2 and rows[0]['shares']==1500
    parsed=parse_basket_detail(detail(),rows[0],DAY)
    assert parsed['1560']==dict(shares=1000,amount_cents=1000000)
    item=detail_item(DAY,rows[0])
    assert item['params']==dict(sub=1,stockType='2',buyNo='ABC123',date='20220104',response='json')


@pytest.mark.parametrize('change',['date','query_keys','parent_total','duplicate'])
def test_parent_basket_identity_or_totals_invalid(change):
    p=parent()
    if change=='date':p['data'][0][1]['date']='20220105'
    elif change=='query_keys':del p['data'][0][1]['sub']
    elif change=='parent_total':p['data'][-1][-1]='19999'
    else:p['data'].insert(1,deepcopy(p['data'][0]))
    with pytest.raises(ValueError):basket_requests(p,DAY)


@pytest.mark.parametrize('change',['wrong_day','wrong_identity','parent_list','missing_security','wrong_amount','duplicate_security','bad_total'])
def test_constituents_must_equal_dated_parent(change):
    p=detail();request=basket_requests(parent(),DAY)[0]
    if change=='wrong_day':p['date']='20220105'
    elif change=='wrong_identity':p['buyNo']='OTHER'
    elif change=='parent_list':p['selectType']='M'
    elif change=='missing_security':p['data'].pop(0)
    elif change=='wrong_amount':p['data'][0][2]='9999'
    elif change=='duplicate_security':p['data'][1][0]='1560'
    else:p['data'][-1][2]='19999'
    with pytest.raises(ValueError):parse_basket_detail(p,request,DAY)


def test_full_subtraction_includes_basket_and_disallows_missing_details():
    parts=twse();parts['block_basket']=parent()
    parts['total']['tables'][0]['data'][0][1]='5210'
    parts['total']['tables'][0]['data'][0][2]='52130'
    request=basket_requests(parent(),DAY)[0]
    with pytest.raises(ValueError,match='missing'):parse_twse_complete(parts,{},DAY)
    result=parse_twse_complete(parts,{basket_key(request):detail()},DAY)['1560']
    assert result['shares']==3000 and result['amount_cents']==3005000
    assert result['components']['block_basket']['shares']==1000
    assert result['transaction_count'] is None
    assert parts['block_basket']==parent()  # sealed input never mutated


def test_empty_basket_needs_no_guessed_detail():
    parts=twse()
    assert parse_twse_complete(parts,{},DAY)['1560']['shares']==3000
    with pytest.raises(ValueError,match='extra'):parse_twse_complete(parts,{'unknown':detail()},DAY)


def test_v2_report_hash_closure_and_live_gate(tmp_path):
    source=tmp_path/'source';source.write_text('official')
    payload=dict(schema='board_tape_reconciliation_v2',input_sha256={'source':digest(source)},code_sha256={},
                 strict_data_ready=False,live_qualified=False,own_order_fill_proven=False)
    path=tmp_path/'report.json'
    def write():path.write_text(json.dumps(payload));path.with_suffix('.sha256').write_text(digest(path))
    write();assert verify_report(path,tmp_path)==payload
    payload['live_qualified']=True;write()
    with pytest.raises(ValueError,match='trading'):verify_report(path,tmp_path)
    payload['live_qualified']=False;write();source.write_text('changed')
    with pytest.raises(ValueError,match='changed'):verify_report(path,tmp_path)


def test_day_acquisition_finishes_constituents_before_next_day(monkeypatch,tmp_path):
    import scripts.audit_twse_board_reconciliation_v2 as audit
    from scripts.prepare_twse_board_reconciliation_v2 import query,identity,DAY_KIND_ORDER
    cache=tmp_path/'cache';cache.mkdir();(cache/'plan.json').write_text('{}')
    entries=[query(kind,day) for kind in reversed(DAY_KIND_ORDER) for day in ('2022-01-05',DAY)]
    for entry in entries:entry['identity']=identity(entry)
    monkeypatch.setattr(audit,'plan',lambda _:dict(entries=entries,max_http_requests=100))
    calls=[]
    class Client:
        def __init__(self,*args,**kwargs):pass
        def attempts(self):return calls
        def fetch(self,item):calls.append((item['date'],item['kind']));return dict(accepted=True)
    monkeypatch.setattr(audit,'Fetcher',Client)
    def payload(item,receipt):
        p=detail() if item['kind']=='block_detail' else twse()[item['kind']]
        if item['kind']=='block_basket':p=parent()
        p['date']=item['date'].replace('-','')
        if item['kind']=='block_basket':p['data'][0][1]['date']=p['date']
        if item['kind']=='total':
            p['tables'][0]['data'][0][1]='5210';p['tables'][0]['data'][0][2]='52130'
        return p
    monkeypatch.setattr(audit,'receipt_payload',payload)
    result=audit.fetch_complete_days(cache)
    assert calls==[(day,kind) for day in (DAY,'2022-01-05') for kind in (*DAY_KIND_ORDER,'block_detail')]
    assert all(row['status']=='fully_decomposed' for row in result['days'])
