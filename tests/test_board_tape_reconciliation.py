from copy import deepcopy
import json

import pandas as pd
import pytest

from skills.board_tape_reconciliation import (
    digest, number, parse_tpex, parse_twse, summarize_ticks, reconcile, verify_report,
)

DAY='2022-01-04'


def tpex():
    return dict(stat='ok',date='20220104',tables=[dict(title='上櫃股票每日收盤行情(不含定價)',date='111/01/04',
        totalCount=1,fields=['代號','成交股數  ',' 成交金額(元)','成交筆數','開盤 ','最高','最低','收盤'],
        data=[['3508','3,000','30,050','3','10.00','10.05','10.00','10.05']])])


def twse():
    base=dict(stat='OK',date='20220104')
    fields=['證券代號','成交股數','成交金額','成交筆數']
    total=dict(fields=fields+['開盤價','最高價','最低價','收盤價'],
        data=[['1560','4210','42130','12','10','10.05','10','10.05']],
        notes=['本統計資訊含一般、零股、盤後定價、鉅額交易，不含拍賣、標購。'])
    return dict(total=dict(base,tables=[total]),
        intraday_odd=dict(base,type='ALL',title='111年01月04日 盤中零股交易行情單',fields=fields,data=[['1560','100','1010','4']]),
        after_odd=dict(base,type='ALL',title='111年01月04日 盤後零股交易行情單',fields=fields,data=[['1560','110','1120','2']]),
        fixed=dict(base,title='111年01月04日盤後定價交易',selectType='ALL',notes=['餘交易單位皆為千股。'],
            fields=['證券代號','成交數量','成交金額','成交筆數'],data=[['1560','1','9950','3']]),
        block_single=dict(base,title='111年01月04日 鉅額交易日成交資訊-單一證券',selectType='S',
            fields=['證券代號','成交股數','成交金額'],data=[]),
        block_basket=dict(base,title='111年01月04日 鉅額交易日成交資訊-股票組合',selectType='M',
            fields=['序號','資料內容'],data=[]))


def raw():
    return pd.DataFrame(dict(date=[DAY]*5,stock_id=['3508']*5,
        deal_price=[10,99,10.05,10.05,10.05],volume=[2,0,1,2,0],
        Time=['09:00:00','10:00:00','13:33:00.123456','14:30:00','14:30:00'],TickType=[1]*5))


def test_tpex_named_units_and_date():
    row=parse_tpex(tpex(),DAY)['3508']
    assert row['shares']==3000 and row['amount_cents']==3005000
    assert row['transaction_count']==3
    assert row['open_cents']==1000 and row['high_cents']==1005


@pytest.mark.parametrize('change', ['date','title','count','duplicate_field','negative','fractional_shares'])
def test_tpex_rejects_ambiguous_or_wrong_sources(change):
    value=tpex();table=value['tables'][0]
    if change=='date':value['date']='20220105'
    elif change=='title':table['title']='上櫃股票行情'
    elif change=='count':table['totalCount']=2
    elif change=='duplicate_field':table['fields'][2]='成交股數'
    elif change=='negative':table['data'][0][1]='-1'
    else:table['data'][0][1]='2.5'
    with pytest.raises((ValueError,KeyError)):parse_tpex(value,DAY)


def test_twse_subtracts_all_sessions_and_fixed_units():
    row=parse_twse(twse(),DAY)['1560']
    assert row['shares']==3000 and row['amount_cents']==3005000
    assert row['transaction_count']==3
    assert row['components']['fixed']['shares']==1000


@pytest.mark.parametrize('component',['intraday_odd','after_odd','fixed','block_single','block_basket'])
def test_missing_component_is_not_zero(component):
    parts=twse();del parts[component]
    with pytest.raises(ValueError,match='Every'):parse_twse(parts,DAY)


def test_nonempty_basket_requires_details():
    parts=twse();parts['block_basket']['data']=[['1',{}]]
    with pytest.raises(ValueError,match='constituent'):parse_twse(parts,DAY)


def test_wrong_fixed_subset_and_missing_units_rejected():
    for key,value in [('selectType','01'),('notes',[])]:
        parts=twse();parts['fixed'][key]=value
        with pytest.raises(ValueError,match='thousand'):parse_twse(parts,DAY)


def test_repeated_block_rows_are_summed_without_inventing_deal_count():
    parts=twse();parts['block_single']['data']=[['1560','1000','10000'],['1560','1000','10000']]
    row=parse_twse(parts,DAY)['1560']
    assert row['shares']==1000 and row['amount_cents']==1005000
    assert row['transaction_count'] is None


def test_negative_residual_and_wrong_total_scope_rejected():
    parts=twse();parts['total']['tables'][0]['data'][0][1]='1'
    with pytest.raises(ValueError,match='Negative'):parse_twse(parts,DAY)
    parts=twse();parts['total']['tables'][0]['notes']=[]
    with pytest.raises(ValueError,match='scope'):parse_twse(parts,DAY)


def test_tick_session_separation_zero_volume_and_delayed_close():
    result=summarize_ticks(raw(),'3508',DAY,'TPEX')
    assert result['shares']==3000 and result['amount_cents']==3005000
    assert result['high_cents']==1005
    assert result['regular_message_rows']==2 and result['fixed_price_rows']==1
    assert result['zero_volume_rows']==2 and result['fixed_price_shares']==2000
    assert result['unknown_session_rows']==0


def test_equal_aggregates_do_not_certify_counts_sequence_or_fill():
    tape=summarize_ticks(raw(),'3508',DAY,'TPEX')
    result=reconcile(tape,parse_tpex(tpex(),DAY)['3508'])
    assert result['same_scope_aggregate_matched']
    assert result['transaction_count_verified'] is False
    assert result['tick_sequence_complete'] is False
    assert result['own_order_fill_proven'] is False
    assert result['accepted_for_strict_replay'] is False


def test_unknown_positive_print_and_amount_corruption_block_match():
    frame=raw();frame.loc[3,'Time']='14:29:00'
    tape=summarize_ticks(frame,'3508',DAY,'TPEX')
    assert reconcile(tape,parse_tpex(tpex(),DAY)['3508'])['status']=='daily_aggregate_conflict'
    tape=summarize_ticks(raw(),'3508',DAY,'TPEX');tape['amount_cents']+=1
    result=reconcile(tape,parse_tpex(tpex(),DAY)['3508'])
    assert set(result['differences'])=={'amount_cents'}


@pytest.mark.parametrize('bad',[True, 'NaN', 'Infinity', '-1', '1.005'])
def test_exact_money_rejects_bad_values(bad):
    with pytest.raises(ValueError):number(bad,cents=True)


def test_report_hash_closure_and_forbidden_promotion(tmp_path):
    source=tmp_path/'source.json';source.write_text('{}')
    report=dict(schema='board_tape_reconciliation_v1',input_sha256={'source.json':digest(source)},code_sha256={},
        strict_data_ready=False,live_qualified=False,own_order_fill_proven=False)
    path=tmp_path/'report.json'
    def save():
        path.write_text(json.dumps(report));path.with_suffix('.sha256').write_text(digest(path))
    save();assert verify_report(path,tmp_path)==report
    source.write_text('[]')
    with pytest.raises(ValueError,match='mismatch'):verify_report(path,tmp_path)
    source.write_text('{}');report['strict_data_ready']=True;save()
    with pytest.raises(ValueError,match='certify'):verify_report(path,tmp_path)


def test_preparation_never_retries_consumed_identity_or_raises_budget(tmp_path,monkeypatch):
    from scripts.audit_board_tape_reconciliation import fetch_tpex
    import scripts.audit_board_tape_reconciliation as cli
    data=dict(cases={'case':dict(ordinary=dict(unverified=[dict(date=DAY,stock_id='3508')],missing=[]))},
        catalog=dict(entries=[dict(date=DAY,stock_id='3508',channel='board',market='TPEX')]))
    tmp_path.joinpath('tpex-fetch-ledger.json').write_text(json.dumps(dict(limit=1,attempts=[DAY],requests=[])))
    class NoNetwork:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get(self,*args,**kwargs):raise AssertionError('Unexpected network request')
    monkeypatch.setattr(cli.requests,'Session',NoNetwork)
    assert fetch_tpex(data,tmp_path,1)['attempts']==[DAY]
    with pytest.raises(ValueError,match='differs'):fetch_tpex(data,tmp_path,2)
    with pytest.raises(ValueError,match='budget'):fetch_tpex(data,tmp_path,251)


@pytest.mark.parametrize('kind',['intraday_odd','after_odd'])
def test_partial_odd_component_cannot_infer_absent_stocks_are_zero(kind):
    parts=twse();parts[kind]['type']='01'
    with pytest.raises(ValueError,match='ALL odd'):parse_twse(parts,DAY)

def test_negative_quarantine_gate_never_qualifies_unknown_or_conflicting_day():
    from skills.board_tape_reconciliation import assert_not_quarantined
    report={'rows':[dict(date=DAY,stock_id='3508',same_scope_aggregate_matched=False)]}
    with pytest.raises(ValueError,match='conflicting'):assert_not_quarantined(report,DAY,'3508')
    with pytest.raises(ValueError,match='missing'):assert_not_quarantined(report,DAY,'9999')
    report['rows'][0]['same_scope_aggregate_matched']=True
    assert assert_not_quarantined(report,DAY,'3508') is None


def test_reconciled_loader_checks_sources_and_keeps_research_only(tmp_path):
    from skills.board_tape_reconciliation import load_reconciled_tape
    source=tmp_path/'tape.parquet';raw().to_parquet(source,index=False)
    row=dict(date=DAY,stock_id='3508',market='TPEX',tape_path='tape.parquet',tape_sha256=digest(source),
             same_scope_aggregate_matched=True)
    report=dict(schema='board_tape_reconciliation_v1',rows=[row],input_sha256={'tape.parquet':digest(source)},
        code_sha256={},strict_data_ready=False,live_qualified=False,own_order_fill_proven=False)
    path=tmp_path/'report.json'
    def save():
        path.write_text(json.dumps(report));path.with_suffix('.sha256').write_text(digest(path))
    save();value=load_reconciled_tape(path,tmp_path,DAY,'3508')
    assert len(value['frame'])==2 and value['frame'].shares.sum()==3000
    assert value['tick_sequence_complete'] is False and value['accepted_for_strict_replay'] is False
    assert value['own_order_fill_proven'] is False
    row['same_scope_aggregate_matched']=False;save()
    with pytest.raises(ValueError,match='conflicting'):load_reconciled_tape(path,tmp_path,DAY,'3508')
    row['same_scope_aggregate_matched']=True;row['tape_sha256']='0'*64;save()
    with pytest.raises(ValueError,match='tape source hash'):load_reconciled_tape(path,tmp_path,DAY,'3508')
