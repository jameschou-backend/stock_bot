from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest

from skills.all_entry_official_execution_v2 import (
    OfficialRegularDay, STATUS, WINDOW, MANIFEST_SHA, CompletedBoardEvidence,
    load_official_completion, match_board, official_auditor, execution_mixin,
    install_official_execution,
)
from skills.poc_range_execution import match_range_board, RangeGapOrders
from skills.all_entry_audit_index import private_auditor
from test_poc_executable_replay import audit_case, tape  # noqa: F401
from test_poc_gap_audit import gap_case  # noqa: F401
from test_poc_range_audit import range_case, range_gap_case  # noqa: F401
from test_poc_gap_execution import engine as gap_engine, Engine, DAY, SID, EID, execute  # noqa: F401
from test_poc_range_execution import engine, intraday  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/all_entry_official_execution_v2_20261011.json'
import json
MANIFEST_SHA = json.loads(DOC.read_text())['manifest_sha256']


def aggregate(shares=200000):
    return OfficialRegularDay('2330','2024-01-03','TWSE',shares,10000,10100,10000,10100,
                              shares*10050,'fixture-proof')


@pytest.mark.parametrize('side,fraction,limit', [('buy',.5,110),('buy',.7,110),('sell',.3,90),
    ('sell',.5,90),('buy',.5,100.5),('sell',.5,100.5)])
@pytest.mark.parametrize('adv', [99999,199999,1_000_000])
def test_same_scope_real_prints_and_aggregate_have_identical_price_capacity(side,fraction,limit,adv):
    raw=tape([('09:00:00',100.,100000),('13:32:00',101.,100000)])
    expected=match_range_board(raw,side,limit,3000,adv,fraction=fraction)
    actual=match_board(aggregate(),side,limit,3000,adv,fraction=fraction)
    for k in expected:
        if k not in ('evidence_status','source_window'): assert actual[k] == expected[k]
    assert actual['evidence_status'] == STATUS and actual['source_window'] == WINDOW
    assert not hasattr(aggregate(),'itertuples') and not hasattr(aggregate(),'time')
    assert actual['actual_fill_verified'] is False


@pytest.fixture
def official_case(range_case):
    account,ticks,*_ = range_case
    value=aggregate()
    ticks.get=lambda *a:(value,'fixture-proof')
    for row in (account['orders'][0],account['trades'][0]):
        row.update(ticks_sha256=None,official_daily_sha256='fixture-proof',
                   evidence_status=STATUS,source_window=WINDOW)
    return range_case


def test_independent_auditor_rebuilds_aggregate_without_a_tick_sequence(official_case):
    result=official_auditor()(*official_case)
    assert result['fills_reconciled'] == 2
    assert result['actual_fill_verified'] is False and result['board_tick_verified'] is False


@pytest.mark.parametrize('change',['quantity','volume','source','fake_tick','status','window','trade_proof','hidden_fee'])
def test_aggregate_tampering_cannot_become_an_audited_fill(official_case,change):
    a=official_case[0];row=a['orders'][0]
    if change=='quantity':row['filled_qty']+=1000
    elif change=='volume':row['source_volume']+=1000
    elif change=='source':row['official_daily_sha256']='wrong'
    elif change=='fake_tick':row['ticks_sha256']='invented-ticks'
    elif change=='status':row['evidence_status']='parent_quality_checked_regular_tape'
    elif change=='window':row['source_window']='guessed-09:00'
    elif change=='trade_proof':a['trades'][0]['official_daily_sha256']='wrong'
    else:a['trades'][0]['commission']=0
    with pytest.raises(ValueError):official_auditor()(*official_case)


@pytest.mark.parametrize('fixture',['range_case','range_gap_case'])
def test_noncompletion_accounts_keep_the_original_independent_audit(request,fixture):
    args=request.getfixturevalue(fixture)
    expected=private_auditor().audit_range_execution(*args)
    actual=official_auditor()(*args)
    assert all(actual[k]==v for k,v in expected.items())


@pytest.mark.parametrize('missing_odd',[False,True])
def test_real_cash_path_and_whole_order_rollback_are_preserved(engine,missing_odd):
    # Replace the same executor seam used by the frozen engine factory, keeping
    # DataGapOrders ahead of it rather than bypassing its transactional wrapper.
    orders=type('CompletedOrders',(RangeGapOrders,execution_mixin()),{})
    cls=type('CompletedFixture',(orders,Engine),{})
    obj=object.__new__(cls);obj.__dict__.update(deepcopy(engine.__dict__))
    value=replace(aggregate(100000),high_cents=10000,close_cents=10000,amount_cents=1_000_000_000)
    obj.ticks.get=lambda *a:(value,'fixture-proof')
    if missing_odd:
        obj.odd_feeds.get_odd=lambda *a:None
        assert execute(obj)==0
        assert obj.cash==1_000_000 and obj.holdings[SID]['qty']==0
        assert not obj.trades and len(obj.data_gap_exclusions)==1
        assert obj.data_gap_exclusions[0]['failure_stage']=='odd'
    else:
        assert execute(obj)==execute(engine)==1003
        assert obj.cash==engine.cash and obj.holdings==engine.holdings
        assert obj.cash_ledger==engine.cash_ledger
        assert obj.orders[0]['ticks_sha256'] is None
        assert obj.orders[0]['official_daily_sha256']=='fixture-proof'


def test_production_completion_rebuilds_exact_components_and_rejects_quote_conflict():
    if not (ROOT/'.cache/all-entry-accounts-20261011/ordinary-components-v1/manifest.json').exists():
        pytest.skip('Local primary evidence is not shipped in the source checkout')
    class Original:
        refs={}
        queries=[]
        def get(self,*args):return 'original-coordinate'
    provider=CompletedBoardEvidence(Original(),ROOT,DOC)
    value,sha=provider.get('6914','2024-06-06','TWSE')
    assert value.shares==217000 and sha==MANIFEST_SHA
    quote=dict(open=160.5,high=162.,low=158.5,close=160.,volume=226180)
    assert provider.audit_day('6914','2024-06-06','TWSE',value,quote)['synthesized_tick_rows'] is False
    with pytest.raises(ValueError):provider.audit_day('6914','2024-06-06','TWSE',value,dict(quote,high=float('nan')))
    with pytest.raises(ValueError):provider.audit_day('6914','2024-06-06','TWSE',replace(value,shares=226000),quote)
    assert provider.get('2330','2024-06-06','TWSE')=='original-coordinate'


def test_factory_preserves_range_gap_wrapper_and_reuses_bound_odd_without_network():
    if not (ROOT/'.cache/all-entry-accounts-20261011/ordinary-components-v1/manifest.json').exists():
        pytest.skip('Local primary evidence is not shipped in the source checkout')
    runner=SimpleNamespace(_all_entry_indexed_audit=None, engine_types=lambda cls, era:(cls,cls))
    board=SimpleNamespace(refs={}, queries=[])
    bound={}
    provider=SimpleNamespace(board_ticks=board, get_odd=lambda *a,**kw:'unchanged',
        _bind=lambda path,sha:bound.update({str(path):sha}))
    install_official_execution(runner,provider,DOC,ROOT)
    cls,_=runner.engine_types(RangeGapOrders,object)
    assert cls._execute_order is RangeGapOrders._execute_order
    mro=cls.__mro__
    assert mro.index(RangeGapOrders)<next(i for i,c in enumerate(mro) if c.__name__=='OfficialAggregateExecution')
    plan=dict(date='2025-02-06',stock_id='8045',planned_qty=2100,board_qty=2000,odd_qty=100)
    value=provider.get_odd('2025-02-06','8045','TWSE',engine=SimpleNamespace(day_plans={'p':plan}))
    assert value['odd_shares']==22057 and value['actual_fill_verified'] is False and bound
    assert provider.get_odd('2025-02-06','2330','TWSE')=='unchanged'
    with pytest.raises(ValueError):provider.get_odd('2025-02-06','8045','TWSE')


@pytest.fixture
def constituent_table():
    from test_board_tape_reconciliation_v2 import parent
    from skills.board_tape_reconciliation_v2 import basket_requests
    payload=dict(stat='OK', date='20220104', sub='1', stockType='2', buyNo='ABC123',
        selectType='M', title='序號：1 111年01月04日 鉅額交易日成交資訊-股票組合',
        fields=['證券代號','證券名稱','成交價','成交量','成交值'],
        data=[['1560','股票一','10','1000','10000'],['2330','股票二','20','500','10000'],
              ['總計','','','1500','20000']])
    return payload,basket_requests(parent(),'2022-01-04')[0]


def test_observed_constituent_schema_keeps_exact_identity_totals_and_input(constituent_table):
    from skills.all_entry_official_execution_v2 import parse_verified_basket
    payload,request=constituent_table; original=deepcopy(payload)
    actual=parse_verified_basket(payload,request,'2022-01-04')
    assert actual=={'1560':{'shares':1000,'amount_cents':1000000},
                    '2330':{'shares':500,'amount_cents':1000000}}
    assert payload==original


@pytest.mark.parametrize('change',['identity','date','title','fields','count','volume','amount','total','duplicate'])
def test_observed_constituents_cannot_hide_wrong_source_or_totals(constituent_table,change):
    from skills.all_entry_official_execution_v2 import parse_verified_basket
    p,r=constituent_table
    if change=='identity':p['buyNo']='OTHER'
    elif change=='date':p['date']='20220105'
    elif change=='title':p['title']='parent list'
    elif change=='fields':p['fields'][-1]='推估成交值'
    elif change=='count':p['data'].pop(0)
    elif change=='volume':p['data'][0][3]='999'
    elif change=='amount':p['data'][0][4]='9999'
    elif change=='total':p['data'][-1][-1]='19999'
    else:p['data'][1][0]='1560'
    with pytest.raises(ValueError):parse_verified_basket(p,r,'2022-01-04')


def test_production_third_day_excludes_verified_basket_securities():
    if not (ROOT/'.cache/all-entry-accounts-20261011/required-official-v3/execution-manifest.json').exists():
        pytest.skip('Local primary evidence is not shipped')
    values,refs=load_official_completion(ROOT,DOC)
    day=values['8045','2025-03-10','TWSE']
    assert len(values)==3 and day.shares==548000 and day.amount_cents==6884850000
    assert day.source_sha256==MANIFEST_SHA and len(refs)>=50
    assert (day.open_cents,day.high_cents,day.low_cents,day.close_cents)==(13250,13300,12100,12400)
