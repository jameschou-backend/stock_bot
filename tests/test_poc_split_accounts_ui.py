from copy import deepcopy
import pytest

from app.poc_split_accounts_ui import validate, case_detail, ARMS
from skills.top3_validation import MODES


@pytest.fixture
def report():
    cases={}
    for mode in MODES:
        configs=[(a,v) for a in ARMS for v in ('original','split50')]+[('benchmark','original'),('combined','combined')]
        for arm,variant in configs:
            initial=500000 if variant=='split50' else 1000000
            ret=1.0 if variant=='original' else .8
            metrics=dict(initial_cash=initial,trading_days=666,total_return=ret,max_drawdown=-.3,final_nav=initial*(1+ret),profit=initial*ret)
            cases[f'{arm}:{variant}:{mode}']=dict(id=arm,variant=variant,mode=mode,completed=True,
                metrics=metrics,summary=deepcopy(metrics),excess_return=ret-1,
                funding_audit=dict(initial_cash=500000,independent_cash=True,rebalance=False,scaled_original_result=False),
                aggregation_audit=dict(passed=True,days=666,cash_transfers=False,rebalance=False,method='sum_independent_nav_recompute_peak'),
                shared_capacity=dict(capacity_collision_count=0,collisions=[],joint_capacity_pass=True,actual_fill_verified=False))
    comparisons={}
    for mode in MODES:
        new=cases[f'combined:combined:{mode}']['metrics'];comparisons[mode]={}
        for arm in (*ARMS,'benchmark'):
            old=cases[f'{arm}:original:{mode}']['metrics']
            comparisons[mode][arm]=dict(return_change=new['total_return']-old['total_return'],return_sacrifice=old['total_return']-new['total_return'],
                drawdown_improvement=0.0,drawdown_relative_reduction=0.0,profit_change=new['profit']-old['profit'])
    return dict(schema='poc_split_accounts_publication_v1',start='2024-01-02',end='2026-10-02',initial_cash=1000000,sleeve_initial_cash=500000,
        position_count=6,slots_per_sleeve=3,rebalance=False,cash_transfers=False,live_qualified=False,paper_enabled=False,actual_fill_verified=False,unseen_validation=False,
        modes={k:dict(commission=v[0],slippage=v[1]) for k,v in MODES.items()},expected_new_accounts=8,completed_new_accounts=8,reused_accounts=12,cases=cases,comparisons=comparisons)


def test_exact_scope_is_accepted(report):
    assert validate(report) is report


@pytest.mark.parametrize('mutation',['missing','capital','cost','comparison','promotion','rebalance','nan','capacity','funding','scaled'])
def test_invalid_scope_or_false_improvement_is_rejected(report,mutation):
    row=report['cases']['combined:combined:baseline']
    if mutation=='missing': del report['cases']['poc_priority:split50:combined']
    if mutation=='capital': row['metrics']['initial_cash']=2000000
    if mutation=='cost': report['modes']['combined']['slippage']=.0045
    if mutation=='comparison': report['comparisons']['baseline']['poc_up_red']['return_sacrifice']=0
    if mutation=='promotion': report['paper_enabled']=True
    if mutation=='rebalance': report['rebalance']=True
    if mutation=='nan': row['metrics']['total_return']=float('nan')
    if mutation=='capacity': row['shared_capacity']['capacity_collision_count']=1
    if mutation=='funding': report['cases']['poc_up_red:split50:baseline']['funding_audit']['independent_cash']=False
    if mutation=='scaled': report['cases']['poc_up_red:split50:baseline']['funding_audit']['scaled_original_result']=True
    with pytest.raises(ValueError): validate(report)


def test_details_fail_closed_on_escape_or_wrong_hash(report,tmp_path):
    row=report['cases']['combined:combined:baseline']
    row.update(path='../secret.json',sha256='0'*64)
    with pytest.raises(ValueError,match='路徑'): case_detail(report,'combined__combined__baseline',tmp_path)
    row['path']='account.json';(tmp_path/'account.json').write_text('{}')
    with pytest.raises(ValueError,match='指紋'): case_detail(report,'combined__combined__baseline',tmp_path)
