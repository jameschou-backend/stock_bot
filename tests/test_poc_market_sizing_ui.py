from copy import deepcopy
import hashlib
import json

import pytest

from app.poc_market_sizing_ui import validate, case_detail
from scripts.summarize_poc_market_sizing import compare, configuration
from skills.top3_validation import MODES


@pytest.fixture
def report():
    r=dict(schema='poc_market_sizing_publication_v1',start='2024-01-02',end='2026-10-02',
        initial_cash=1000000,position_count=3,variant='market120_half',completed_new_accounts=4,
        reused_accounts=8,modes={k:dict(commission=v[0],slippage=v[1]) for k,v in MODES.items()},
        actual_fill_verified=False,unseen_validation=False,paper_enabled=False,live_qualified=False,
        rule=dict(market='0050',price='adjusted_close',window_valid_observations=120,weak_condition='close_strictly_below_MA120',weak_new_buy_factor=.5,normal_new_buy_factor=1.,execution_lag_sessions=1,existing_holdings='original_exits_only_no_trim_no_top_up'),
        preregistration=dict(path='spec.json',sha256='x'),cases={},comparisons={},
        candidate_market_overlap=dict(candidate_count=100,candidate_states=dict(normal=90,weak=10,unknown=0),scope='original_red_candidates_before_POC_hard_filter'))
    for i,m in enumerate(MODES):
        for arm,variant in [('poc_up_red','original'),('benchmark','original'),('poc_up_red','market120_half')]:
            ret=(2.4 if variant=='market120_half' else 2.7 if arm=='poc_up_red' else 2.5)-i*.1
            metrics=dict(total_return=ret,max_drawdown=-.3 if variant=='market120_half' else -.4,
                profit=ret*1000000,final_nav=(1+ret)*1000000,trading_days=666)
            r['cases'][f'{arm}:{variant}:{m}']=dict(id=arm,variant=variant,mode=m,completed=True,
                reused=variant=='original',metrics=metrics,summary=deepcopy(metrics),
                market_sizing_audit=dict(passed=True),unchanged_before_first_trigger=True)
        new=r['cases'][f'poc_up_red:market120_half:{m}']['metrics']
        r['comparisons'][m]={k:compare(new,r['cases'][f'{a}:original:{m}']['metrics'])
            for k,a in [('original','poc_up_red'),('benchmark','benchmark')]}
    return r


def test_complete_scope_and_signed_deltas(report):
    assert validate(report) is report
    assert report['comparisons']['baseline']['original']['drawdown_improvement']>0
    assert report['comparisons']['baseline']['original']['return_change']<0


def test_exclusion_counts_are_rebuilt_outcomes_and_policy_stays_a_parameter():
    account=dict(data_gap_exclusions=[dict(excluded_children=[dict(channel='board'),dict(channel='odd')])],
        settings=dict(data_gap_policy='unchanged',excluded_event_count=1,excluded_board_children=1,excluded_odd_children=1))
    assert configuration(account)==dict(data_gap_policy='unchanged')
    account['settings']['excluded_board_children']=0
    with pytest.raises(ValueError,match='actual excluded orders'):configuration(account)


@pytest.mark.parametrize('fault',['missing','qualification','wrong_cost','summary','audit','rule','reuse','overlap'])
def test_reject_incomplete_or_mismatched_publication(report,fault):
    row=report['cases']['poc_up_red:market120_half:baseline']
    if fault=='overlap':report['candidate_market_overlap']['candidate_count']+=1
    if fault=='missing':report['cases'].pop('benchmark:original:fee2')
    if fault=='qualification':report['actual_fill_verified']=True
    if fault=='wrong_cost':report['comparisons']['baseline']=report['comparisons']['fee2']
    if fault=='summary':row['summary']['final_nav']+=100
    if fault=='audit':row['market_sizing_audit']['passed']=False
    if fault=='rule':report['rule']['execution_lag_sessions']=0
    if fault=='reuse':report['cases']['poc_up_red:original:baseline']['reused']=False
    with pytest.raises(ValueError):validate(report)


def test_detail_checks_actual_bytes_and_same_summary(report,tmp_path):
    row=report['cases']['poc_up_red:market120_half:baseline']
    path=tmp_path/'account.json'
    data=dict(completed=True,live_qualified=False,actual_fill_verified=False,summary=row['summary'],
        risk_controls=dict(arm='poc_up_red',mode='baseline',variant='market120_half',preregistration=report['preregistration']),
        account=dict(daily=[dict(date=report['start'])]*665+[dict(date=report['end'])],trades=[],
            settings=dict(initial_cash=1000000,commission=MODES['baseline'][0],slippage=MODES['baseline'][1])))
    path.write_text(json.dumps(data));row.update(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    detail=case_detail(report,'poc_up_red__market120_half__baseline',tmp_path)
    assert detail['summary']==row['summary']
    path.write_text('{}')
    with pytest.raises(ValueError,match='指紋'):case_detail(report,'poc_up_red__market120_half__baseline',tmp_path)


def test_completed_case_is_reused_and_cannot_be_rerun(tmp_path,monkeypatch):
    from scripts import research_poc_market_sizing as runner
    from skills.all_entry_accounts import write,digest
    monkeypatch.setattr(runner,'BASE',tmp_path)
    dest=tmp_path/'run-v1'/'baseline'/'account'
    report=dict(all_completed=True,risk_controls=dict(variant='market120_half',mode='baseline',
        preregistration=dict(sha256=digest(runner.SPEC))))
    path=dest/'report.json';sha=write(path,report);path.with_suffix('.sha256').write_text(sha+'\n')
    assert runner.run('baseline',dest)==report
    with pytest.raises(ValueError,match='Completed configuration'):
        runner.run('baseline',tmp_path/'run-v2'/'baseline'/'account')
