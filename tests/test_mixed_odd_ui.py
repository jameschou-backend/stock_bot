import pytest
from app.mixed_odd_ui import load
from scripts.research_mixed_odd import CASES,analyze
from scripts.research_exit_scenarios import write,sha


def publication(root):
    results={name:dict(completed=False,summary=None,reason='source missing') for name in CASES}
    cases={}
    for name,result in results.items():
        path=root/(name+'.json');write(path,result)
        cases[name]=dict(result,config=CASES[name],result=dict(path=path.name,sha256=sha(path)))
    return dict(schema='mixed_odd_v1',odd_tick_verified=False,odd_execution_evidence='daily_envelope_estimate',offline_identical=True,compared_cases=4,cases=cases,
        analysis=analyze(results,{}),controls={},source_sha256={},all_completed=False,network_calls=0,
        live_qualified=False,unseen_validation=False,opening_auction_inferred=True,cancellation_latency_verified=False)


def save(root,value):
    path=root/'report.json';write(path,value);path.with_suffix('.sha256').write_text(sha(path));return path


def test_blocked_report_must_remain_blocked(tmp_path):
    value=publication(tmp_path)
    assert load(save(tmp_path,value),tmp_path)==value
    value['cases']['strategy_normal']['summary']={'total_return':1.}
    with pytest.raises(ValueError):load(save(tmp_path,value),tmp_path)


@pytest.mark.parametrize('field,value',[('odd_tick_verified',True),('live_qualified',True),('opening_auction_inferred',False),('cancellation_latency_verified',True)])
def test_limitations_cannot_be_hidden(tmp_path,field,value):
    report=publication(tmp_path);report[field]=value
    with pytest.raises(ValueError,match='零股日估算'):load(save(tmp_path,report),tmp_path)


def test_missing_case_and_fabricated_analysis_rejected(tmp_path):
    value=publication(tmp_path);value['cases'].pop('benchmark_stress')
    with pytest.raises(ValueError):load(save(tmp_path,value),tmp_path)
    value=publication(tmp_path);value['analysis']['strategy_normal']['excess_return']=7
    with pytest.raises(ValueError,match='零股比較'):load(save(tmp_path,value),tmp_path)
