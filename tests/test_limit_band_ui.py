import pytest
from app.limit_band_ui import load
from scripts.research_limit_bands import CASES,analyze
from scripts.research_exit_scenarios import write,sha


def publication(root):
    results={name:dict(completed=False,summary=None,reason='source missing') for name in CASES}
    cases={}
    for name,result in results.items():
        path=root/(name+'.json');write(path,result)
        cases[name]=dict(result,config=CASES[name],result=dict(path=path.name,sha256=sha(path)))
    return dict(schema='limit_bands_v1',offline_identical=True,compared_cases=12,cases=cases,
        analysis=analyze(results),source_sha256={},all_completed=False,network_calls=0,
        live_qualified=False,unseen_validation=False)


def save(root,value):
    path=root/'report.json';write(path,value);path.with_suffix('.sha256').write_text(sha(path));return path


def test_blocked_comparison_visible_without_fabricated_profit(tmp_path):
    value=publication(tmp_path)
    assert load(save(tmp_path,value),tmp_path)==value
    value['live_qualified']=True
    with pytest.raises(ValueError,match='重播證據'):load(save(tmp_path,value),tmp_path)


def test_report_cannot_change_paired_benchmark_or_drop_failed_case(tmp_path):
    value=publication(tmp_path);value['cases'].pop('strategy_3_stress')
    with pytest.raises(ValueError,match='重播證據'):load(save(tmp_path,value),tmp_path)
    value=publication(tmp_path);value['cases']['strategy_3_stress']['config']=CASES['strategy_0_stress']
    with pytest.raises(ValueError,match='組別'):load(save(tmp_path,value),tmp_path)


def test_analysis_is_recomputed_from_sealed_results(tmp_path):
    value=publication(tmp_path);value['analysis']['strategy_0_normal']['excess_return']=10.
    with pytest.raises(ValueError,match='跨組比較'):load(save(tmp_path,value),tmp_path)
