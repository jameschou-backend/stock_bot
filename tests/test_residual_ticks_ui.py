import json

import pytest

from app.residual_ticks_ui import load, LABELS
from scripts.research_exit_scenarios import sha,write


def publication(root):
    cases={}
    for name in LABELS:
        path=root/(name+'.json')
        write(path,dict(completed=False,summary=None,reason='missing source',
                        partial_diagnostics=dict(plans=[],orders=[],trades=[])))
        cases[name]=dict(completed=False,summary=None,reason='missing source',
                         result=dict(path=path.name,sha256=sha(path)))
    return dict(schema='residual_ticks_v1',offline_identical=True,baseline_reproduced=True,
                compared_cases=4,live_qualified=False,unseen_validation=False,network_calls=0,
                cases=cases,source_sha256={},all_completed=False)


def save(root,value):
    path=root/'report.json'
    write(path,value);path.with_suffix('.sha256').write_text(sha(path))
    return path


def test_blocked_cases_are_visible_without_performance_or_qualification(tmp_path):
    value=publication(tmp_path)
    assert load(save(tmp_path,value),tmp_path)==value
    value['live_qualified']=True
    with pytest.raises(ValueError,match='重播證據'):
        load(save(tmp_path,value),tmp_path)


def test_blocked_result_cannot_publish_partial_period_return(tmp_path):
    value=publication(tmp_path)
    row=value['cases']['strategy_normal']
    path=tmp_path/row['result']['path']
    result=json.loads(path.read_text())
    row['summary']=result['summary']={'total_return':7.}
    write(path,result);row['result']['sha256']=sha(path)
    with pytest.raises(ValueError,match='資料不足'):
        load(save(tmp_path,value),tmp_path)


def test_modified_sources_block_display(tmp_path):
    value=publication(tmp_path)
    source=tmp_path/'source.txt';source.write_text('fixed')
    value['source_sha256'][source.name]=sha(source)
    path=save(tmp_path,value)
    source.write_text('changed')
    with pytest.raises(ValueError,match='來源已變更'):
        load(path,tmp_path)


def test_unknown_liquidity_in_old_completed_account_withdraws_publication(tmp_path):
    value=publication(tmp_path)
    row=value['cases']['strategy_normal'];path=tmp_path/row['result']['path']
    result=dict(completed=True,summary={},account=dict(orders=[dict(
        failure='missing_previous_price_or_adv',requested_qty=1000)]))
    write(path,result)
    row.update(completed=True,summary={},result=dict(path=path.name,sha256=sha(path)))
    with pytest.raises(ValueError,match='資料未知'):
        load(save(tmp_path,value),tmp_path)
