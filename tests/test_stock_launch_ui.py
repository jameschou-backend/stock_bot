from pathlib import Path
import hashlib
import json
from copy import deepcopy
import pytest
from app.stock_launch_ui import load,case_rows,PUBLICATION


def save(root,path,value):
    target=root/path;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(value))
    return dict(path=path,sha256=hashlib.sha256(target.read_bytes()).hexdigest())


def fixture(root):
    r=dict(schema='stock_launch_v1',completed=True,strategy_net_return=None,live_qualified=False,
           adopted=False,unseen_validation=False,portfolio_returns_computed=False,
           source_sha256={'source':'abc'},artifacts={'events':dict(sha256='123')})
    desc=save(root,'.cache/run1/report.json',r)
    m=dict(report_sha256=desc['sha256'],files=r['artifacts'],source_sha256=r['source_sha256'])
    runs=[save(root,f'.cache/run{i}/manifest.json',m) for i in (1,2)]
    pub=dict(schema='stock_launch_publication_v1',report=desc,reproducibility=dict(passed=True,runs=runs,csv_sha256={'events':'123'}))
    return r,pub


def publish(root,pub):
    d=save(root,PUBLICATION,pub)
    (root/PUBLICATION).with_suffix('.sha256').write_text(d['sha256'])


def test_distinct_verified_reproductions_required(tmp_path):
    r,pub=fixture(tmp_path);publish(tmp_path,pub)
    assert load(tmp_path)==r
    pub['reproducibility']['runs'][1]=pub['reproducibility']['runs'][0];publish(tmp_path,pub)
    with pytest.raises(ValueError,match='兩輪'):load(tmp_path)


@pytest.mark.parametrize('defect',['tamper','promote','detached_report','different_sources'])
def test_rejects_modified_or_misrepresented_research(tmp_path,defect):
    r,pub=fixture(tmp_path)
    if defect=='tamper':(tmp_path/pub['report']['path']).write_text('{}')
    if defect=='promote':
        r['live_qualified']=True;pub['report']=save(tmp_path,pub['report']['path'],r)
    if defect=='detached_report':pub['report']=save(tmp_path,'.cache/other/report.json',r)
    if defect=='different_sources':
        path=pub['reproducibility']['runs'][1]['path'];m=json.loads((tmp_path/path).read_text())
        m['source_sha256']={'source':'different'};pub['reproducibility']['runs'][1]=save(tmp_path,path,m)
    publish(tmp_path,pub)
    with pytest.raises(ValueError):load(tmp_path)


def test_display_keeps_no_event_and_does_not_call_event_return_profit():
    r={'latest_cases':[dict(horizon=20,stock_id='2308',name='台達電',event_found=False)]}
    assert case_rows(r,20)==[{'股票':'2308 台達電','回溯觀察日':'無符合事件'}]
