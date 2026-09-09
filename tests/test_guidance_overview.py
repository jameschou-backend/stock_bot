import copy
import hashlib
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import guidance_research as service

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return json.loads((ROOT/'docs/research_guidance_20260910.json').read_text())


def test_sealed_pilot_rejects_modified_sources_inputs_or_missing_controls(monkeypatch,tmp_path):
    report = fixture()
    monkeypatch.setattr(service,'ROOT',tmp_path)
    assert not service.overview()['available']
    def write(name,content):
        path=tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    for key in ('code_sha256','source_sha256'):
        for name in report[key]:
            report[key][name]=write(name,'fixture')
    report['preregistration_sha256']=write('docs/prereg_guidance_20260910.md','spec')
    for name in report['inputs']['files_sha256']:
        report['inputs']['files_sha256'][name]=write('.cache/guidance-research/'+name,'price')
    report['input_manifest_sha256']=write('.cache/guidance-research/inputs.json',json.dumps(report['inputs']))
    name='.cache/guidance-research/report.summary.json'
    write(name,json.dumps(report))
    assert service.overview()['available']
    changed=copy.deepcopy(report); changed['results'][-1]=changed['results'][0]
    write(name,json.dumps(changed)); assert not service.overview()['available']
    changed=copy.deepcopy(report); changed['baselines'].pop()
    write(name,json.dumps(changed)); assert not service.overview()['available']
    changed=copy.deepcopy(report); changed['live_qualified']=True
    write(name,json.dumps(changed)); assert not service.overview()['available']
    changed=copy.deepcopy(report); changed['results'][2]['timing_diagnostic']=False
    write(name,json.dumps(changed)); assert not service.overview()['available']
    write(name,json.dumps(report))
    write('docs/guidance_annual_sources_20260910.json','changed')
    assert not service.overview()['available']
    write('docs/guidance_annual_sources_20260910.json','fixture')
    write('.cache/guidance-research/raw.parquet','changed')
    assert not service.overview()['available']


def test_ui_separates_annual_timing_diagnostics_and_switches_costs_without_backtest(monkeypatch):
    from app import guidance_research_ui as ui
    report={**fixture(),'available':True}
    monkeypatch.setattr(ui,'overview',lambda:report)
    app=AppTest.from_string('from app.guidance_research_ui import render\nrender()').run(timeout=10)
    assert not app.exception
    table=next(x.value for x in app.dataframe if '累積報酬（扣費）' in x.value)
    assert len(table)==4
    assert not table['方法'].str.contains('時間診斷').any()
    app.checkbox[0].set_value(True).run()
    table=next(x.value for x in app.dataframe if '累積報酬（扣費）' in x.value)
    assert len(table)==6 and table['方法'].str.contains('時間診斷').sum()==2
    app.selectbox[0].set_value('再晚一天進場').run()
    assert not app.exception
    target=next(r for r in report['results'] if (r['rule'],r['basis'],r['scenario'],r['delay'])==('beat_confirm','official','stress',1))
    assert app.metric[0].value==ui.pct(target['summary']['total_return'])
    app.selectbox[0].set_value('舊價格・基本成本').run()
    assert not app.exception
    monkeypatch.setattr(ui,'overview',lambda:{'available':False,'note':'資料有變更'})
    app.run()
    assert not app.metric and app.info[0].value=='資料有變更'
