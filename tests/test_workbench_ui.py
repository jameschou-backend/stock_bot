from contextlib import contextmanager
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from streamlit.testing.v1 import AppTest
from app.models import Base,RawPrice
from app.workbench_models import TABLES
from app import workbench_ui as ui,workbench_service as service


def element(elements,label):
    return next(e for e in elements if e.label==label)


def test_overview_marks_missing_or_broken_report_unavailable(monkeypatch,tmp_path):
    monkeypatch.setattr(service,'ROOT',tmp_path)
    assert service.rule_research_overview()['available'] is False
    folder=tmp_path/'.cache/rule-research';folder.mkdir(parents=True)
    for content in ('[]','{"schema":1,"research_only":true,"results":[]}'):
        (folder/'report.json').write_text(content)
        result=service.rule_research_overview()
        assert result['available'] is False and result['live_qualified'] is False
        assert '不完整' in result['note']


def test_flow_overview_requires_all_contrasts_without_triggering_computation(monkeypatch,tmp_path):
    import json
    monkeypatch.setattr(service,'ROOT',tmp_path)
    assert not service.flow_research_overview()['available']
    folder=tmp_path/'.cache/growth-flow-research';folder.mkdir(parents=True)
    report={'schema':1,'experiment':'flow_20260909','research_only':True,'live_qualified':False,
            'source':{},'elapsed_seconds':1,'limitations':[],
            'results':[{'rule':r,'scenario':s} for r in ('price','trust','volume','trust_volume')
                       for s in ('base','stress')]}
    (folder/'report.summary.json').write_text(json.dumps(report))
    assert service.flow_research_overview()['available']
    report['results'][-1]=report['results'][0]
    (folder/'report.summary.json').write_text(json.dumps(report))
    assert not service.flow_research_overview()['available']


def test_create_paper_account_and_record_fill_without_duplicate(monkeypatch,tmp_path):
    engine=create_engine('sqlite://',connect_args={'check_same_thread':False},poolclass=StaticPool)
    Base.metadata.create_all(engine,tables=[*TABLES,RawPrice.__table__])
    @contextmanager
    def session():
        with Session(engine) as s:
            yield s
            s.commit()
    monkeypatch.setattr(ui,'get_session',session)
    monkeypatch.setattr(service,'get_session',session)
    monkeypatch.setattr(ui,'status_data',lambda:{'price_date':'2026-09-08',
        'quota':{'requests_in_window':20,'remaining_requests':5380,'retry_after_seconds':0},
        'problems':[],'data_ready':True,'markets':[],'adjustment_note':'待對帳'})
    monkeypatch.setattr(ui,'candidate_data',lambda:[])
    monkeypatch.setattr(ui.jobs,'JOBS_DIR',tmp_path)
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app/dashboard_v2/main.py')).run(timeout=20)
    assert not app.exception
    element(app.checkbox,'顯示前一輪價格研究（尚未排除上市櫃前行情）').set_value(True).run()
    assert not app.exception
    element(app.number_input,'期初現金（元）').set_value(100000)
    element(app.button,'建立帳本').click().run()
    assert not app.exception
    element(app.text_input,'股票代號').set_value('2330')
    element(app.number_input,'實付手續費').set_value(29)
    element(app.button,'儲存成交紀錄').click().run()
    assert not app.exception
    book=service.portfolio('paper')
    assert len(book['fills'])==1
    assert book['cash']==89971
    assert book['net_pnl'] is None  # No fabricated market quote.
    assert not service.portfolio('real')['initialized']
    assert all(b.label!='儲存成交紀錄' for b in app.button)
    app.run()
    assert len(service.portfolio('paper')['fills'])==1
    element(app.button,'記錄下一筆成交').click().run()
    assert any(b.label=='儲存成交紀錄' for b in app.button)
