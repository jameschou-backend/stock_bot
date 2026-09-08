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
