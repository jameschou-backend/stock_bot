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


def test_theme_overview_requires_complete_retrospective_evidence(monkeypatch,tmp_path):
    import json
    source=Path(__file__).resolve().parents[1]/'docs/research_themes_20260909.json'
    report=json.loads(source.read_text())
    monkeypatch.setattr(service,'ROOT',tmp_path)
    assert not service.theme_research_overview()['available']
    folder=tmp_path/'.cache/theme-research';folder.mkdir(parents=True)
    path=folder/'report.summary.json'
    path.write_text(json.dumps(report))
    assert service.theme_research_overview()['available']
    report['cases'][0].pop('counter')
    path.write_text(json.dumps(report))
    assert not service.theme_research_overview()['available']
    report=json.loads(source.read_text())
    report['results'][-1]=report['results'][0]
    path.write_text(json.dumps(report))
    assert not service.theme_research_overview()['available']
    report=json.loads(source.read_text())
    report['live_qualified']=True
    path.write_text(json.dumps(report))
    assert not service.theme_research_overview()['available']


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
    # UI verification must work in a fresh checkout with no local research cache.
    # Allocation rendering and sealed real-data verification have their own
    # tests; paper-fill actions must not depend on an evolving local replay.
    from app import execution_holder_ui, forward_ui
    monkeypatch.setattr(execution_holder_ui, 'load', lambda folder: None)
    monkeypatch.setattr(forward_ui, 'render', lambda: None)
    # These sealed histories have dedicated UI tests; a cash-ledger interaction
    # must not rehash machine-local research files on each widget rerun.
    import importlib
    for name in ('chip_research_ui', 'exit_research', 'capacity_research_ui',
                 'regime_switch_ui', 'diffusion_research_ui', 'guidance_research_ui'):
        monkeypatch.setattr(importlib.import_module('app.'+name), 'render', lambda: None)
    from app import cash_allocation_research as allocation
    monkeypatch.setattr(allocation,'overview',lambda:{'available':False,
        'note':'測試未準備資金配置快取','source_verification':{'status':'pending'}})
    from app import technical_research as technical
    monkeypatch.setattr(technical,'overview',lambda:{'available':False,
        'note':'測試未準備技術策略快取','source_verification':{'status':'pending'}})
    import json
    theme=json.loads((Path(__file__).resolve().parents[1]/'docs/research_themes_20260909.json').read_text())
    monkeypatch.setattr(service,'theme_research_overview',lambda:{**theme,'available':True})
    from app import news_research as news
    from skills.news_radar import analyze
    nr=analyze([{'stock_id':'2408','title':'南亞科 DDR4 出貨增加','source':'測試媒體','link':'https://example.com',
                 'provider_datetime':'2025-07-01T04:00:00','first_recorded_at':'2026-09-09T04:00:00+00:00'}],
               {'2408':'南亞科'})
    nr.update(available=True,stock_id='2408',stock_name='南亞科',names={'2408':'南亞科'},cutoff='2025-07-10',
              start='2025-04-01',end='2025-07-09',analyzed_at='2026-09-09T04:00:00+00:00',elapsed_seconds=.2,
              evidence_note='標題未核對',time_note='事後回補',
              source={'legacy_latest':'2026-05-25','local_days':[],'coverage_note':'覆蓋未確認'},price_source={'note':'歷史背景'})
    monkeypatch.setattr(news,'overview',lambda mode='scan':nr)
    from app import chain_flow_research as chain
    monkeypatch.setattr(chain,'overview',lambda:{'available':False,'note':'測試未準備族群快取'})
    from app import revenue_research as revenue
    revenue_report=json.loads((Path(__file__).resolve().parents[1]/'docs/research_revenue_20260909.json').read_text())
    monkeypatch.setattr(revenue,'overview',lambda:{**revenue_report,'available':True})
    from app import event_group_research as event_research
    event_report=json.loads((Path(__file__).resolve().parents[1]/'docs/research_event_groups_20260909.json').read_text())
    monkeypatch.setattr(event_research,'overview',lambda:{**event_report,'available':True})
    submitted=[]
    def submit_news(request):
        submitted.append(request)
        return {'job_id':'b'*32}
    monkeypatch.setattr(ui.jobs,'submit',submit_news)
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app/dashboard_v2/main.py')).run(timeout=20)
    assert not app.exception
    assert any('新聞日期核對未通過' in e.value for e in app.error)
    assert not any('模擬累積淨報酬' in d.value for d in app.dataframe)
    element(app.checkbox,'展開程式診斷數字（不能作為策略績效）').set_value(True).run()
    assert not app.exception
    element(app.selectbox,'事件診斷情境').set_value('再晚一天進場').run()
    element(app.radio,'事件持有上限').set_value(126).run()
    assert not app.exception
    expected=next(r for r in event_report['results'] if (r['basis'],r['scenario'],r['delay'],r['horizon'],r['rule'])==('official','stress',1,126,'combined'))
    assert any(ui.percent(expected['summary']['total_return']) in d.value['模擬累積淨報酬'].tolist()
               for d in app.dataframe if '模擬累積淨報酬' in d.value)
    element(app.selectbox,'營收比較情境').set_value('再延後 15 天').run()
    assert not app.exception
    assert any(ui.percent(.6554) in d.value['累積淨報酬'].tolist()
               for d in app.dataframe if '累積淨報酬' in d.value)
    element(app.selectbox,'營收比較情境').set_value('只看上市').run()
    assert not app.exception
    assert any(ui.percent(1.9038) in d.value['累積淨報酬'].tolist()
               for d in app.dataframe if '累積淨報酬' in d.value)
    element(app.button,'更新近 7 天並分析').click().run()
    assert submitted[-1].kind=='news_scan' and submitted[-1].fetch_news
    assert submitted[-1].news_days==7
    element(app.radio,'新聞研究方式').set_value('review').run()
    element(app.button,'重建新聞時間線').click().run()
    assert submitted[-1].kind=='news_review' and not submitted[-1].fetch_news
    assert submitted[-1].news_stock_id=='2408'
    assert not app.exception
    element(app.button,'更新族群資金').click().run()
    assert submitted[-1].kind=='chain_flow' and submitted[-1].fetch_flow
    assert not app.exception
    element(app.selectbox,'查看題材證據與受惠候選').set_value('leo').run()
    element(app.radio,'題材持有方式').set_value('risk_exit').run()
    assert not app.exception
    assert any('低軌衛星' in str(d.value) for d in app.dataframe)
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
