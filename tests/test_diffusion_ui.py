import copy
import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from app import diffusion_research_ui as ui


def test_diffusion_ui_explains_causal_events_and_switches_saved_results(monkeypatch,tmp_path):
    root = Path(__file__).resolve().parents[1]
    report = {**json.loads((root/'docs/research_diffusion_20260910.json').read_text()),'available':True}
    monkeypatch.setattr(ui,'overview',lambda:report)
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    cache=tmp_path/'.cache/diffusion-research';cache.mkdir(parents=True)
    pd.DataFrame({'stock_id':['2330'],'name':['台積電']}).to_parquet(cache/'companies.parquet',index=False)
    app=AppTest.from_string('from app.diffusion_research_ui import render\nrender()').run(timeout=15)
    assert not app.exception
    table=next(d.value for d in app.dataframe if '累積試算淨報酬' in d.value)
    assert len(table)==5 and '可疑估值筆數' in table
    assert any('先找突破、放量' in m.value for m in app.markdown)
    assert any('名冊' in w.value for w in app.warning)
    for choice in ('官方價格・再晚一天買','官方訊號固定・換舊價格對帳','舊價格與舊訊號・基本成本'):
        app.selectbox(key='diffusion_scenario').set_value(choice).run()
        assert not app.exception
    row=next(r for r in report['results'] if (r['rule'],r['basis'],r['signal_basis'],r['scenario'],r['delay'])
             ==('follower_after','snapshot','snapshot','base',0))
    assert app.metric[0].value==ui.pct(row['summary']['total_return'])
    app.selectbox(key='diffusion_event_status').set_value('10 日內未接力').run()
    event_table=next(d.value for d in app.dataframe if '結果' in d.value and '領先日' in d.value)
    assert set(event_table['結果'])=={'10 日內未接力'}
    app.selectbox(key='diffusion_trade_method').set_value('領先出現就買').run()
    assert not app.exception
    assert any('領先出現就買收到' in m.value for m in app.markdown)
    changed=copy.deepcopy(report)
    target=next(r for r in changed['results'] if (r['rule'],r['basis'],r['signal_basis'],r['scenario'],r['delay'])
                ==('follower_after','snapshot','snapshot','base',0))
    target['valuation_audit']['finding_count']=1
    target['summary']['final_liquidation_complete']=False
    monkeypatch.setattr(ui,'overview',lambda:changed)
    app.run()
    assert any('1 筆持有估值疑點' in e.value for e in app.error)
    assert any('未平倉估值' in w.value for w in app.warning)
    monkeypatch.setattr(ui,'overview',lambda:{'available':False,'note':'來源已變更，請重新研究'})
    app.run()
    assert not app.metric and app.info[0].value=='來源已變更，請重新研究'
