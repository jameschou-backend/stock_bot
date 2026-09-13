import json
from streamlit.testing.v1 import AppTest


def app(path):
    from pathlib import Path
    from app.five_axis_ui import render
    render(Path(path))


def report():
    return dict(offline_identical=True,research_code_sha256={},offline_seconds=20,
        caption='基準3檔，分散組5檔',warning='時間假設研究',
        comparison={'研究進度':[{'狀態':'仍待歷史母體'}]},conclusion='尚未取得實盤資格',remaining=[],
        future_revenue_forecast={'frozen_at':'2026-09-13T00:00:00Z','revenue_month':'2026-09',
            'rows':[{'stock_id':'2330','name':'台積電','expected_revenue_twd':100000000},
                    {'stock_id':'1101','name':'台泥','expected_revenue_twd':None}]})


def test_forecast_search_preserves_missing_values_and_separates_from_returns(tmp_path):
    path=tmp_path/'report.json';path.write_text(json.dumps(report()))
    at=AppTest.from_function(app,args=(str(path),)).run()
    assert not at.exception and not at.error
    assert any('分散組5檔' in c.value for c in at.caption)
    at.text_input(key='five_axis_forecast_filter').set_value('1101').run()
    assert not at.exception
    frame=at.dataframe[-1].value
    assert len(frame)==1 and frame.iloc[0]['狀態']=='歷史資料不足'
    assert frame['預估營收（億元）'].isna().all()


def test_invalid_evidence_hides_forecasts_too(tmp_path):
    path=tmp_path/'report.json';data=report();data['offline_identical']=False
    path.write_text(json.dumps(data))
    at=AppTest.from_function(app,args=(str(path),)).run()
    assert not at.exception and at.error
    assert not at.dataframe and not at.text_input
