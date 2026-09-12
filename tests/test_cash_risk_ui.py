import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def app(path):
    from pathlib import Path
    from app.cash_risk_ui import render
    render(Path(path))


def test_unfinished_research_cannot_display_performance(tmp_path):
    at = AppTest.from_function(app, args=(str(tmp_path/'missing.json'),)).run()
    assert not at.exception
    assert '尚未完成封存' in at.info[0].value
    assert not at.dataframe


def test_unverified_or_changed_evidence_is_blocked(tmp_path):
    path = tmp_path/'report.json'
    path.write_text(json.dumps(dict(research_code_sha256={}, offline_identical=False)))
    at = AppTest.from_function(app, args=(str(path),)).run()
    assert not at.exception and at.error
    assert not at.dataframe


def test_simple_table_switches_without_triggering_research(tmp_path):
    path = tmp_path/'report.json'
    path.write_text(json.dumps(dict(research_code_sha256={}, offline_identical=True,
        comparison={'成交壓力':[{'情境':'原現金版','報酬':'+541.90%'}],
                    '整戶減碼':[{'情境':'測試規則','報酬':'+100.00%'}]},
        conclusion='保留研究資格限制', remaining=['歷史股票名冊尚待重建'], offline_seconds=12)))
    at = AppTest.from_function(app, args=(str(path),)).run()
    assert not at.exception and at.warning
    at.selectbox[0].select('整戶減碼').run()
    assert not at.exception
    assert at.dataframe[0].value.iloc[0]['情境'] == '測試規則'
