import hashlib
import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from app.intraday_limit_ui import load_case,load_report


def test_missing_research_does_not_show_a_return(tmp_path):
    path=tmp_path/'missing.json'
    at=AppTest.from_string('from pathlib import Path\nfrom app.intraday_limit_ui import render\nrender(Path('+repr(str(path))+'))').run()
    assert not at.exception and not at.error and at.info
    assert not at.dataframe


def test_changed_case_is_rejected(tmp_path):
    path=tmp_path/'case.json';path.write_text('{"account":{}}')
    item={'path':'case.json','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert load_case(item,tmp_path)=={}
    path.write_text('{"account":{"fake":true}}')
    with pytest.raises(ValueError,match='帳本已變動'):load_case(item,tmp_path)


def test_case_cannot_escape_research_root(tmp_path):
    with pytest.raises(ValueError,match='路徑不合法'):
        load_case({'path':'../elsewhere.json','sha256':''},tmp_path)


def test_unverified_report_cannot_present_performance(tmp_path):
    report=dict(completed=True,live_qualified=False,offline={'identical':False,'requests':0},cases={})
    path=tmp_path/'report.json';path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='離線驗證'):load_report(path,tmp_path)
