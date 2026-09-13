import json
import hashlib
import pytest
from streamlit.testing.v1 import AppTest
from app.contingent_portfolio_ui import load_report


def test_missing_crossday_report_shows_no_performance(tmp_path):
    app=AppTest.from_string('from pathlib import Path\nfrom app.contingent_portfolio_ui import render\nrender(Path('+repr(str(tmp_path/'missing.json'))+'))').run()
    assert not app.exception and not app.dataframe


def test_incomplete_portfolio_cannot_show_full_return(tmp_path):
    p=tmp_path/'r.json';identity=tmp_path/'identity.json';identity.write_text('{}')
    r=dict(scope='crossday_contingent_research',live_qualified=False,network_calls=0,code_sha256={},sources_sha256={},
        identity_path=identity.name,identity_sha256=hashlib.sha256(identity.read_bytes()).hexdigest(),
        cases={'strategy':dict(completed=False,total_return=6.37)})
    p.write_text(json.dumps(r));p.with_suffix('.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='完整期間'):load_report(p,tmp_path)


def test_changed_report_is_not_displayed(tmp_path):
    p=tmp_path/'r.json';p.write_text('{}');p.with_suffix('.sha256').write_text('old')
    with pytest.raises(ValueError,match='報告已變動'):load_report(p,tmp_path)
