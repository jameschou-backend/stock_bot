from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_current_dashboard_exposes_read_only_broker_receipt_entry():
    page=Path(__file__).resolve().parents[1]/'app/dashboard_v2/pages/13_券商紀錄核對.py'
    app=AppTest.from_file(str(page)).run()
    assert not app.exception
    assert app.title[0].value=='券商紀錄核對'
    assert any('請勿補填猜測' in c.value for c in app.caption)
    assert any('中文逐筆填寫' in r.options for r in app.radio)
    assert any('期初' in e.label for e in app.expander) or app.get('form')
