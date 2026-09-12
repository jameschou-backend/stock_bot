from streamlit.testing.v1 import AppTest
from app import forward_simulation as s,forward_simulation_ui as ui
from tests.test_forward_simulation import initialized


def test_simulation_dashboard_is_explicit_read_only_until_button(tmp_path,monkeypatch):
    root,original,bench=initialized(tmp_path);monkeypatch.setattr(s,'ROOT',root)
    from app import forward_corporate_ui,forward_evidence_ui
    monkeypatch.setattr(forward_corporate_ui,'render',lambda *a,**kw:None)
    monkeypatch.setattr(forward_evidence_ui,'render_halts',lambda *a,**kw:None)
    before=s.read(root/'strategy.sqlite3')
    app=AppTest.from_string('from app.forward_simulation_ui import render\nrender()').run()
    assert not app.exception
    assert any('模型推定' in x.value for x in app.caption)
    assert app.button(key='sim_approve_strategy').disabled
    assert s.read(root/'strategy.sqlite3')==before
