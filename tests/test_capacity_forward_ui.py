from streamlit.testing.v1 import AppTest
from tests.test_capacity_forward import prepare
from app import capacity_forward as cap


def test_three_books_and_searchable_plans_render_without_mutating_books(tmp_path):
    root,_,_=prepare(tmp_path)
    before={role:cap.verify(root/(role+'.sqlite3')) for role in cap.ROLES}
    at=AppTest.from_string('from pathlib import Path\nfrom app.capacity_forward_ui import render\nrender(Path('+repr(str(root))+'))').run()
    assert not at.exception and not at.error
    assert len(at.metric)==3
    assert any('前向模擬' in w.value for w in at.warning)
    at.selectbox(key='capacity_role').set_value('control').run()
    assert not at.exception
    assert {role:cap.verify(root/(role+'.sqlite3')) for role in cap.ROLES}==before


def test_ui_distinguishes_fresh_at_fill_from_current_expiry(tmp_path,monkeypatch):
    from datetime import timedelta
    from tests.test_capacity_source_guard import setup,quote
    from app import capacity_source_guard as guard,forward_corporate_audit as corporate
    root,_,evidence,now=setup(tmp_path)
    history=corporate.source_history(evidence)
    monkeypatch.setattr(corporate,'source_history',lambda evidence_path=corporate.PATH:history)
    guard.match(root,'strategy',quote(),lambda:now,evidence)
    guard.match(root,'strategy',quote(20,21000),lambda:now+timedelta(seconds=20),evidence)
    before=cap.verify(root/'strategy.sqlite3')
    at=AppTest.from_string('from pathlib import Path\nfrom app.capacity_forward_ui import render\nrender(Path('+repr(str(root))+'))').run()
    assert not at.exception and not at.error
    assert any('1／1 筆通過' in m.value for m in at.markdown)
    assert any('目前來源過期' in c.value for c in at.caption)
    assert any('今晚能結算嗎' in m.value for m in at.markdown)
    assert any(b.label == '下載三帳本結算前清單' for b in at.get('download_button'))
    assert cap.verify(root/'strategy.sqlite3')==before
