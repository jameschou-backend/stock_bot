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
