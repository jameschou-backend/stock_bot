from streamlit.testing.v1 import AppTest
import streamlit as st
from app import forward_ui as ui, forward_simulation_ui, forward_comparison_ui


def test_original_book_can_open_its_own_expanders(monkeypatch):
    monkeypatch.setattr(forward_simulation_ui,'render',lambda *args,**kwargs:None)
    monkeypatch.setattr(forward_comparison_ui,'render',lambda:None)
    monkeypatch.setattr(ui.journal,'summary',lambda:dict(prospective_days=0,orders=0,confirmed_fills=0,rows=[]))
    def portfolio():
        with st.expander('原始帳本內層面板'):
            st.write('原始資料保留')
    monkeypatch.setattr(ui,'render_portfolio',portfolio)
    app=AppTest.from_string('from app.forward_ui import render\nrender()').run()
    assert not app.exception
    app.checkbox(key='show_original_forward_books').check().run()
    assert not app.exception
    assert any(x.value=='原始資料保留' for x in app.markdown)
