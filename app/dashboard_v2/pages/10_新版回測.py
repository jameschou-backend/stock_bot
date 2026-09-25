"""Open the latest repaired replay without loading unrelated research panels."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.historical_selector_ui import render

st.set_page_config(page_title='新版回測｜台股投資工作台', layout='wide')
st.title('新版回測')
st.caption('查看資料修正的影響、成交壓力與逐筆帳戶；不會執行交易或更新行情。')
render()
