"""Expose the existing receipt audit in the current dashboard, without orders."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.manual_execution_ui import render

st.set_page_config(page_title='券商紀錄核對｜台股投資工作台', layout='wide')
st.title('券商紀錄核對')
st.write('用既有的大戶投成交與庫存紀錄，檢查股數、費用、委託時序及可用買進額度。')
st.info('先準備同一交易日的期初庫存、成交明細、期末庫存及可用額度；'
        '有撤單或待交付配股時，再附對應紀錄。帳號與個資可遮蔽。')
st.caption('只有歷史成交清單時，可以先核對成交與費用；不足以證明當時可用額度、'
           '撤單先後或事前訊號。請勿補填猜測的時間或金額。')
st.caption('餘股釋放名額仍是研究結果；這裡沿用保守規則，剩餘庫存及待交付配股都占名額。')
render()
