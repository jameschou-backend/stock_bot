"""Inspect fixed-factor execution sensitivity without running a backtest."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.execution_factorial_ui import render

st.set_page_config(page_title='成交壓力拆解｜台股投資工作台', layout='wide')
render()
