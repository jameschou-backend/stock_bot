"""Sealed residual-slot study, no trading or background jobs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.residual_slots_ui import render

st.set_page_config(page_title='餘股與持股名額｜台股投資工作台', layout='wide')
render()
