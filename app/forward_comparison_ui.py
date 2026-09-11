"""Aligned evidence first; no scorecard while either book is incomplete."""
import pandas as pd
import streamlit as st
from app import forward_comparison as c, forward_halts as h, forward_portfolio as p


def render():
    st.subheader('策略與0050：同本金前向比較')
    st.caption('比較固定使用兩份原始帳本；上方選取的更正版本不會替換原始前向績效。')
    try: result=h.compare(p.PATH,c.BENCHMARK)
    except (ValueError,OSError) as exc:
        st.error('比較證據未通過檢查：'+str(exc));return
    if not result['ready']:
        reasons=[r.replace('benchmark','0050比較帳本').replace('strategy','策略帳本') for r in result['reasons']]
        st.info('尚不能比較報酬：'+'；'.join(reasons))
        return
    cols=st.columns(3)
    cols[0].metric('策略紙上報酬',f"{result['strategy_return']:.2%}")
    cols[1].metric('0050紙上報酬',f"{result['benchmark_return']:.2%}")
    cols[2].metric('策略領先／落後',f"{result['excess_percentage_points']:+.2f} 個百分點")
    st.caption(f"共同開始交易日：{result['entry_session']}；結算至{result['as_of']}。{result['note']}")
    frame=pd.DataFrame(result['points']).set_index('date').astype(float)
    st.line_chart(frame.rename(columns={'strategy_nav':'策略資產','benchmark_nav':'0050資產'}))
