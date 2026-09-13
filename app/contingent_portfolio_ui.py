"""Read-only first-blocker and fixed-order display for full-calendar replay."""
from pathlib import Path
import json
import hashlib
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/forward_simulation/crossday_contingent_delivery_20260914.json'


def load_report(path=REPORT,root=ROOT):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=path.with_suffix('.sha256').read_text().strip():
        raise ValueError('跨日回放報告已變動')
    r=json.loads(path.read_text())
    if r['scope']!='crossday_contingent_research' or r['live_qualified'] is not False or r['network_calls']!=0:
        raise ValueError('跨日研究不能冒稱實戰驗證')
    for name,digest in (r['code_sha256']|r['sources_sha256']).items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('跨日研究程式或來源已變動')
    if hashlib.sha256((root/r['identity_path']).read_bytes()).hexdigest()!=r['identity_sha256']:
        raise ValueError('跨日研究來源清單已變動')
    for c in r['cases'].values():
        if not c['completed'] and c['total_return'] is not None:
            raise ValueError('缺資料的回放不能顯示完整期間報酬')
        if hashlib.sha256((root/c['path']).read_bytes()).hexdigest()!=c['sha256']:
            raise ValueError('跨日帳本已變動')
    return r


def render(path=REPORT):
    if not Path(path).exists():return
    with st.expander('完整期間跨日回放：目前卡在哪一筆？',expanded=True):
        try:r=load_report(path)
        except (OSError,KeyError,ValueError) as exc:
            st.error(str(exc));return
        st.write('已串接隔日計畫、部分成交、複利、股利／配股交付與出場餘單。歷史資料仍需逐筆齊全，才能產生完整績效。')
        st.caption('範圍2022/1/3–2026/9/9，本金100萬元；每一天從實際回放的前日持股和現金重新規劃。')
        rows=[]
        for key,label in [('strategy','成交金額排序'),('benchmark','0050')]:
            c=r['cases'][key];b=c['blocked'] or {}
            missing=b.get('missing',[])
            rows.append({'帳戶':label,'第一個未完成日':b.get('date','已完成'),
                '原因':', '.join(m['stock_id']+' '+('零股逐筆' if m['channel']=='odd' else '整張逐筆') for m in missing) or b.get('reason',''),
                '完整期間報酬':'未提供' if c['total_return'] is None else f"{c['total_return']:.2%}"})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        key=st.selectbox('查看缺件當日的事前委託',['strategy','benchmark'],
            format_func=lambda k:'成交金額排序' if k=='strategy' else '0050',key='crossday_case')
        c=r['cases'][key]
        if c['first_plan']:
            frame=pd.DataFrame([{'股票':p['stock_id'],'方向':'買進' if p['side']=='buy' else '賣出',
                '盤別':'零股' if p['channel']=='odd' else '整張','股數':p['qty'],
                '事前限價':p['limit_cents']/100,'訊號日期':p['signal_date']} for p in c['first_plan']['spec']['plans']])
            st.dataframe(frame,hide_index=True,use_container_width=True)
            st.download_button('下載跨日缺件與固定委託',json.dumps(c,ensure_ascii=False,indent=2).encode(),
                key+'-crossday-blocker.json','application/json',key='crossday_download')
        st.info('缺資料的整天未計入帳本；策略目前只完成空手起始日，尚無已驗證的買進。這份計畫不是真實下單指示。')
