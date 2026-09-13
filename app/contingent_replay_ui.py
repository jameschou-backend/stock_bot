"""Show adapter progress without relabelling a one-day example as strategy return."""
from pathlib import Path
import json
import hashlib
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/forward_simulation/contingent_replay_delivery_20260914.json'


def load_report(path=REPORT,root=ROOT):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=path.with_suffix('.sha256').read_text().strip():
        raise ValueError('逐筆串接報告已變動')
    report=json.loads(path.read_text())
    if (report['scope']!='contingent_adapter_demonstration' or report['live_qualified'] is not False
            or report['historical_replay_completed'] is not False or report['total_return'] is not None
            or report['missing_historical_odd'] is not True or report['network_calls']!=0):
        raise ValueError('單日串接示範不能宣稱完整歷史績效')
    for name,digest in (report['code_sha256']|report['inputs_sha256']).items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('逐筆串接規則或行情已變動')
    for key,item in report['cases'].items():
        for path_key,digest_key in [('path','sha256'),('plan_path','plan_sha256')]:
            if hashlib.sha256((root/item[path_key]).read_bytes()).hexdigest()!=item[digest_key]:
                raise ValueError('示範計畫或成交帳本已變動')
        if item['completed']!=(key!='missing_odd'):
            raise ValueError('示範與缺件狀態不符')
    return report


def clock(at):
    seconds,micro=divmod(at,1_000_000);hours,seconds=divmod(seconds,3600);minutes,seconds=divmod(seconds,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{micro:06d}'


def render(path=REPORT):
    if not Path(path).exists():return
    with st.expander('逐筆串接進度：固定計畫 → 賣出 → 買進',expanded=True):
        try:report=load_report(path)
        except (OSError,KeyError,ValueError) as exc:
            st.error(str(exc));return
        st.success(f"上一輪換倉清單的整張逐筆已補齊：{report['board_stock_days']}/43個股票日，新增{report['finmind_preparation_requests']}次FinMind請求。")
        st.caption('這是清單所需資料的覆蓋，不是所有股票或完整歷史策略資料已齊全。')
        st.write(report['warning'])
        st.caption('2022/1/10單日整張示範：持有8261共2,000股、現金12.60元；固定賣出2,000股後買2884共6,000股。兩單限價均取前一交易日收盤。')
        rows=[]
        for key,label in [('no_reuse','當日賣款不回用'),('delay_0','假設回報後立即可用'),('delay_1s','假設回報後1秒可用')]:
            fills=report['cases'][key]['fills'];buys=[f for f in fills if f['side']=='buy']
            rows.append({'額度情境':label,'賣出股數':sum(f['qty'] for f in fills if f['side']=='sell'),
                '買進股數':sum(f['qty'] for f in buys),'買單送出時間':clock(buys[0]['sent_at']) if buys else '未送出',
                '買進完成時間':clock(buys[-1]['at']) if buys else '無成交'})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.info('加入8261的693股零股賣單後，因缺零股序列，整日回放會停止且不產生成交。完整原策略逐筆報酬仍未提供。')
        st.download_button('下載單日逐筆串接結果',json.dumps(report,ensure_ascii=False,indent=2).encode(),
                           'contingent-replay-demo.json','application/json',key='contingent_replay_download')
