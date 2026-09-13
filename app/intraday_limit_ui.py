"""Read-only historical execution research with downloadable audit journals."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/forward_simulation/intraday_limit_delivery_20260914.json'
LABELS={'original':'原排序','capacity':'成交金額排序','benchmark':'0050基準'}
COLUMNS={'date':'交易日','signal_date':'訊號日','stock_id':'股票代號','name':'名稱',
    'side':'買賣','channel':'交易別','qty':'成交股數','requested_qty':'委託股數',
    'filled_qty':'成交股數','planned_qty':'計畫股數','limit_price':'限價',
    'reference_price':'估計成交價','last_fill_time':'最後估計成交時間',
    'failure':'未成交原因','rejection':'略過原因','reason':'委託原因',
    'reserved_cash':'事前預留現金','total_cost':'總成本','cash_after':'成交後現金',
    'nav':'總資產','cash':'現金','market_value':'持股市值','receivable':'待收股利與股票',
    'drawdown':'距資產高點跌幅','cost':'當日成本'}
REASONS={'historical_odd_tick_unavailable':'缺歷史零股逐筆，剩餘持股保留',
    'partial_trade_through_capacity':'限價內可參與量不足，僅部分成交',
    'no_trade_through_capacity':'沒有足夠穿過限價的成交量',
    'opening_slots_liquidity_cash_or_one_lot':'事前名額、流動性或整張預算不足',
    'missing_previous_price_or_adv':'缺前日價格或均量',
    'leader_entry':'選股訊號進場','loss12':'跌幅達12%出場','time63':'持有63日出場',
    'initial_allocation':'基準初始配置','idle_cash':'基準現金再投入'}


def load_report(path=REPORT,root=ROOT):
    report=json.loads(Path(path).read_text())
    if (not report['completed'] or report['live_qualified'] is not False
            or not report['offline']['identical'] or report['offline']['requests']!=0
            or len(report['cases'])!=6):
        raise ValueError('完整離線驗證尚未通過')
    for name,digest in report['code_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('研究規則已變動，需重新驗證')
    return report


def load_case(item,root=ROOT):
    path=(root/item['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('研究檔案路徑不合法')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item['sha256']:
        raise ValueError('交易帳本已變動，停止顯示')
    return json.loads(raw)['account']


def render(path=REPORT):
    with st.expander('隔日買得到嗎？限價與逐筆成交回測',expanded=True):
        if not Path(path).is_file():
            st.info('逐筆回測準備中；結果須通過完整帳務及離線重跑才會顯示。')
            return
        try:report=load_report(path)
        except (ValueError,KeyError,OSError) as exc:st.error(str(exc));return
        st.caption(report['caption']);st.warning(report['warning'])
        rows=[]
        for ranking,label in LABELS.items():
            a,b=(report['cases'][ranking+'_'+mode]['summary'] for mode in ('normal','stress'))
            rows.append({'策略':label,'一般總報酬':f"{a['total_return']*100:+.2f}%",
                '一般最大回撤':f"{a['max_drawdown']*100:.2f}%",
                '壓力總報酬':f"{b['total_return']*100:+.2f}%",
                '壓力最大回撤':f"{b['max_drawdown']*100:.2f}%"})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        for rule in report['rules']:st.write('• '+rule)
        scenario=st.selectbox('成交條件',['normal','stress'],
            format_func=lambda x:'一般' if x=='normal' else '較低成交量＋較高成本',key='intraday_scenario')
        ranking=st.selectbox('查看策略',list(LABELS),format_func=LABELS.get,key='intraday_ranking')
        try:account=load_case(report['cases'][ranking+'_'+scenario])
        except (ValueError,KeyError,OSError) as exc:st.error(str(exc));return
        daily=pd.DataFrame(account['daily'])
        last=daily.iloc[-1]
        st.metric('期末總資產（本金100萬元）',f"{last['nav']:,.0f} 元")
        odd=[h for h in account['holdings'] if h['date']==last['date'] and h['qty']%1000]
        if odd:
            detail='、'.join(f"{h['stock_id']} 剩餘零股 {h['qty']%1000} 股" for h in odd)
            st.info(detail+'。零股仍計入資產與持股名額，這會影響後續可買的股票。')
        st.line_chart(daily.set_index('date')[['nav','cash']].rename(columns={'nav':'總資產','cash':'現金'}))
        section=st.selectbox('查看明細',['trades','orders','plans','daily'],
            format_func={'trades':'買賣成交','orders':'委託與未成交原因','plans':'前日固定計畫','daily':'每日資產'}.get,
            key='intraday_section')
        frame=pd.DataFrame(account[section])
        query=st.text_input('搜尋股票代號',key='intraday_stock')
        if query and 'stock_id' in frame:frame=frame[frame.stock_id.str.contains(query,regex=False)]
        shown=frame[[k for k in COLUMNS if k in frame]].copy()
        for col in ('failure','rejection','reason'):
            if col in shown:shown[col]=shown[col].map(lambda v:REASONS.get(v,v))
        if 'side' in shown:shown['side']=shown.side.map({'buy':'買進','sell':'賣出'})
        if 'channel' in shown:shown['channel']=shown.channel.map({'board':'整張','odd':'零股'})
        st.dataframe(shown.rename(columns=COLUMNS),hide_index=True,use_container_width=True)
        st.download_button('下載目前明細 CSV',frame.to_csv(index=False).encode('utf-8-sig'),
            ranking+'-'+scenario+'-'+section+'.csv','text/csv',key='intraday_csv')
        st.caption(f"六組帳戶已離線重現；本次離線重跑 {report['offline']['elapsed_seconds']:.1f} 秒、0 次網路請求。")
        for limitation in report['limitations']:st.caption(limitation)
