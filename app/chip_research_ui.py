"""Read-only chip comparison; rendering cannot fetch data or run experiments."""
from functools import lru_cache
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
from app.exit_research import _read, _file, _stamp
from app.technical_research import REASONS

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('.cache/chip-research')
LABELS = {'control':'原策略', 'trust':'投信買超加速', 'foreign':'外資買超加速',
    'price':'不追高（距20日線≤10%）', 'trust_price':'投信加速＋不追高',
    'foreign_price':'外資加速＋不追高', 'holder':'大戶持股增加', 'margin':'融資減少',
    'holder_margin':'大戶增加＋融資減少', 'holder14_margin':'大戶延後14日＋融資減少',
    'sbl':'借券賣出餘額減少', 'broker':'分點買超集中', 'group_flow':'同族群成交占比增加',
    'sell_full':'法人轉賣後全出', 'sell_half':'法人轉賣後減半'}


def signature():
    manifest_path = ROOT/CACHE/'manifest.json'
    stamp = _stamp(manifest_path)
    meta = _read(manifest_path)
    if meta.get('offline_identical') is not True or meta.get('live_qualified') is not False:
        raise ValueError('籌碼研究還沒有完成離線重現')
    files = meta.get('files_sha256', {})
    required = {str(CACHE/'summary.json'), str(CACHE/'cases/control.json'),
                'scripts/research_chip.py', 'skills/chip_research.py',
                'docs/prereg_chip_20260911.md', '.cache/chip-inputs/manifest.json'}
    if not required.issubset(files):
        raise ValueError('籌碼研究缺少來源索引')
    inventory = tuple((name,digest,_stamp(_file(ROOT,name))) for name,digest in sorted(files.items()))
    if _stamp(manifest_path) != stamp:
        raise ValueError('籌碼來源正在改變')
    return stamp, inventory


@lru_cache(maxsize=2)
def verified(state):
    for name, expected, _ in state[1]:
        digest = hashlib.sha256()
        with _file(ROOT,name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(4*1024*1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError('籌碼研究檔案已變更：'+name)
    summary = _read(ROOT/CACHE/'summary.json')
    if summary.get('live_qualified') is not False or summary.get('unseen_validation') is not False:
        raise ValueError('不正確的研究資格')
    if signature() != state:
        raise ValueError('籌碼來源在驗證中改變')
    return summary


def comparison_rows(summary):
    rows = []
    control = summary['cases']['control']['summary']['total_return']
    benchmark = summary['benchmark']['total_return']
    for mode,label in LABELS.items():
        case = summary['cases'][mode]
        row = {'方法':label,'狀態':('完成，含配股估值假設' if case.get('summary',{}).get('corporate_assumptions') else '完成') if case['completed'] else '證據不足'}
        if case['completed']:
            result = case['summary']
            available = summary['cases'].get('available_'+mode)
            base = available['summary']['total_return'] if available and available['completed'] else None
            row.update({'期末資產（元）':round(result['final_nav']),
                '總淨報酬（%）':round(result['total_return']*100,2),
                '最大回撤（%）':round(result['max_drawdown']*100,2),
                '比0050多（百分點）':round((result['total_return']-benchmark)*100,2),
                '比原策略多（百分點）':round((result['total_return']-control)*100,2),
                '比同資料對照多（百分點）':None if base is None else round((result['total_return']-base)*100,2)})
        else:
            row['原因'] = case['reason']
        rows.append(row)
    return rows


def render():
    st.subheader('籌碼有沒有幫助？')
    if not (ROOT/CACHE/'manifest.json').exists():
        st.info('籌碼研究準備中；完成資料核對與離線重現後才顯示結果。')
        return
    try:
        state=signature()
        summary=verified(state)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        st.warning('暫不顯示籌碼報酬：'+str(exc))
        return
    st.caption('2022/01/03–2026/09/09｜100萬元複利｜可買零股｜三檔＋閒置0050｜扣原稅費、最低手續費及滑價。')
    st.info('這是反覆研究過的歷史。帳戶報酬、同資料對照、逐筆訊號與排序敏感度須一起看；沒有自動啟用新策略。')
    st.dataframe(pd.DataFrame(comparison_rows(summary)),hide_index=True,use_container_width=True)
    options=[m for m in LABELS if summary['cases'][m]['completed']]
    mode=st.selectbox('查看籌碼方法的資產與成交',options,format_func=LABELS.get,key='chip_case')
    case=_read(ROOT/CACHE/'cases'/(mode+'.json'))
    account=case['account']
    if case['summary'].get('corporate_assumptions'):
        st.warning('這組包含洋基2023配股比率推導及畸零估值假設，尚未完整核實。')
    columns=st.columns(3)
    columns[0].metric('期末資產',f"{case['summary']['final_nav']:,.0f}元")
    columns[1].metric('年化報酬',f"{case['summary']['cagr']:.2%}")
    columns[2].metric('累計交易成本',f"{case['summary']['costs']['total_cost']:,.0f}元")
    chart=pd.DataFrame(account['daily']).set_index('date')[['nav']].rename(columns={'nav':LABELS[mode]})
    for name,path in [('原策略',ROOT/CACHE/'cases/control.json'),('0050',ROOT/'.cache/technical-research/cases/benchmark.json')]:
        if name != LABELS[mode]:
            chart[name]=pd.DataFrame(_read(path)['account']['daily']).set_index('date').nav
    st.line_chart(chart)
    annual=pd.DataFrame(case['summary']['annual'])
    annual=annual[[c for c in ('year','start_nav','end_nav','profit','total_return') if c in annual]].copy()
    annual['total_return']=annual.total_return.map(lambda value:f'{value:.2%}')
    st.dataframe(annual.rename(columns={'year':'年度','start_nav':'期初資產','end_nav':'期末資產',
        'profit':'損益','total_return':'年度淨報酬'}),hide_index=True,use_container_width=True)
    trades=pd.DataFrame(account['trades'])
    with st.expander('查看逐筆買賣與原因'):
        selected=st.text_input('輸入股票代碼或名稱',key='chip_trade_stock')
        shown=trades
        if selected:
            shown=trades[trades.stock_id.str.contains(selected,regex=False)|trades.name.str.contains(selected,regex=False)]
        shown=shown[[c for c in ('date','stock_id','name','side','qty','reference_price','total_cost','cash_after','reason') if c in shown]].copy()
        reasons=REASONS|{'chip_sell_full':'法人轉賣，全部出場','chip_sell_half':'法人轉賣，減碼一半',
            'leader_entry':'選股訊號進場','fund_stock':'賣0050準備買股','initial_allocation':'初始買入0050','idle_cash':'閒置資金買0050'}
        shown['side']=shown.side.map({'buy':'買進','sell':'賣出'})
        shown['reason']=shown.reason.map(lambda value:reasons.get(value,value))
        st.dataframe(shown.rename(columns={'date':'日期','stock_id':'代碼','name':'名稱','side':'買賣',
            'qty':'股數','reference_price':'參考成交價','total_cost':'成本','cash_after':'成交後現金','reason':'原因'}),
            hide_index=True,use_container_width=True)
    st.download_button('下載全部買賣與費用CSV',trades.to_csv(index=False).encode('utf-8-sig'),
        file_name='chip_'+mode+'_trades.csv',mime='text/csv',key='chip_trades')
    st.download_button('下載每日資產CSV',pd.DataFrame(account['daily']).to_csv(index=False).encode('utf-8-sig'),
        file_name='chip_'+mode+'_daily.csv',mime='text/csv',key='chip_daily')
    with st.expander('交叉驗證與資料限制'):
        st.write('逐筆訊號診斷有434筆完整63日資料；不等於可成交帳戶。95%區間未修正多重試驗。')
        pairs=[]
        for name,result in summary['paired_priority'].items():
            quantiles=result['nav_difference_quantiles']
            pairs.append({'方法':LABELS[name],'完成配對':result['complete_pairs'],
                '贏過同排序原策略次數':result['wins'],
                '期末資產中位差（元）':None if quantiles is None else round(quantiles[1])})
        st.dataframe(pd.DataFrame(pairs),hide_index=True,use_container_width=True)
        st.caption('固定20種同日候選排序；同種子配對比較。這是資金路徑壓力測試，不是20個獨立市場。其他方法尚未做這項測試。')
        diagnostic=summary['event_study']['comparisons'].get(mode)
        if diagnostic:
            coverage=summary['event_study']['coverage'][mode]
            st.write(f"458筆候選中，{coverage['known']}筆資料可判讀，{coverage['passed']}筆通過條件。")
            difference=diagnostic['pass_minus_reject']
            interval=diagnostic['ci95']
            if difference is not None and interval is not None:
                st.write(f'通過相對未通過的平均超額差：{difference*100:+.2f}個百分點；95%區間：{interval[0]*100:+.2f}至{interval[1]*100:+.2f}個百分點。')
                st.caption('區間跨過零，表示目前不能確認條件具有正向優勢。' if interval[0] <= 0 <= interval[1] else '此區間仍未處理多重試驗與反覆研究同一段歷史的偏誤。')
        st.write('大戶延遲7／14日是假設，缺歷史首次發布版本。洋基2023配股率由發行與參與股數推導，精確公告小數及畸零淨額仍未核實。分點只測單日，不能辨認主力身分；零股日量不是當下可成交深度。')
