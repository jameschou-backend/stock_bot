"""Human-readable front end for the separate prospective paper account."""
import json
from datetime import datetime
import pandas as pd
import streamlit as st
from app import forward_portfolio as p, forward_portfolio_service as service, forward_journal as j, forward_comparison as comparison


def render():
    st.subheader('前向驗證：100萬元紙上資產帳本')
    account=st.radio('查看與登錄哪份前向帳本', ['策略帳本','0050比較帳本'],horizontal=True,key='pf_account')
    benchmark=account=='0050比較帳本'
    path=comparison.BENCHMARK if benchmark else p.PATH
    if benchmark and not path.exists():
        st.info('0050比較帳本尚未建立。')
        return
    st.caption('0050買入持有、股息入帳後再投入｜獨立紙上成交紀錄' if benchmark else '策略按上一收盤總資產複利配置｜不送券商、不用報價假設成交')
    try: data=p.summary(path)
    except (ValueError,OSError) as exc:
        st.error(str(exc));return
    columns=st.columns(3)
    columns[0].metric('總資產',f"${float(data['nav']):,.0f}" if data['nav'] is not None else '未知，待完整估值')
    columns[1].metric('可用現金',f"${float(data['available_cash']):,.0f}")
    columns[2].metric('委託預留',f"${float(data['reserved_cash']):,.0f}")
    st.caption(f"估值日期：{data['price_date'] or '尚未封存收盤'}；已登錄紙上成交：{data['fill_count']} 筆。回報由使用者提供，尚未經券商認證。")
    st.info('先核對成交及公司行動 → 封存收盤資產 → 建立下一交易日計畫。現階段尚不能宣稱可實戰或已勝過0050。')
    with st.expander('今日結算與下一交易日計畫',expanded=True):
        reviewed=st.checkbox('已核對持股的除息、分割及股款交付；如有事件已先登錄',key='pf_actions_reviewed_'+account)
        left,right=st.columns(2)
        if left.button('封存今日收盤資產',key='pf_close_'+account,use_container_width=True):
            try:
                (comparison.capture_benchmark_close if benchmark else service.capture_close)(path,actions_reviewed=reviewed)
                st.success('已封存；再次操作保留第一次紀錄。');st.rerun()
            except (ValueError,KeyError) as exc: st.error(str(exc))
        if right.button('建立下一交易日紙上計畫',key='pf_plan_'+account,use_container_width=True):
            try:
                if benchmark:
                    plans=comparison.reinvest_dividends(path)
                    st.success(f'已保存 {len(plans)} 筆股息再投入計畫；未成交初始本金不自動重試。')
                else:
                    result=comparison.save_strategy_plans(path)
                    if result['entry_block']: st.warning('出場已獨立檢查；新買進暫停：'+result['entry_block'])
                    else: st.success('已保存出場及買入計畫。')
            except (ValueError,KeyError) as exc: st.error(str(exc))
        if benchmark: st.caption('0050買入持有；只將已交付股息再投入，所有成交、費稅需分別登錄。')
        else: st.caption('個股最多3檔，每檔上限為前日總資產的1/3；剩餘資金提出0050計畫。需要賣0050才能買個股時，先保留資金需求，等賣出成交才釋放買單。')
    if data['holdings']:
        st.dataframe(pd.DataFrame(data['holdings']).rename(columns={'stock_id':'代號','qty':'持股數','cost':'含費成本','average_cost':'平均成本'}),hide_index=True)
    if data['orders']:
        table=[{'委託編號':o['order_id'],'代號':o['stock_id'],'買賣':'買' if o['side']=='buy' else '賣',
                '日期':o['session'],'交易別':'零股' if o['channel']=='odd' else '整張','限價':o['limit_price'],
                '委託股數':o['qty'],'已成交':o['filled'],'剩餘':o['remaining'],'狀態':o['status']} for o in data['orders']]
        st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
        with st.expander('登錄部分／全部成交，或取消剩餘委託'):
            order=st.selectbox('紙上委託',data['orders'],format_func=lambda o:f"{o['order_id']}｜{o['status']}",key='pf_order_'+account)
            with st.form('pf_fill_form_'+account):
                report_id=st.text_input('成交回報唯一編號（重送使用同一編號）')
                n=st.number_input('本筆成交股數',min_value=1,value=max(1,order['remaining']),step=1)
                price=st.number_input('本筆成交價',min_value=.01,value=float(order['limit_price']),format='%.2f')
                fees=st.number_input('本筆手續費',min_value=0.,value=20.,format='%.2f')
                tax=st.number_input('本筆交易稅（買進為0）',min_value=0.,value=0.,format='%.2f')
                at=st.text_input('實際發生時間（含時區）',value=f"{order['session']}T10:00:00+08:00")
                saved=st.form_submit_button('儲存紙上成交回報')
            if saved:
                try:
                    c=dict(kind='fill',id=report_id,order_id=order['order_id'],qty=n,price=str(price),fee=str(fees),tax=str(tax),
                        executed_at=at,evidence=dict(source='paper_execution_report',report_id=report_id))
                    if benchmark: comparison.record_benchmark(c,path)
                    else:
                        with j.connection(path) as con: p.submit(con,c)
                    st.success('成交已入帳，持股與剩餘預留已更新。');st.rerun()
                except (ValueError,KeyError) as exc: st.error(str(exc))
            reason=st.text_input('取消剩餘原因',value='核對當日回報，確認剩餘未成交',key='pf_cancel_reason_'+account)
            if st.button('取消剩餘數量',key='pf_cancel_'+account):
                try:
                    command=dict(kind='cancel',id=order['order_id'],order_id=order['order_id'],reason=reason)
                    if benchmark: comparison.record_benchmark(command,path)
                    else:
                        with j.connection(path) as con: p.submit(con,command)
                    st.rerun()
                except (ValueError,KeyError) as exc: st.error(str(exc))
    intents=[r for r in data['rows'] if r['kind']=='funding_intent']
    if intents:
        st.warning('曾有個股因現金不足而保留盤前資金需求；不代表已下單或已成交。')
        if st.button('0050成交入帳後，釋放今日資金需求',key='pf_release_'+account):
            try:
                service.release_funded_intents(path);st.rerun()
            except (ValueError,KeyError) as exc: st.error(str(exc))
    with st.expander('規則、公司行動與完整帳本'):
        if benchmark: st.write('0050基準不採個股停損或換股；初始未成交餘款留現金，只將已交付股息提出再投入計畫。')
        else: st.write('出場：收盤跌至調整後進場價的88%，或持有63個交易日，下一交易日提出限價賣出；未成交剩餘持股保留出場決策。限價委託不保證成交。')
        st.write('股息先列應收，實際入帳後才可使用；分割股未交付前禁止交易及完整估值。減資、碎股等未支援事件須暫停並對帳。')
        st.write('零股即時深度尚未接入；0050比較帳本使用相同會計引擎，但兩邊需要各自成交與結算證據。公司行動自動核對尚未完成。')
        report=st.file_uploader('進階：已核對公司行動 JSON（entitlement／delivery）',type=['json'],key='pf_action_'+account)
        if st.button('登錄公司行動證據',disabled=report is None,key='pf_action_save_'+account):
            try:
                c=json.loads(report.getvalue())
                if c['kind'] not in ('entitlement','delivery'): raise ValueError('只接受權益及交付紀錄')
                if benchmark: comparison.record_benchmark(c,path)
                else:
                    with j.connection(path) as con: p.submit(con,c)
                st.rerun()
            except (ValueError,KeyError,TypeError) as exc: st.error(str(exc))
        st.download_button('下載新版完整帳本',json.dumps(data['rows'],ensure_ascii=False,indent=2),'benchmark-0050.json' if benchmark else 'portfolio-v2.json','application/json',key='pf_download_'+account)
