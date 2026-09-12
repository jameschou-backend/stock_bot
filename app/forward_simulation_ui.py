"""A clearly separate scorecard for automated counterfactual observations."""
import json
import pandas as pd
import streamlit as st
from app import forward_simulation as sim, forward_automation as auto, forward_portfolio as p, forward_halts as h


def render(root=None, cash_mode=False):
    root=sim.ROOT if root is None else root
    from app import forward_cash_policy as cash
    if cash_mode:
        st.info("目前使用：個股＋現金。閒置資金保留現金；0050僅作獨立比較基準。")
        from app.forward_readiness_ui import render as render_readiness
        render_readiness(root)
    st.subheader('自動前向驗證｜模擬帳本')
    st.caption('策略與0050各100萬元，使用相同撮合規則。以下成交為模型推定，與下方原始回報帳本分開；不是券商成交。')
    if not root.exists():
        st.info('尚未初始化；必須在初始計畫交易日開盤前建立。');return
    try:
        books={role:p.summary(root/(role+'.sqlite3')) for role in ('strategy','benchmark')}
        for role in books:
            if cash_mode: cash.verify(root/(role+'.sqlite3'),role)
            else:
                with sim.j.connection(root/(role+'.sqlite3')) as con:sim.verify(con)
    except (ValueError,OSError) as exc:st.error(str(exc));return
    cols=st.columns(2)
    for col,role,label in zip(cols,books,['策略模擬','0050模擬']):
        b=books[role];col.metric(label+'資產',f"${float(b['nav']):,.0f}" if b['nav'] is not None else '待結算')
        col.caption(f"結算日：{b['price_date']}；模擬成交 {b['fill_count']} 筆")
        if cash_mode and role=='strategy':
            col.caption(f"現金 {float(b['cash']):,.0f} 元；委託預留 {float(b['reserved_cash']):,.0f} 元；可用 {float(b['available_cash']):,.0f} 元")
    runs=auto._runs(root)[-10:]
    if runs:st.dataframe(pd.DataFrame([dict(時間=r['recorded_at'],步驟=r['body']['stage'],結果=r['body']['status']) for r in runs]),hide_index=True)
    else:st.info('尚無交易日執行紀錄。休市時不抓取行情、不補造過去成交。')
    st.caption('排程每15分鐘喚醒一次，在交易時段取得兩次相隔20秒的快照；未觀察區間不推定成交。收盤流程18點後執行。需保持本機與Codex開啟、MySQL及網路可用。')
    with st.expander('固定撮合規則與自動流程'):
        st.write('兩次新鮮報價，間隔15至90秒；使用兩次可見深度的10%與期間成交增量的10%兩者較小值。整張1000股、零股1股；加入0.1%不利滑價，超出限價不成交。')
        st.write('每筆模擬成交計最低20元手續費、牌告費率0.1425%，股票賣出稅0.3%、0050賣出稅0.1%。同一帳本的同一觀察容量不得重複使用。')
        st.write('收盤取消未成交餘量 → 增量更新資料與封存訊號 → 核對公司行動 → 保存資產 → 建立下一交易日計畫。公司行動待核對時會停在該步。')
        st.json(sim.RULES)
    report=root/'latest-report.json'
    if report.exists():
        result=json.loads(report.read_text());comp=result['comparison']
        if comp['ready']:
            st.write(f"模擬報酬：策略 {comp['strategy_return']:.2%}，0050 {comp['benchmark_return']:.2%}；差距 {comp['excess_percentage_points']:+.2f} 個百分點。")
        st.download_button('下載模擬成交、每日資產與執行紀錄',report.read_text(),'forward-simulation.json','application/json',key='sim_report')
    role=st.radio('核對哪份模擬帳本',['strategy','benchmark'],format_func=lambda x:'策略模擬' if x=='strategy' else '0050模擬',horizontal=True,key='sim_role')
    path=root/(role+'.sqlite3');account=('現金模擬_' if cash_mode else '模擬_')+role
    with st.expander('模擬持股與未完成計畫'):
        if books[role]['holdings']:st.dataframe(pd.DataFrame(books[role]['holdings']),hide_index=True)
        st.dataframe(pd.DataFrame([{k:o[k] for k in ('stock_id','side','channel','session','limit_price','qty','filled','remaining','status')} for o in books[role]['orders']]),hide_index=True)
    from app import forward_corporate_ui,forward_evidence_ui
    forward_corporate_ui.render(path,account)
    forward_evidence_ui.render_halts(path,account)
    with st.expander('完成今日公司行動核對／登錄權益'):
        st.caption('先在上方更新來源並核對公告。這個確認不會修改現金或股數；有權益須先登錄，來源改變後需重新確認。')
        reviewer=st.text_input('核對人',key='sim_reviewer_'+role)
        note=st.text_area('核對內容（至少10字）',key='sim_note_'+role)
        checked=st.checkbox('已核對今日公告、停復牌及持股權益',key='sim_checked_'+role)
        if st.button('保存今日核對',disabled=not checked,key='sim_approve_'+role):
            try:
                if cash_mode: cash.verify(path,role)
                auto.approve(path,reviewer,note);st.success('核對已保存，下一次排程會嘗試完成結算。')
            except (ValueError,KeyError) as exc:st.error(str(exc))
        uploaded=st.file_uploader('已核對的權益／交付 JSON',type=['json'],key='sim_action_'+role)
        if st.button('登錄模擬帳本權益',disabled=uploaded is None,key='sim_action_save_'+role):
            try:
                cmd=json.loads(uploaded.getvalue())
                if cmd['kind'] not in ('entitlement','delivery'):raise ValueError('只接受權益及交付；不接受手填模擬成交')
                if cash_mode:
                    cash.verify(path,role)
                    if role=='strategy' and cmd.get('stock_id')=='0050':raise ValueError('現金策略不持有0050，不能登錄其權益')
                h.record(path,cmd,benchmark=role=='benchmark');st.rerun()
            except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))
    st.divider()
