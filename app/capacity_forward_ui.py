"""Read-only by default. All actions target the new simulation books only."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from app import capacity_forward as policy, capacity_forward_runner as runner
from app import forward_automation as base
from app.file_lock import file_lock

LABELS={'strategy':'成交金額排序','control':'原排序現金對照','benchmark':'0050獨立基準'}


def render(root=policy.ROOT):
    root=Path(root)
    with st.expander('實戰前準備：成交金額排序的每日模擬',expanded=True):
        st.caption('與舊帳本分開；只觀察已取得報價，不把歷史回測報酬搬進新帳戶。')
        if not root.is_dir():
            st.info('獨立前向版本尚未初始化。');return
        if st.button('更新公司行動來源（合計最多12次）',key='capacity_refresh'):
            try:
                with file_lock(root/'.run.lock',timeout=0):r=runner.refresh(runner.books(root))
                st.success(f"查詢 {r['calls']} 次、重用 {r['reused']} 筆；不足的來源留待下一批。")
            except Exception as exc:st.error(type(exc).__name__+'：'+str(exc))
        try:report=runner.status(root)
        except (ValueError,OSError) as exc:st.error(str(exc));return
        st.warning('前向模擬，尚未取得實盤資格；完整歷史母體、還原價及券商成交仍待驗證。')
        st.write(f"完整交易日流程 {report['completed_days']} 天；三份帳本皆由100萬元起算。")
        for cell,(role,b) in zip(st.columns(3),report['books'].items()):
            with cell:
                st.metric(LABELS[role],f"{float(b['nav']):,.0f} 元" if b['nav'] is not None else '淨值待核對')
                st.caption(f"現金 {float(b['cash']):,.0f} 元｜模擬成交 {b['fill_count']} 筆｜估值日 {b['price_date']}")
        role=st.selectbox('查看帳本',['strategy','control','benchmark'],format_func=LABELS.get,key='capacity_role')
        book=report['books'][role]
        from app.capacity_source_review import review
        try:
            temporal=review(root)
            checks=temporal['books'][role]['fills']
            passed=sum(f['source_freshness_passed'] for f in checks)
            st.write(f"成交當時的來源期限：{passed}／{len(checks)} 筆通過時間重建核對。")
            st.caption('目前來源過期不會自動推翻較早成交；這項核對只檢查當時資料是否已取得且未過期，不能代替完整公司行動或券商對帳。')
            if temporal['guard']['active']:
                st.caption('逐筆來源檢查啟用時間：'+pd.Timestamp(temporal['guard']['effective_at']).tz_convert('Asia/Taipei').strftime('%Y/%m/%d %H:%M:%S')+'（台北）；啟用前後分段保留。')
                if temporal['guard']['pending_checks']:
                    st.warning('有來源檢查尚未記錄完成結果，需核對中斷時的帳本；不可直接重推成交。')
            else:
                st.caption('逐筆來源檢查尚未啟用。')
            if checks:
                table=[dict(成交時間=pd.Timestamp(f['executed_at']).tz_convert('Asia/Taipei').strftime('%m/%d %H:%M:%S'),股數=f['qty'],
                            當時來源期限='通過' if f['source_freshness_passed'] else '缺件／過期',
                            最大來源年齡秒=round(max(s['age_seconds'] for s in f['sources'] if s['age_seconds'] is not None),2)
                            if any(s['age_seconds'] is not None for s in f['sources']) else None) for f in checks]
                st.dataframe(pd.DataFrame(table),hide_index=True)
            st.download_button('下載成交當時與目前來源核對',json.dumps(temporal,ensure_ascii=False,indent=2),
                               'capacity-source-review.json','application/json',key='capacity_source_review_download')
        except (ValueError,OSError) as exc:
            st.error('來源時間核對未完成：'+str(exc))
        st.write('資料狀態：'+('來源缺漏或有待核對事件' if book['source_blocked'] else '未發現已知來源衝突；仍須核對官方公告'))
        if book['issues']:st.dataframe(pd.DataFrame(book['issues']),hide_index=True)
        if book['plan']:
            plan=book['plan']
            st.write(f"訊號 {plan['signal_date']} → 委託交易日 {plan['entry_session']}")
            st.dataframe(pd.DataFrame(plan['decisions']).rename(columns={'stock_id':'股票','rank':'順序',
                'mean_turnover20_twd':'前20日成交金額估計','selected':'已建立計畫','reason':'選取或略過原因'}),hide_index=True)
        st.caption('隔日以訊號日收盤價掛限價；未成交不追價，收盤取消剩餘。計畫不是成交。')
        if book['orders']:
            frame=pd.DataFrame(book['orders'])
            keys=[k for k in ('stock_id','side','channel','session','qty','filled','limit_price','closed') if k in frame]
            st.dataframe(frame[keys].rename(columns={'stock_id':'股票','side':'買賣','channel':'交易別','session':'交易日',
                'qty':'委託股數','filled':'模擬成交股數','limit_price':'限價','closed':'已取消剩餘'}),hide_index=True)
        if role!='benchmark':
            guard=book['risk']
            st.write('新增買進：'+('暫停 — '+str(guard['reason']) if guard['blocked'] else '尚未觸及風險門檻'))
            from app.trading_risk_preference import load as load_risk_preference
            preference=load_risk_preference()
            st.caption(f"你的資金設定：本金 {float(preference['initial_capital_twd']):,.0f} 元、最大虧損 {float(preference['maximum_loss_fraction']):.0%}，設計採收盤資產高點回撤。現有封存模擬仍在20%暫停新增；50%設定尚未套用，原個股停損保留。")
            reason=st.text_input('暫停／恢復核對說明（至少10字）',key='capacity_pause_reason')
            left,right=st.columns(2)
            for cell,paused,label in [(left,True,'暫停新增並取消未成交買單'),(right,False,'核對後恢復後續計畫')]:
                if cell.button(label,key='capacity_pause_'+str(paused),disabled=len(reason.strip())<10):
                    try:
                        with file_lock(root/'.run.lock',timeout=0):policy.pause(root/(role+'.sqlite3'),paused,reason)
                        st.success('已保存；賣出委託保留。');st.rerun()
                    except (ValueError,OSError) as exc:st.error(str(exc))
        from app.capacity_close_preview_ui import render as render_close_preview
        render_close_preview(root,role)
        if st.checkbox('有持股後：顯示每日公司行動核對',key='capacity_show_review'):
            from app.official_review_packet_ui import render as render_official_packet
            render_official_packet(root,role)
            reviewer=st.text_input('核對人',key='capacity_reviewer')
            note=st.text_area('核對公告、停復牌及權益的結果',key='capacity_review_note')
            checked=st.checkbox('已完成所選帳本持股的今日公告核對',key='capacity_review_checked')
            if st.button('保存核對',key='capacity_approve',disabled=not checked):
                try:
                    with file_lock(root/'.run.lock',timeout=0):base.approve(root/(role+'.sqlite3'),reviewer,note)
                    st.success('核對已保存；仍有權益異常時不會強行結算。')
                except (ValueError,OSError) as exc:st.error(str(exc))
        if st.checkbox('比對券商成交回報（不改寫模擬帳本）',key='capacity_show_reconcile'):
            st.caption('先將券商檔案對應為欄位：execution_id, order_id, stock_id, side, channel, qty, price, fee, tax, executed_at；股數單位為股，時間須含時區。')
            upload=st.file_uploader('已對應的成交CSV',type=['csv'],key='capacity_execution_csv')
            if upload is not None:
                try:
                    import csv,io,hashlib
                    from app.capacity_execution_review import compare
                    raw=upload.getvalue()
                    result=compare(policy.verify(root/(role+'.sqlite3'),role),list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))))
                    result['source_csv_sha256']=hashlib.sha256(raw).hexdigest()
                    st.write(result['note']);st.dataframe(pd.DataFrame(result['rows']),hide_index=True)
                    st.download_button('下載成交差異',json.dumps(result,ensure_ascii=False,indent=2),
                        'execution-comparison.json','application/json',key='capacity_execution_download')
                except (ValueError,UnicodeError,OSError) as exc:st.error(str(exc))
        if report['days']:st.dataframe(pd.DataFrame(report['days'][-10:]),hide_index=True)
        for item in report['remaining']:
            if item=='20%整戶回撤暫定值僅供模擬，實盤水位須使用者選定':
                item='實盤資金偏好已記錄；現有20%模擬規則尚未切換至新風控版本。'
            st.write('• '+item)
        st.download_button('下載三帳本狀態與每日決策',json.dumps(report,ensure_ascii=False,indent=2),
            'capacity-forward-status.json','application/json',key='capacity_report_download')
