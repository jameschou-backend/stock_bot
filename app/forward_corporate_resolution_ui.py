"""Preview first; saving terms does not post money or shares."""
from datetime import date
import json
import streamlit as st
from app import forward_corporate_resolution as r, forward_journal as j


def render(path, account, report):
    with st.expander('依公告核對條款／處理資料修訂'):
        st.caption('保存公告連結、相關文字與核對人。系統只檢查格式及帳本一致性，不會認證公告真偽，也不會因此增加現金或股數。')
        ids = report['scope']['stock_ids']
        if not ids:
            st.info('目前沒有需要核對的股票。')
            return
        prefix = 'corp_terms_' + account
        sid = st.selectbox('核對股票', ids, key=prefix+'_sid')
        kind = st.selectbox('權益種類', ['現金股息', '分割／反分割（無碎股）'], key=prefix+'_type')
        with st.form(prefix+'_form'):
            ex = st.date_input('除息日／分割恢復交易日', value=date.today())
            delivery = st.date_input('公告交付日', value=date.today())
            if kind == '現金股息':
                value = st.text_input('每股現金股息（元）', value='')
            else:
                value = st.text_input('分割後股數 ÷ 分割前股數', value='')
                halt = st.date_input('分割停牌起日', value=date.today())
            title = st.text_input('公告標題')
            url = st.text_input('公告HTTPS連結')
            published = st.text_input('公告發布時間（含時區，例如2026-09-11T18:00:00+08:00）')
            text = st.text_area('貼上與本次股息／分割相關的公告條款')
            reviewer = st.text_input('核對人')
            clicked = st.form_submit_button('預覽核對結果')
        cache_key = prefix+'_preview_'+str(path)
        if clicked:
            st.session_state.pop(cache_key, None)
            try:
                terms=dict(stock_id=sid,ex_date=str(ex),delivery_date=str(delivery),action_type='cash' if kind=='現金股息' else 'split')
                if kind=='現金股息': terms['cash_per_share']=value
                else: terms.update(ratio=value,halt_start=str(halt))
                st.session_state[cache_key] = r.preview(path, terms, dict(url=url,title=title,
                    published_at=published,text=text,reviewer=reviewer))
            except (ValueError,TypeError,KeyError,ArithmeticError) as exc:
                st.error(str(exc))
        preview = st.session_state.get(cache_key)
        if preview:
            command = preview['command']; terms = command['terms']
            st.write(f"待保存預覽：{terms['stock_id']}｜{terms['ex_date']}｜交付日 {terms['delivery_date']}")
            st.write('每股股息：'+terms['cash_per_share']+' 元' if terms['action_type']=='cash' else '分割比率：'+terms['ratio'])
            st.caption('保存的是下列預覽版本；修改上方欄位後需再按預覽。權益及實際交付仍需另行登錄。')
            st.write('核對人：'+command['evidence']['reviewer']+'；公告：'+command['evidence']['url'])
            remaining = [i for i in preview['after']['issues'] if i['blocking']]
            st.write('保存後仍需處理：'+'；'.join(i['message'] for i in remaining) if remaining else '預覽未發現其他阻擋；仍需完整官方公告核對。')
            if st.button('保存這份核對紀錄（不入帳）',key=prefix+'_save'):
                try:
                    r.save(path,command)
                    st.session_state.pop(cache_key,None)
                    st.success('已保存核對紀錄，現金和股數未變。');st.rerun()
                except (ValueError,TypeError,KeyError,ArithmeticError) as exc: st.error(str(exc))
        for record in report.get('resolutions',[]):
            terms=record['terms']
            st.write(f"已保存：{terms['stock_id']} {terms['ex_date']}｜{record['status']}")
            try:
                draft=r.entitlement_draft(path,record['hash'])
            except (ValueError,KeyError,TypeError,ArithmeticError) as exc:
                st.caption('尚不能建立當日權益草稿：'+str(exc))
            else:
                st.download_button('下載當日權益草稿（核對後於下方公司行動區匯入）',
                    json.dumps(draft,ensure_ascii=False,indent=2),'entitlement-draft.json','application/json',key=prefix+record['hash'])
