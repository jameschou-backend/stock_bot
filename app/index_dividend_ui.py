"""Read the additive issuer reconciliation without changing sealed qualifications."""
import json
from pathlib import Path
from app.index_continuous_ui import ROOT
from app.backtest_tool_ui import verified_bytes
from skills.backtest_case_cache import file_identities
from skills.index_dividend_reconciliation import reconcile

PUBLICATION=Path('artifacts/forward_simulation/index_dividends_20260927.json')


def load(root=ROOT):
    root=Path(root);path=root/PUBLICATION
    value=json.loads(verified_bytes(dict(path=str(PUBLICATION),sha256=path.with_suffix('.sha256').read_text().strip()),root,'.json'))
    if (value.get('passed') is not True or any(value.get(k) is not False for k in ('strict_data_ready','live_qualified','unseen_validation'))
            or value.get('changed_accounts')!=0 or value.get('new_backtests')!=0):raise ValueError('配息核對狀態不符')
    if file_identities([root/p for p in value['audit_code']],root)!=value['audit_code']:raise ValueError('配息核對程式已變更')
    actual=reconcile(root)
    if value!={**actual,'audit_code':value['audit_code']}:raise ValueError('配息核對未能重現')
    return value


def overview(root=ROOT):
    try:
        v=load(root)
        return dict(available=True,passed=True,unique_period_dividends=v['unique_period_dividends'],
                    accounts_verified=len(v['account_checks']),changed_accounts=0,
                    benchmark_payment_dates_issuer_matched=True,live_qualified=False,
                    source_url=v['source_url'],observed_at_utc=v['observed_at_utc'])
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return dict(available=False,reason=str(exc),live_qualified=False)


def render():
    import streamlit as st
    if not (ROOT/PUBLICATION).exists():return
    try:
        v=load()
        st.success(f"新增官方配息核對：{v['unique_period_dividends']}筆除息、金額與發放日均一致，{len(v['account_checks'])}個0050基準帳戶的現金皆在發放日入帳。報酬未改變。")
        st.caption('補上支付日的獨立來源；原始公告時間、每日限價推算及日資料成交限制仍保留。舊封存報告不覆寫，不提升實戰資格。')
        st.markdown('[查看元大官方歷史配息]('+v['source_url']+')')
        st.download_button('下載官方配息與現金核對',json.dumps(v,ensure_ascii=False),file_name='0050-dividend-reconciliation.json',mime='application/json',key='index_dividend_download')
    except (OSError,ValueError,KeyError,TypeError) as exc:st.error('官方配息核對失敗：'+str(exc))
