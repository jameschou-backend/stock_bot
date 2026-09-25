"""Bounded replay controls and verified, read-only result presentation."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re

from app import workbench_jobs as jobs

ROOT = Path(__file__).resolve().parents[1]
REPORT_STATUSES = {'preflight_ready', 'blocked', 'exploratory'}
CASE_STATUSES = {'ready', 'blocked', 'completed_daily'}
STATUS_LABELS = {'preflight_ready': '預檢通過，尚未執行', 'blocked': '證據不足，已阻擋',
                 'exploratory': '日資料研究完成', 'ready': '預檢通過',
                 'completed_daily': '日資料估算完成'}


def _safe_path(name, root, suffix):
    if not isinstance(name, (str, Path)) or not str(name):
        raise ValueError('回測結果缺少檔案路徑。')
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path.suffix != suffix:
        raise ValueError('回測結果路徑超出允許範圍。')
    return path


def verified_bytes(descriptor, root=None, suffix='.csv'):
    """Verify the same bytes that the UI will offer for download."""
    root = Path(root or ROOT).resolve()
    if not isinstance(descriptor, dict) or not re.fullmatch(r'[0-9a-f]{64}', str(descriptor.get('sha256', ''))):
        raise ValueError('回測結果缺少有效的檔案指紋。')
    raw = _safe_path(descriptor.get('path'), root, suffix).read_bytes()
    if hashlib.sha256(raw).hexdigest() != descriptor['sha256']:
        raise ValueError('回測結果或明細已變動，請重新執行驗證。')
    return raw


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_report(report, root=None):
    """Reject incomplete evidence before any performance figure is displayed."""
    root = Path(root or ROOT).resolve()
    if (not isinstance(report, dict) or report.get('format') != 'backtest_tool_v1'
            or report.get('live_qualified') is not False or report.get('unseen_validation') is not False
            or report.get('status') not in REPORT_STATUSES):
        raise ValueError('回測報告格式或研究資格標示不正確，暫不顯示結果。')
    preflight, metrics, rows = report.get('preflight'), report.get('metrics'), report.get('case_rows')
    if not isinstance(preflight, dict) or not isinstance(metrics, dict) or not isinstance(rows, list):
        raise ValueError('回測報告缺少預檢、執行紀錄或案例。')
    if not isinstance(preflight.get('issues'), list):
        raise ValueError('回測預檢缺少問題清單。')
    if (any(type(preflight.get(key)) is not int or preflight[key] < 0
            for key in ('ready_cases', 'blocked_cases')) or 'scope' not in preflight):
        raise ValueError('回測預檢缺少案例數或範圍。')
    for issue in preflight['issues']:
        if (not isinstance(issue, dict) or not isinstance(issue.get('code'), str)
                or not isinstance(issue.get('message'), str)):
            raise ValueError('回測預檢問題格式不正確。')
    for name in ('elapsed_seconds', 'source_validation_seconds', 'executed_cases', 'reused_cases'):
        if not _number(metrics.get(name)) or metrics[name] < 0:
            raise ValueError('回測執行紀錄不完整。')
    if metrics.get('network_calls') != 0 or isinstance(metrics.get('network_calls'), bool):
        raise ValueError('回測報告未確認全程使用本機資料。')
    names = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get('name'), str) or not row['name']
                or row['name'] in names or not isinstance(row.get('label'), str)
                or row.get('status') not in CASE_STATUSES or not isinstance(row.get('cache_hit'), bool)
                or row.get('reason') is not None and not isinstance(row.get('reason'), str)):
            raise ValueError('回測案例格式不正確或重複。')
        names.add(row['name'])
        if row['status'] != 'completed_daily':
            if row['status'] == 'blocked' and not row.get('reason'):
                raise ValueError('被阻擋案例缺少原因。')
            if (row.get('total_return') is not None or row.get('max_drawdown') is not None
                    or row.get('result_path') or row.get('artifact_paths')):
                raise ValueError('未完成案例含有中途收益或明細，暫不顯示結果。')
            continue
        if (report['status'] == 'preflight_ready' or report.get('mode') == 'strict'
                or not all(_number(row.get(k)) for k in ('total_return', 'max_drawdown'))):
            raise ValueError('回測完成狀態與績效不一致。')
        result = json.loads(verified_bytes(
            {'path': row.get('result_path'), 'sha256': row.get('result_sha256')}, root, '.json'))
        if (not isinstance(result, dict) or result.get('completed') is not True
                or result.get('live_qualified') is not False or result.get('unseen_validation') is not False
                or not isinstance(row.get('config'), dict) or result.get('config') != row['config']):
            raise ValueError('案例附檔的完成狀態、設定或研究資格與報告不一致。')
        summary = result.get('summary')
        if (not isinstance(summary, dict) or any(not _number(summary.get(key)) or summary[key] != row[key]
                for key in ('total_return', 'max_drawdown'))):
            raise ValueError('報告顯示績效與案例附檔摘要不一致。')
        artifacts = row.get('artifact_paths')
        if not isinstance(artifacts, dict) or set(artifacts) != {'trades', 'daily'}:
            raise ValueError('回測案例缺少成交或每日資產明細。')
        for descriptor in artifacts.values():
            verified_bytes(descriptor, root)
    return report


def load_report(path, root=None, expected_sha256=None):
    root = Path(root or ROOT).resolve()
    path = _safe_path(path, root, '.json')
    raw = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('回測報告與背景工作紀錄不一致，請重新執行。')
    return validate_report(json.loads(raw), root)


def latest_report(recent=None, root=None):
    """Use the latest completed tool job; never substitute an older valid result."""
    root = Path(root or ROOT).resolve()
    recent = jobs.recent_jobs(50) if recent is None else recent
    job = next((j for j in recent if j.get('request', {}).get('kind') == 'verified_backtest'
                and j.get('status') == 'completed'), None)
    if job is None:
        return {'available': False, 'note': '先執行資料預檢，再依結果開始回測。'}
    try:
        expected = (jobs.JOBS_DIR / f"{job['job_id']}.result.json").resolve()
        if (not re.fullmatch(r'[0-9a-f]{32}', job['job_id'])
                or _safe_path(job.get('result_path'), root, '.json') != expected
                or not re.fullmatch(r'[0-9a-f]{64}', str(job.get('result_sha256', '')))):
            raise ValueError('背景工作缺少可驗證的結果路徑或指紋。')
        report = load_report(expected, root, job['result_sha256'])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {'available': False, 'note': f'最近一份回測結果無法核對：{exc}'}
    return {'available': True, 'job': job, 'report': report}


def comparison_rows(report):
    rows = []
    for case in report['case_rows']:
        complete = case['status'] == 'completed_daily'
        rows.append({'案例': case['label'], '結果': STATUS_LABELS[case['status']],
                     '累積淨報酬': f"{case['total_return']:.2%}" if complete else '未計算',
                     '最大回撤': f"{case['max_drawdown']:.2%}" if complete else '未計算',
                     '重用結果': '是' if case['cache_hit'] else '否', '原因': case.get('reason') or ''})
    return rows


def render():
    import pandas as pd
    import streamlit as st

    st.subheader('原封存條件回測工具')
    st.caption('期初 100 萬元 · 2022/1/3～2026/9/9 · 封存 458 筆候選訊號。'
               '閒置資金保留現金；0050 只作比較基準，不配置閒置資金。')
    st.caption('此區保留補件前的封存結果供重現；本輪公司行動與族群帳戶結果請看上方補齊進度。')
    mode_labels = {'daily': '日資料估算', 'strict': '逐筆成交證據檢查'}
    policy_labels = {'board_only': '整張', 'mixed': '整張加零股', 'all': '全部比較'}
    stress_labels = {'control': '一般成本', 'combined': '加嚴', 'all': '全部比較'}
    with st.form('verified_backtest_controls'):
        columns = st.columns(3)
        mode = columns[0].selectbox('回測方式', list(mode_labels), format_func=mode_labels.get)
        policy = columns[1].selectbox('成交單位', list(policy_labels), index=2, format_func=policy_labels.get)
        stress = columns[2].selectbox('成交壓力', list(stress_labels), index=2, format_func=stress_labels.get)
        fresh = st.checkbox('重新計算並核對既有結果', value=False,
                            help='相同輸入通常重用已驗證結果；勾選後重新計算並比較一致性。')
        st.caption('日資料估算仍依賴成交假設；逐筆成交證據檢查缺證據時會阻擋。'
                   '兩者都是歷史研究，尚未通過未見期間驗證或實盤資格。')
        columns = st.columns(2)
        preflight = columns[0].form_submit_button('背景資料預檢')
        run = columns[1].form_submit_button('開始背景驗證回測', type='primary')
    if preflight or run:
        try:
            job = jobs.submit(jobs.WorkRequest(kind='verified_backtest', replay_mode=mode,
                replay_policy=policy, replay_stress=stress, replay_preflight=preflight, replay_fresh=fresh))
            st.success(f"工作 {job['job_id'][:8]} 已啟動；可在頁尾查看進度，完成後按頁首「重新整理」。")
        except (ValueError, TimeoutError) as exc:
            st.warning(str(exc))
    st.caption('背景工作只讀取本機證據，最長 30 分鐘；不抓行情、不下單、不調整排程。')
    snapshot = latest_report()
    if not snapshot['available']:
        st.info(snapshot['note'])
        return
    report, job = snapshot['report'], snapshot['job']
    st.write(f"**最近結果：{STATUS_LABELS[report['status']]}** · 工作 {job['job_id'][:8]}")
    request = job.get('request', {})
    st.caption('顯示工作設定：' + ' · '.join([
        mode_labels.get(request.get('replay_mode'), '依報告設定'),
        policy_labels.get(request.get('replay_policy'), '依報告設定'),
        stress_labels.get(request.get('replay_stress'), '依報告設定')]))
    if report['status'] == 'blocked':
        st.warning('部分或全部案例缺少必要證據。被阻擋案例不提供中途收益；僅完成案例顯示全期日資料估算。')
    elif report['status'] == 'preflight_ready':
        st.info('預檢完成；尚未計算報酬。可使用相同條件開始背景驗證回測。')
    else:
        st.info('以下是歷史日資料估算，含成本且非年化；不代表逐筆成交已驗證或可投入實盤。')
    metrics = report['metrics']
    columns = st.columns(4)
    for column, label, value in zip(columns, ['總耗時', '來源核對', '本次計算案例', '重用案例'],
            [f"{metrics['elapsed_seconds']:.1f} 秒", f"{metrics['source_validation_seconds']:.1f} 秒",
             metrics['executed_cases'], metrics['reused_cases']]):
        column.metric(label, value)
    if report['preflight']['issues']:
        with st.expander('資料預檢與缺件', expanded=report['status'] == 'blocked'):
            for issue in report['preflight']['issues']:
                st.write(f"{issue.get('case_name') or '共用資料'}：{issue['message']}")
    frame = pd.DataFrame(comparison_rows(report))
    if not frame.empty:
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.download_button('下載案例比較 CSV', frame.to_csv(index=False).encode('utf-8-sig'),
                           'backtest-comparison.csv', 'text/csv', key='verified_comparison_download')
    st.download_button('下載這次回測報告', json.dumps(report, ensure_ascii=False, indent=2),
                       'verified-backtest.json', 'application/json', key='verified_report_download')
    completed = [row for row in report['case_rows'] if row['status'] == 'completed_daily']
    if completed:
        selected = st.selectbox('下載案例明細', range(len(completed)),
                                format_func=lambda index: completed[index]['label'], key='verified_case_download')
        case = completed[selected]
        try:
            daily_raw = verified_bytes(case['artifact_paths']['daily'])
            daily = pd.read_csv(BytesIO(daily_raw))
            if {'date', 'nav'}.issubset(daily.columns):
                st.caption('選定案例每日資產（元）')
                st.line_chart(daily.set_index('date')[['nav']].rename(columns={'nav': '每日資產'}))
            for key, title in [('trades', '成交'), ('daily', '每日資產')]:
                raw = daily_raw if key == 'daily' else verified_bytes(case['artifact_paths'][key])
                st.download_button(f'下載{title} CSV', raw, f"{Path(case['artifact_paths'][key]['path']).name}",
                                   'text/csv', key='verified_' + key + '_download')
        except (OSError, ValueError) as exc:
            st.warning(str(exc))
