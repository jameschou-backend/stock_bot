"""Present completed accounts separately from missing historical execution evidence."""
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import re

from app.backtest_tool_ui import verified_bytes
from app import workbench_jobs as jobs
from skills.backtest_data_evidence import verify_report as verify_data_report

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'artifacts/backtest_completion_20260925.json'


def validate_sector_report(report, root=ROOT):
    """Validate a complete twelve-case report and the exact downloadable bytes."""
    if (not isinstance(report, dict) or report.get('format') != 'sector_account_v1'
            or report.get('live_qualified') is not False or report.get('unseen_validation') is not False
            or report.get('membership_point_in_time') is not False
            or type(report.get('preflight_only')) is not bool or type(report.get('strict_pit')) is not bool
            or report.get('status') not in ('exploratory', 'blocked')
            or report.get('start') != '2022-01-03' or report.get('end') != '2026-09-09'
            or report.get('initial_cash') != 1_000_000):
        raise ValueError('族群報告格式、期間或研究資格不正確')
    metrics = report.get('metrics', {})
    if not isinstance(metrics, dict) or any(type(metrics.get(k)) is not int or metrics[k] != 0
           for k in ('network_calls', 'finmind_requests', 'database_writes')):
        raise ValueError('族群背景工作必須全程離線')
    rows = report.get('case_rows')
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError('族群報告必須保留固定十二案，包含缺件案例')
    names = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('label'), str):
            raise ValueError('族群案例格式不正確')
        config = row.get('config', {})
        if not isinstance(config, dict):
            raise ValueError('族群案例設定不正確')
        arm, stress, board = config.get('arm'), config.get('stress'), config.get('board_only')
        if (arm not in ('benchmark', 'relative_strength', 'strength_with_turnover')
                or stress not in ('control', 'combined') or type(board) is not bool
                or config.get('benchmark') is not (arm == 'benchmark')
                or config.get('position_count') != (0 if arm == 'benchmark' else 5)):
            raise ValueError('族群案例設定不正確')
        name = f'{arm}_{stress}_{"board_only" if board else "mixed"}'
        if row.get('name') != name or name in names or row.get('status') not in ('blocked', 'completed_daily'):
            raise ValueError('族群案例名稱、狀態或數目不正確')
        names.add(name)
        result = json.loads(verified_bytes(dict(path=row.get('result_path'), sha256=row.get('result_sha256')), root, '.json'))
        complete = row['status'] == 'completed_daily'
        if (not isinstance(result, dict) or result.get('completed') is not complete or result.get('config') != config
                or result.get('live_qualified') is not False or result.get('unseen_validation') is not False):
            raise ValueError('族群案例附檔與完成狀態不一致')
        artifacts = row.get('artifact_paths')
        if complete:
            if report['preflight_only'] or report['strict_pit']:
                raise ValueError('預檢或嚴格 PIT 檢查不得顯示收益')
            summary = result.get('summary', {})
            if not isinstance(summary, dict):
                raise ValueError('族群帳戶摘要不正確')
            for key in ('total_return', 'max_drawdown'):
                value = row.get(key)
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or summary.get(key) != value):
                    raise ValueError('族群顯示收益與帳戶不一致')
            if not isinstance(artifacts, dict) or set(artifacts) != {'daily', 'trades'}:
                raise ValueError('族群完成案例缺少每日資產或交易明細')
            for descriptor in artifacts.values():
                verified_bytes(descriptor, root)
        elif (not row.get('reason') or result.get('summary')
              or row.get('total_return') is not None or row.get('max_drawdown') is not None or artifacts != {}):
            raise ValueError('未完成族群案例不得帶中途收益或明細')
    complete = all(row['status'] == 'completed_daily' for row in rows)
    if report.get('completed') is not complete or report['status'] != ('exploratory' if complete else 'blocked'):
        raise ValueError('族群整體完成狀態與各案例不一致')
    return report


def load_sector_job(path, root=ROOT, expected_sha256=None, request=None):
    root, path = Path(root).resolve(), Path(path)
    if not path.is_absolute():
        path = root / path
    if (path.parent != root / '.cache/workbench/jobs'
            or not re.fullmatch(r'[0-9a-f]{32}\.result\.json', path.name)
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError('族群背景工作結果路徑不正確')
    raw = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('族群背景工作結果指紋不一致')
    value = json.loads(raw)
    job_id = path.name.split('.')[0]
    if (value.get('format') != 'sector_account_job_v1' or value.get('job_id') != job_id
            or value.get('live_qualified') is not False or value.get('unseen_validation') is not False
            or type(value.get('network_calls')) is not int or value['network_calls'] != 0):
        raise ValueError('族群背景工作格式或研究資格不正確')
    run_dir = root / '.cache/sector-account-jobs' / job_id
    if value.get('report', {}).get('path') != str((run_dir / 'report.json').relative_to(root)):
        raise ValueError('族群背景工作未綁定自己的報告')
    if value.get('requirements', {}).get('path') != str((run_dir / 'requirements.json').relative_to(root)):
        raise ValueError('族群背景工作未綁定自己的預檢缺件')
    json.loads(verified_bytes(value['requirements'], root, '.json'))
    report = validate_sector_report(json.loads(verified_bytes(value['report'], root, '.json')), root)
    mode = dict(preflight_only=report['preflight_only'], strict_pit=report['strict_pit'])
    expected_status = ('blocked' if mode['strict_pit'] else 'preflight_complete' if mode['preflight_only']
                       else report['status'])
    if value.get('request') != mode or value.get('status') != expected_status:
        raise ValueError('族群背景工作模式與報告不一致')
    if request is not None and (request.get('kind') != 'sector_backtest'
            or request.get('sector_preflight', True) != mode['preflight_only']
            or request.get('sector_strict_pit', False) != mode['strict_pit']):
        raise ValueError('族群工作設定與已提交請求不同')
    for row in report['case_rows']:
        descriptors = [dict(path=row['result_path'], sha256=row['result_sha256']), *row['artifact_paths'].values()]
        if any(not (root / item['path']).resolve().is_relative_to(run_dir) for item in descriptors):
            raise ValueError('族群案例或明細不屬於本次工作')
    return value, report


def latest_sector_job(recent=None, root=ROOT):
    recent = jobs.recent_jobs(50) if recent is None else recent
    job = next((j for j in recent if j.get('request', {}).get('kind') == 'sector_backtest'), None)
    if job is None:
        return dict(available=False, has_job=False, note='可先預檢新族群資料，再重跑固定帳戶。')
    if job.get('status') != 'completed':
        return dict(available=False, has_job=True, job=job, note=job.get('message', '族群工作尚未完成'))
    try:
        if not re.fullmatch(r'[0-9a-f]{32}', str(job.get('job_id', ''))):
            raise ValueError('族群工作識別碼不正確')
        expected = Path(root) / '.cache/workbench/jobs' / (job['job_id'] + '.result.json')
        if (Path(job.get('result_path', '')) != expected
                or not re.fullmatch(r'[0-9a-f]{64}', str(job.get('result_sha256', '')))):
            raise ValueError('族群工作缺少結果路徑或指紋')
        value, report = load_sector_job(expected, root, job['result_sha256'], job['request'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(available=False, has_job=True, job=job, note=f'最新族群工作無法核對：{exc}')
    return dict(available=True, has_job=True, job=job, value=value, report=report)


def corporate_label(config):
    return ' · '.join(('0050' if config['benchmark'] else '原策略',
        '一般成本' if config['stress'] == 'control' else '加嚴成交',
        '只整張' if config['board_only'] else '整張＋零股'))


def validate_data_scope(corporate, sector, evidence):
    """Coverage must describe these precise accounts, not another valid audit."""
    expected = [(row['path'], row['sha256']) for row in corporate['cases'].values()]
    expected.extend((row['result_path'], row['result_sha256']) for row in sector['case_rows'])
    declared = evidence.get('case_sources', {})
    actual = [(row['path'], row['sha256']) for row in declared.values()]
    if (not expected or len(set(expected)) != len(expected)
            or len(actual) != len(expected) or set(actual) != set(expected)
            or set(evidence.get('cases', {})) != set(declared)
            or evidence.get('coverage_totals', {}).get('case_count') != len(expected)):
        raise ValueError('成交資料覆蓋報告與顯示帳戶的範圍不一致')


def load(path=REPORT, root=ROOT):
    root, path = Path(root).resolve(), Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('補齊報告與已發布指紋不同')
    index = json.loads(raw)
    if index.get('format') != 'backtest_completion_v1' or index.get('live_qualified') is not False:
        raise ValueError('補齊報告格式或資格錯誤')
    reports = {key: json.loads(verified_bytes(value, root, '.json'))
               for key, value in index['reports'].items()}
    if set(reports) != {'corporate', 'sector', 'data'}:
        raise ValueError('缺少公司行動、族群帳戶或資料證據報告')
    if verify_data_report(root / index['reports']['data']['path'], root) != reports['data']:
        raise ValueError('資料證據在來源核對期間已變動')
    validate_data_scope(reports['corporate'], reports['sector'], reports['data'])
    for report in reports.values():
        if report.get('live_qualified') is not False:
            raise ValueError('研究報告不能提升實盤資格')
    rows = []
    for name, row in reports['corporate']['cases'].items():
        rows.append(dict(name=name, label=name, status='completed_daily' if row['completed'] else 'blocked',
            result_path=row['path'], result_sha256=row['sha256'], config=row['config'],
            total_return=(row.get('summary') or {}).get('total_return'),
            max_drawdown=(row.get('summary') or {}).get('max_drawdown'), artifact_paths={}))
    rows.extend(reports['sector']['case_rows'])
    for row in rows:
        result = json.loads(verified_bytes(dict(path=row['result_path'], sha256=row['result_sha256']), root, '.json'))
        complete = row['status'] == 'completed_daily'
        if result.get('completed') is not complete or result.get('config') != row['config']:
            raise ValueError('帳戶完成狀態或設定不一致')
        if result.get('live_qualified') is not False or result.get('unseen_validation') is not False:
            raise ValueError('帳戶研究資格不正確')
        if complete:
            if any(result['summary'][key] != row[key] for key in ('total_return', 'max_drawdown')):
                raise ValueError('顯示收益與帳戶不一致')
            for descriptor in row['artifact_paths'].values():
                verified_bytes(descriptor, root)
        elif row.get('total_return') is not None or row.get('max_drawdown') is not None or row['artifact_paths']:
            raise ValueError('未完成帳戶不得顯示中途收益')
    return index, reports


def render():
    import pandas as pd
    import streamlit as st
    with st.container():
        st.subheader('回測補齊進度：公司行動、族群策略與歷史成交')
        st.caption('2022/1/3～2026/9/9，期初100萬元、獲利續投、閒錢保留現金。'
                   '0050只作獨立比較。以下均為已見歷史研究，尚未取得實盤資格。')
        with st.form('sector_account_controls'):
            strict = st.checkbox('要求歷史當時的產業名冊證據（嚴格 PIT）', value=False,
                help='現有名冊是目前分類回推歷史；勾選後會阻擋帳戶計算，直到歷史名冊證據補齊。')
            columns = st.columns(2)
            preflight = columns[0].form_submit_button('新族群資料預檢')
            rerun = columns[1].form_submit_button('重跑新族群帳戶', type='primary')
            st.caption('固定十二案、全程使用本機資料；不抓行情、不下單、不改排程。背景工作上限30分鐘。')
        if preflight or rerun:
            try:
                job = jobs.submit(jobs.WorkRequest(kind='sector_backtest',
                    sector_preflight=preflight, sector_strict_pit=strict))
                st.success(f"族群工作 {job['job_id'][:8]} 已啟動，完成後按頁首重新整理。")
            except (ValueError, TimeoutError) as exc:
                st.warning(str(exc))
        snapshot = latest_sector_job()
        if snapshot['has_job']:
            st.caption('最近族群工作：' + snapshot['job']['job_id'][:8])
        if not snapshot['available']:
            st.info(snapshot['note'])
        index, reports = None, {}
        try:
            index, reports = load()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.info(f'原封存補齊報告尚未發布或無法核對：{exc}')
        corporate = reports.get('corporate', {'cases': {}})
        # A new failed/running/invalid job never silently falls back to old returns.
        sector = (snapshot['report'] if snapshot['available'] else
                  reports.get('sector') if not snapshot['has_job'] else None)
        if snapshot['available']:
            value = snapshot['value']
            st.info({'preflight_complete':'資料預檢完成，尚未計算報酬。',
                     'blocked':'檢查完成；缺件或嚴格PIT條件未通過的案例不顯示收益。',
                     'exploratory':'最新十二案日資料帳戶完成；屬歷史估算。'}[value['status']])
            with st.expander('最新族群預檢與來源缺件', expanded=value['status'] != 'exploratory'):
                st.json(json.loads(verified_bytes(value['requirements'], suffix='.json')))
            st.download_button('下載最新族群工作報告', verified_bytes(value['report'], suffix='.json'),
                snapshot['job']['job_id']+'-sector-report.json', 'application/json', key='latest_sector_report')
        elif sector is not None:
            st.caption('新族群區目前顯示原封存結果；可用上方按鈕重新執行。')
        rows = []
        for name, row in corporate['cases'].items():
            summary = row.get('summary') or {}
            rows.append({'研究': '原策略補公司行動', '案例': corporate_label(row['config']),
                '狀態': '日資料完成' if row['completed'] else '缺資料',
                '累積淨報酬': f"{summary['total_return']:.2%}" if row['completed'] else '未計算',
                '最大回撤': f"{summary['max_drawdown']:.2%}" if row['completed'] else '未計算',
                '原因': row.get('reason') or ''})
        for row in (sector or {}).get('case_rows', []):
            complete = row['status'] == 'completed_daily'
            preflight = sector.get('preflight_only') and not sector.get('strict_pit')
            rows.append({'研究': '新族群帳戶', '案例': row['label'],
                '狀態': '日資料完成' if complete else '尚未執行' if preflight else '缺資料',
                '累積淨報酬': f"{row['total_return']:.2%}" if complete else '未計算',
                '最大回撤': f"{row['max_drawdown']:.2%}" if complete else '未計算',
                '原因': row.get('reason') or ''})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        completed = [r for r in (sector or {}).get('case_rows', []) if r['status'] == 'completed_daily']
        if completed:
            choice = st.selectbox('新族群帳戶明細', range(len(completed)),
                format_func=lambda i: completed[i]['label'], key='completion_case')
            selected = completed[choice]
            daily = pd.read_csv(BytesIO(verified_bytes(selected['artifact_paths']['daily'])))
            st.line_chart(daily.set_index('date')[['nav']].rename(columns={'nav': '每日資產'}))
            for name, label in (('trades', '買賣'), ('daily', '每日資產')):
                st.download_button(f'下載新族群{label}明細', verified_bytes(selected['artifact_paths'][name]),
                    f"{selected['name']}-{name}.csv", 'text/csv', key='completion_'+name)
        with st.expander('完整缺件與研究限制'):
            if reports:
                st.caption('以下資料證據來自原封存補齊報告。')
                st.json(reports['data'])
            for limitation in (sector or {}).get('limitations', []):
                st.write(limitation)
        if index is not None:
            with st.expander('原封存補齊摘要（不含最新背景工作）'):
                for item in index['disposition']:
                    st.write(f"**{item['title']}：{item['status']}** — {item['detail']}")
                st.download_button('下載原封存補齊摘要', json.dumps(index, ensure_ascii=False, indent=2),
                    'backtest-completion-sealed.json', 'application/json', key='completion_summary')
