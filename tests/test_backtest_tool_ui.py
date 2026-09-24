import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from app import backtest_tool_ui as ui, workbench_jobs as jobs, workbench_worker as worker


def descriptor(path, root, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return {'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def report_at(root, status='blocked'):
    completed = status == 'exploratory'
    case = {'name': 'capacity_control_mixed', 'label': '五檔／整張加零股／一般成本',
            'status': 'completed_daily' if completed else 'ready' if status == 'preflight_ready' else 'blocked',
            'total_return': .125 if completed else None, 'max_drawdown': -.1 if completed else None,
            'config': {'benchmark': False, 'board_only': False, 'position_count': 5, 'stress': 'control'},
            'reason': '' if completed else '缺少逐筆來源', 'cache_hit': False,
            'result_path': None, 'result_sha256': None, 'artifact_paths': {}}
    if completed:
        account = descriptor(root / '.cache/replay/case.json', root, json.dumps({
            'completed': True, 'live_qualified': False, 'unseen_validation': False, 'config': case['config'],
            'summary': {'total_return': case['total_return'], 'max_drawdown': case['max_drawdown']}}))
        case.update(result_path=account['path'], result_sha256=account['sha256'], artifact_paths={
            'trades': descriptor(root / '.cache/replay/trades.csv', root, 'date,side\n2022-01-04,buy\n'),
            'daily': descriptor(root / '.cache/replay/daily.csv', root, 'date,nav\n2022-01-04,1000000\n'),
        })
    return {'format': 'backtest_tool_v1', 'status': status, 'live_qualified': False,
            'unseen_validation': False, 'preflight': {'issues': [], 'ready_cases': int(status != 'blocked'),
            'blocked_cases': int(status == 'blocked'), 'scope': 'fixed_local'}, 'case_rows': [case],
            'metrics': {'elapsed_seconds': 2.5, 'source_validation_seconds': .5, 'executed_cases': int(completed),
                        'reused_cases': 0, 'network_calls': 0}}


def completed_job(root, monkeypatch, report):
    folder = root / '.cache/workbench/jobs'
    monkeypatch.setattr(jobs, 'JOBS_DIR', folder)
    job_id = 'a' * 32
    result = descriptor(folder / f'{job_id}.result.json', root, json.dumps(report))
    return {'job_id': job_id, 'status': 'completed', 'request': {'kind': 'verified_backtest'},
            'result_path': str(root / result['path']), 'result_sha256': result['sha256']}


def test_validated_latest_result_and_csv_fingerprints(tmp_path, monkeypatch):
    report = report_at(tmp_path, 'exploratory')
    job = completed_job(tmp_path, monkeypatch, report)
    snapshot = ui.latest_report([job], root=tmp_path)
    assert snapshot['available']
    row = ui.comparison_rows(snapshot['report'])[0]
    assert row['累積淨報酬'] == '12.50%'
    assert row['最大回撤'] == '-10.00%'
    artifact = tmp_path / report['case_rows'][0]['artifact_paths']['daily']['path']
    artifact.write_text('changed')
    snapshot = ui.latest_report([job], root=tmp_path)
    assert not snapshot['available']
    assert '已變動' in snapshot['note']


@pytest.mark.parametrize('mutation', ['format', 'live_qualified', 'unseen_validation', 'nan', 'missing_metrics',
                                      'partial_returns', 'outside_path', 'broken_hash'])
def test_invalid_reports_withheld(tmp_path, monkeypatch, mutation):
    report = report_at(tmp_path, 'exploratory')
    if mutation == 'format': report['format'] = 'legacy'
    elif mutation in ('live_qualified', 'unseen_validation'): report[mutation] = True
    elif mutation == 'nan': report['case_rows'][0]['total_return'] = float('nan')
    elif mutation == 'missing_metrics': report.pop('metrics')
    elif mutation == 'partial_returns': report['case_rows'][0]['status'] = 'blocked'
    elif mutation == 'outside_path': report['case_rows'][0]['result_path'] = '../outside.json'
    elif mutation == 'broken_hash': report['case_rows'][0]['result_sha256'] = '0' * 64
    job = completed_job(tmp_path, monkeypatch, report)
    assert not ui.latest_report([job], root=tmp_path)['available']


def test_changed_report_and_symlink_escape_are_withheld(tmp_path, monkeypatch):
    report = report_at(tmp_path, 'exploratory')
    job = completed_job(tmp_path, monkeypatch, report)
    Path(job['result_path']).write_text(json.dumps({**report, 'status': 'blocked'}))
    assert not ui.latest_report([job], root=tmp_path)['available']
    outside = tmp_path.parent / f'{tmp_path.name}-outside.csv'
    outside.write_text('private')
    link = tmp_path / 'linked.csv'
    link.symlink_to(outside)
    with pytest.raises(ValueError, match='範圍'):
        ui.verified_bytes({'path': 'linked.csv', 'sha256': hashlib.sha256(b'private').hexdigest()}, tmp_path)


@pytest.mark.parametrize('mutation', ['total_return', 'max_drawdown', 'completed', 'live_qualified',
                                      'unseen_validation', 'config'])
def test_report_figures_and_qualification_must_match_verified_case(tmp_path, monkeypatch, mutation):
    report = report_at(tmp_path, 'exploratory')
    row = report['case_rows'][0]
    path = tmp_path / row['result_path']
    result = json.loads(path.read_text())
    if mutation in ('total_return', 'max_drawdown'):
        row[mutation] += .1
    elif mutation == 'config': result['config']['board_only'] = True
    elif mutation == 'completed': result[mutation] = False
    else: result[mutation] = True
    # Keep both file digests valid so rejection proves semantic validation.
    case = descriptor(path, tmp_path, json.dumps(result))
    row['result_sha256'] = case['sha256']
    job = completed_job(tmp_path, monkeypatch, report)
    snapshot = ui.latest_report([job], root=tmp_path)
    assert not snapshot['available']
    assert '不一致' in snapshot['note']


def test_blocked_case_never_formats_partial_return():
    row = {'label': '未完成', 'status': 'blocked', 'total_return': 999, 'max_drawdown': -.5,
           'cache_hit': False, 'reason': '缺少逐筆來源'}
    result = ui.comparison_rows({'case_rows': [row]})[0]
    assert result['累積淨報酬'] == result['最大回撤'] == '未計算'


def test_blocked_batch_keeps_completed_case_but_withholds_unfinished_case(tmp_path, monkeypatch):
    report = report_at(tmp_path, 'exploratory')
    report['status'] = 'blocked'
    blocked = report_at(tmp_path)['case_rows'][0]
    blocked['name'] = 'capacity_control_board_only'
    report['case_rows'].append(blocked)
    job = completed_job(tmp_path, monkeypatch, report)
    snapshot = ui.latest_report([job], root=tmp_path)
    assert snapshot['available']
    assert [row['累積淨報酬'] for row in ui.comparison_rows(snapshot['report'])] == ['12.50%', '未計算']


@pytest.mark.parametrize('report_status', ['blocked', 'preflight_ready', 'exploratory'])
def test_worker_completed_means_report_generated_not_live_qualified(tmp_path, monkeypatch, report_status):
    report = report_at(tmp_path, report_status)
    folder = tmp_path / '.cache/workbench/jobs'
    folder.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, 'JOBS_DIR', folder)
    request = jobs.WorkRequest(kind='verified_backtest')
    job = {'job_id': 'b' * 32, 'request': request.model_dump(mode='json'), 'created_at': 1}
    writes, calls = [], []
    monkeypatch.setattr(worker, 'read_job', lambda job_id: job.copy())
    monkeypatch.setattr(worker, 'write_job', lambda value: writes.append(value.copy()))
    def run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[command.index('--output') + 1]).write_text(json.dumps(report))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(worker.subprocess, 'run', run)
    worker.main(job['job_id'])
    result = writes[-1]
    assert result['status'] == 'completed'
    assert result['report_status'] == report_status
    assert result['live_qualified'] is False and result['unseen_validation'] is False
    assert result['result_sha256'] == hashlib.sha256(Path(result['result_path']).read_bytes()).hexdigest()
    assert calls[0][1]['timeout'] == 1800
    assert '--fetch' not in calls[0][0]


def test_worker_rejects_success_exit_with_missing_report(tmp_path, monkeypatch):
    folder = tmp_path / '.cache/workbench/jobs'
    folder.mkdir(parents=True)
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, 'JOBS_DIR', folder)
    job = {'job_id': 'b' * 32, 'request': jobs.WorkRequest(kind='verified_backtest').model_dump(mode='json')}
    writes = []
    monkeypatch.setattr(worker, 'read_job', lambda job_id: job.copy())
    monkeypatch.setattr(worker, 'write_job', lambda value: writes.append(value.copy()))
    monkeypatch.setattr(worker.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    worker.main(job['job_id'])
    assert writes[-1]['status'] == 'failed'
    assert 'result_path' not in writes[-1]


def test_ui_only_submits_bounded_background_work(monkeypatch):
    requests = []
    monkeypatch.setattr(ui, 'latest_report', lambda: {'available': False, 'note': '先預檢'})
    def submit(request):
        requests.append(request)
        return {'job_id': 'c' * 32}
    monkeypatch.setattr(jobs, 'submit', submit)
    app = AppTest.from_string('from app.backtest_tool_ui import render\nrender()').run()
    assert not app.exception
    assert '100 萬元' in app.caption[0].value and '458 筆候選訊號' in app.caption[0].value
    next(button for button in app.button if button.label == '背景資料預檢').click().run()
    assert requests[-1].replay_preflight is True
    assert requests[-1].kind == 'verified_backtest'
    next(select for select in app.selectbox if select.label == '回測方式').select('strict')
    next(select for select in app.selectbox if select.label == '成交單位').select('board_only')
    next(button for button in app.button if button.label == '開始背景驗證回測').click().run()
    assert not app.exception
    assert requests[-1].replay_preflight is False
    assert requests[-1].replay_mode == 'strict' and requests[-1].replay_policy == 'board_only'
    assert requests[-1].fetch_news is False and requests[-1].fetch_flow is False


def test_ui_blocked_report_has_no_performance_metric(tmp_path, monkeypatch):
    report = report_at(tmp_path)
    monkeypatch.setattr(ui, 'latest_report', lambda: {'available': True,
        'job': {'job_id': 'd' * 32}, 'report': report})
    app = AppTest.from_string('from app.backtest_tool_ui import render\nrender()').run()
    assert not app.exception
    assert all('報酬' not in metric.label and '回撤' not in metric.label for metric in app.metric)
    assert app.dataframe[0].value['累積淨報酬'].tolist() == ['未計算']
