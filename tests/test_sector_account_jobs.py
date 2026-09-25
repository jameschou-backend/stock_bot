import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import backtest_completion_ui as ui
from app import workbench_jobs as jobs
from scripts import run_sector_account_job as entry


def put(root, path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
    path.write_bytes(raw)
    return dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(raw).hexdigest())


def report_fixture(root, output, *, preflight_only=False, strict_pit=False):
    complete = not (preflight_only or strict_pit)
    rows = []
    for board in (False, True):
        for stress in ('control', 'combined'):
            for arm in ('benchmark', 'relative_strength', 'strength_with_turnover'):
                name = f'{arm}_{stress}_{"board_only" if board else "mixed"}'
                config = dict(arm=arm, stress=stress, benchmark=arm == 'benchmark', board_only=board,
                              position_count=0 if arm == 'benchmark' else 5)
                result = dict(completed=complete, config=config, live_qualified=False, unseen_validation=False)
                reason = '' if complete else 'Strict PIT unavailable' if strict_pit else 'Preflight only'
                if complete:
                    result['summary'] = dict(total_return=.1, max_drawdown=-.2)
                else:
                    result['reason'] = reason
                case = put(root, output / 'cases' / (name + '.json'), result)
                artifacts = {} if not complete else {kind: put(root, output / 'exports' / f'{name}-{kind}.csv',
                    b'date,nav\n2022-01-03,1000000\n') for kind in ('daily', 'trades')}
                rows.append(dict(name=name, label=name, config=config,
                    status='completed_daily' if complete else 'blocked', reason=reason,
                    result_path=case['path'], result_sha256=case['sha256'], artifact_paths=artifacts,
                    total_return=.1 if complete else None, max_drawdown=-.2 if complete else None))
    source = put(root, root / 'source.json', dict(source='fixed'))
    report = dict(format='sector_account_v1', status='exploratory' if complete else 'blocked',
        completed=complete, start='2022-01-03', end='2026-09-09', initial_cash=1_000_000,
        case_rows=rows, live_qualified=False, unseen_validation=False, membership_point_in_time=False,
        preflight_only=preflight_only, strict_pit=strict_pit, limitations=['Current membership'],
        metrics=dict(network_calls=0, finmind_requests=0, database_writes=0),
        source_sha256={source['path']: source['sha256']})
    put(root, output / 'report.json', report)
    put(root, output / 'requirements.json', dict(missing_limits_stocks=[]))
    return report


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from scripts import research_sector_accounts as driver
    monkeypatch.setattr(entry, 'ROOT', tmp_path)
    wrapper = tmp_path / 'wrapper.py'
    wrapper.write_text('fixed wrapper')
    monkeypatch.setattr(entry, '__file__', str(wrapper))
    calls = []
    def replay(output, cache, additions, **kwargs):
        calls.append((output, cache, additions, kwargs))
        return report_fixture(tmp_path, output, **kwargs)
    monkeypatch.setattr(driver, 'run', replay)
    return tmp_path, calls


def output_path(root, digit='a'):
    return root / '.cache/workbench/jobs' / (digit * 32 + '.result.json')


def completed_job(root, path, request):
    return dict(job_id=path.name.split('.')[0], request=request, status='completed',
        result_path=str(path), result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize('preflight,strict,status', [(False, False, 'exploratory'),
    (True, False, 'preflight_complete'), (False, True, 'blocked'), (True, True, 'blocked')])
def test_cli_modes_publish_verified_bounded_job(prepared, preflight, strict, status):
    root, calls = prepared
    path = output_path(root)
    value = entry.run(path, preflight_only=preflight, strict_pit=strict)
    assert value['status'] == status
    assert calls[0][1] == entry.CACHE
    assert calls[0][2] == entry.ADDITIONS and len(entry.ADDITIONS) == 2
    request = jobs.WorkRequest(kind='sector_backtest', sector_preflight=preflight,
                              sector_strict_pit=strict).model_dump(mode='json')
    job = completed_job(root, path, request)
    latest = ui.latest_sector_job([job], root)
    assert latest['available']
    if preflight or strict:
        assert all(row['total_return'] is None and not row['artifact_paths']
                   for row in latest['report']['case_rows'])
    with pytest.raises(ValueError, match='new workbench'):
        entry.run(path)


def test_cli_does_not_publish_if_source_changes(prepared, monkeypatch):
    root, _ = prepared
    original = entry.validate_sector_report
    def mutate(report, root):
        original(report, root)
        (root / 'source.json').write_text('changed')
    monkeypatch.setattr(entry, 'validate_sector_report', mutate)
    path = output_path(root)
    with pytest.raises(ValueError, match='sources changed'):
        entry.run(path)
    assert not path.exists()


@pytest.mark.parametrize('relative', ['sealed.json', '.cache/old-seal/report.json',
    '.cache/workbench/jobs/nope.result.json', '.cache/workbench/jobs/' + 'a'*32 + '.json'])
def test_output_boundary_rejects_non_job_artifacts(prepared, relative):
    root, calls = prepared
    with pytest.raises(ValueError, match='new workbench'):
        entry.run(root / relative)
    assert calls == []


@pytest.mark.parametrize('target', ['case', 'csv', 'report', 'requirements', 'job'])
def test_latest_job_rejects_tampered_downloads(prepared, target):
    root, _ = prepared
    path = output_path(root)
    value = entry.run(path)
    job = completed_job(root, path, jobs.WorkRequest(kind='sector_backtest', sector_preflight=False).model_dump())
    report = json.loads((root / value['report']['path']).read_text())
    paths = dict(job=path, report=root / value['report']['path'],
        requirements=root / value['requirements']['path'],
        case=root / report['case_rows'][0]['result_path'],
        csv=root / report['case_rows'][0]['artifact_paths']['daily']['path'])
    paths[target].write_bytes(b'{}')
    assert ui.latest_sector_job([job], root)['available'] is False


def test_latest_job_does_not_fall_back_to_older_complete_account(prepared):
    root, _ = prepared
    path = output_path(root)
    entry.run(path)
    request = jobs.WorkRequest(kind='sector_backtest', sector_preflight=False).model_dump()
    older = completed_job(root, path, request)
    for status in ('failed', 'running', 'queued'):
        newer = dict(job_id='b'*32, request=request, status=status, message='new result unavailable')
        latest = ui.latest_sector_job([newer, older], root)
        assert latest['has_job'] and not latest['available']
        assert latest['job'] == newer


def test_strict_or_preflight_cannot_attach_completed_returns(prepared):
    root, _ = prepared
    path = output_path(root)
    value = entry.run(path)
    report = json.loads((root / value['report']['path']).read_text())
    for key in ('strict_pit', 'preflight_only'):
        report[key] = True
        with pytest.raises(ValueError, match='不得顯示收益'):
            ui.validate_sector_report(report, root)
        report[key] = False
    report['case_rows'][0]['total_return'] = float('inf')
    with pytest.raises(ValueError, match='顯示收益'):
        ui.validate_sector_report(report, root)


def test_request_mode_must_match_result(prepared):
    root, _ = prepared
    path = output_path(root)
    entry.run(path)
    with pytest.raises(ValueError, match='提交請求'):
        ui.load_sector_job(path, root, request=jobs.WorkRequest(kind='sector_backtest').model_dump())


def test_sector_command_is_offline_and_keeps_original_tool(tmp_path):
    request = jobs.WorkRequest(kind='sector_backtest', sector_strict_pit=True)
    command = jobs.command_for(request, tmp_path / 'result.json')
    assert command[1:] == ['scripts/run_sector_account_job.py', '--output', str(tmp_path / 'result.json'),
                          '--preflight-only', '--strict-pit']
    assert '--preflight-only' not in jobs.command_for(jobs.WorkRequest(kind='sector_backtest',
        sector_preflight=False), tmp_path / 'result.json')
    for field in ('fetch_news', 'fetch_flow'):
        with pytest.raises(ValueError):
            jobs.WorkRequest(kind='sector_backtest', **{field: True})
    assert jobs.command_for(jobs.WorkRequest(kind='verified_backtest'), tmp_path / 'old.json')[1] == 'scripts/run_verified_backtest.py'


def test_worker_verifies_job_and_uses_existing_timeout(prepared, monkeypatch):
    from app import workbench_worker as worker
    root, _ = prepared
    path = output_path(root)
    entry.run(path)
    request = jobs.WorkRequest(kind='sector_backtest', sector_preflight=False).model_dump()
    job = dict(job_id='a'*32, request=request, status='queued')
    captured, transport = [], []
    monkeypatch.setattr(worker, 'ROOT', root)
    monkeypatch.setattr(worker, 'JOBS_DIR', path.parent)
    monkeypatch.setattr(worker, 'read_job', lambda _: dict(job))
    monkeypatch.setattr(worker, 'write_job', lambda row: captured.append(dict(row)))
    monkeypatch.setattr(worker.subprocess, 'run', lambda *args, **kwargs:
        transport.append((args, kwargs)) or SimpleNamespace(returncode=0))
    worker.main(job['job_id'])
    assert captured[-1]['status'] == 'completed'
    assert captured[-1]['result_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert transport[0][1]['timeout'] == 1800


def test_corporate_label_is_readable_chinese():
    assert ui.corporate_label(dict(benchmark=False, stress='control', board_only=True)) == '原策略 · 一般成本 · 只整張'
    assert ui.corporate_label(dict(benchmark=True, stress='combined', board_only=False)) == '0050 · 加嚴成交 · 整張＋零股'


def test_buttons_work_before_sealed_summary_is_available(monkeypatch):
    from streamlit.testing.v1 import AppTest
    submitted = []
    monkeypatch.setattr(ui, 'load', lambda: (_ for _ in ()).throw(FileNotFoundError('not published')))
    monkeypatch.setattr(ui, 'latest_sector_job', lambda: dict(available=False, has_job=False, note='not run'))
    monkeypatch.setattr(jobs, 'submit', lambda request: submitted.append(request) or dict(job_id='c'*32))
    app = AppTest.from_string('from app.backtest_completion_ui import render\nrender()').run()
    assert not app.exception
    next(b for b in app.button if b.label == '新族群資料預檢').click().run()
    assert submitted[-1].kind == 'sector_backtest' and submitted[-1].sector_preflight
    app.checkbox[0].set_value(True)
    next(b for b in app.button if b.label == '重跑新族群帳戶').click().run()
    assert not app.exception
    assert submitted[-1].sector_strict_pit and not submitted[-1].sector_preflight


def test_renderer_uses_latest_job_and_labels_old_summary_separately(prepared, monkeypatch):
    from streamlit.testing.v1 import AppTest
    root, _ = prepared
    path = output_path(root)
    entry.run(path)
    job = completed_job(root, path, jobs.WorkRequest(kind='sector_backtest', sector_preflight=False).model_dump())
    latest = ui.latest_sector_job([job], root)
    old = dict(case_rows=[dict(label='OLD SECTOR RETURN', status='completed_daily', total_return=.7,
                             max_drawdown=-.4)], limitations=[])
    monkeypatch.setattr(ui, 'load', lambda: (dict(disposition=[]), dict(corporate={'cases': {}}, sector=old, data={})))
    monkeypatch.setattr(ui, 'latest_sector_job', lambda: latest)
    original = ui.verified_bytes
    monkeypatch.setattr(ui, 'verified_bytes', lambda descriptor, root=None, suffix='.csv':
                        original(descriptor, prepared[0], suffix))
    app = AppTest.from_string('from app.backtest_completion_ui import render\nrender()').run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 12
    assert 'OLD SECTOR RETURN' not in str(app.dataframe[0].value)
    assert all(x == '10.00%' for x in app.dataframe[0].value['累積淨報酬'])
    labels = [x.label for x in app.get('download_button')]
    assert '下載最新族群工作報告' in labels and '下載原封存補齊摘要' in labels
    # A newer running job hides these sector returns, while the sealed index
    # remains explicitly available as an older, separate artifact.
    monkeypatch.setattr(ui, 'latest_sector_job', lambda: dict(available=False, has_job=True,
        job=dict(job_id='b'*32), note='running'))
    app.run()
    assert not app.exception and not app.dataframe


def test_worker_timeout_stops_without_publishing_performance(prepared, monkeypatch):
    from app import workbench_worker as worker
    root, _ = prepared
    path = output_path(root)
    path.parent.mkdir(parents=True)
    job = dict(job_id='a'*32, request=jobs.WorkRequest(kind='sector_backtest').model_dump(), status='queued')
    captured = []
    monkeypatch.setattr(worker, 'ROOT', root)
    monkeypatch.setattr(worker, 'JOBS_DIR', path.parent)
    monkeypatch.setattr(worker, 'read_job', lambda _: dict(job))
    monkeypatch.setattr(worker, 'write_job', lambda row: captured.append(dict(row)))
    def timed_out(*args, **kwargs):
        assert kwargs['timeout'] == 1800
        raise worker.subprocess.TimeoutExpired('sector', 1800)
    monkeypatch.setattr(worker.subprocess, 'run', timed_out)
    worker.main(job['job_id'])
    assert captured[-1]['status'] == 'failed'
    assert 'result_path' not in captured[-1] and 'summary' not in captured[-1]
