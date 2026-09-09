import os
from types import SimpleNamespace
import pytest
from app import workbench_jobs as jobs


@pytest.fixture
def store(tmp_path,monkeypatch):
    monkeypatch.setattr(jobs,'JOBS_DIR',tmp_path)
    return tmp_path


def test_duplicate_click_starts_only_one_process(store,monkeypatch):
    starts=[]
    def launch(*args,**kwargs):
        starts.append(args)
        return SimpleNamespace(pid=os.getpid())
    monkeypatch.setattr(jobs.subprocess,'Popen',launch)
    request=jobs.WorkRequest(kind='backtest',months=3)
    first=jobs.submit(request)
    assert jobs.submit(request)['job_id']==first['job_id']
    assert len(starts)==1
    with pytest.raises(ValueError,match='已有工作'):
        jobs.submit(jobs.WorkRequest(kind='update_data'))


def test_failed_launch_and_result_files_do_not_break_history(store,monkeypatch):
    def fail(*a,**kw): raise OSError('cannot execute')
    monkeypatch.setattr(jobs.subprocess,'Popen',fail)
    with pytest.raises(ValueError,match='無法啟動'):
        jobs.submit(jobs.WorkRequest(kind='backtest'))
    (store/('a'*32+'.result.json')).write_text('{"summary":{}}')
    assert len(jobs.recent_jobs())==1
    assert jobs.recent_jobs()[0]['status']=='failed'


def test_research_command_requires_cost_slippage_and_next_day_execution(tmp_path):
    command=jobs.command_for(jobs.WorkRequest(kind='backtest'),tmp_path/'result.json')
    assert command[command.index('--entry-delay')+1]=='1'
    assert command[command.index('--cost')+1]=='0.00585'
    assert '--slippage' in command
    with pytest.raises(ValueError):
        jobs.WorkRequest(kind='backtest',cost=0)
    with pytest.raises(ValueError):
        jobs.job_path('../../.env')


def test_news_jobs_have_explicit_network_scope_and_json_safe_dates(tmp_path):
    import json
    from datetime import date
    request=jobs.WorkRequest(kind='news_review',news_stock_id='2408',news_end=date(2025,7,10),news_days=100)
    command=jobs.command_for(request,tmp_path/'result.json')
    assert 'review' in command and '--fetch' not in command
    assert command[command.index('--end')+1]=='2025-07-10'
    assert json.loads(json.dumps(request.model_dump(mode='json')))['news_end']=='2025-07-10'
    with pytest.raises(ValueError): jobs.WorkRequest(kind='news_scan',news_days=366,fetch_news=True)
    with pytest.raises(ValueError): jobs.WorkRequest(kind='news_review',fetch_news=True)
    with pytest.raises(ValueError): jobs.WorkRequest(kind='news_review',news_stock_id='../../.env')
