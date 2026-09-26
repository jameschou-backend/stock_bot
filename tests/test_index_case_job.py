from copy import deepcopy
from pathlib import Path
import json
import socket
import pytest
from skills import index_case_job as job
from app import index_case_backtest_ui as ui
from app import workbench_jobs as jobs
from tests.test_index_exposure_replay import data


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(job,'ROOT',tmp_path);monkeypatch.setattr(job,'CACHE',tmp_path/'.cache/replays')
    x=data(sessions=8);engine=job.IndexExposureReplay(x,0,7);account=engine.run()
    case=dict(completed=True,config=dict(window=0,factor_mask=7),account=account,decisions=engine.decisions,
              plans=engine.plans,pending=engine.pending,summary=job.summarize(account),live_qualified=False,unseen_validation=False)
    case['audit']=job.audit_index(case,x)
    def save(path,value):
        path=tmp_path/path;job.write(path,value)
        return dict(path=str(path.relative_to(tmp_path)),sha256=job.sha(path))
    quote=save(Path('.cache/quotes.json'),x)
    source=save(Path('.cache/sources.json'),{quote['path']:quote['sha256']})
    canonical=save(Path('.cache/canonical.json'),case)
    benchmark=save(Path('.cache/benchmark.json'),case)
    pub=dict(start=x['days'][0],end=x['days'][-1],initial_cash=1000000,source_identity=source,
             cases={'equal_7':dict(result=canonical)},benchmarks={'combined':dict(result=benchmark)},
             data_quality=dict(strict_data_ready=False),limitations=['test fixture'])
    monkeypatch.setattr(job,'publication',lambda root:deepcopy(pub))
    monkeypatch.setattr(job,'detail',lambda *a:(deepcopy(case),deepcopy(case)))
    monkeypatch.setattr(ui,'load',lambda root:deepcopy(pub))
    monkeypatch.setattr(ui,'detail',lambda *a:(deepcopy(case),deepcopy(case)))
    calls=[];trials=[]
    def inputs():calls.append('input');return deepcopy(x)
    monkeypatch.setattr(job,'inputs',inputs)
    monkeypatch.setattr(job,'append_trial_registry',lambda record:trials.append(record))
    for path in ('skills/index_case_job.py','scripts/run_index_case.py','skills/backtest_case_cache.py','docs/index_case_backtest.md'):
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
    return dict(root=tmp_path,case=case,source=tmp_path/quote['path'],calls=calls,trials=trials,save=save)


def test_fresh_replay_cache_reuse_and_forced_replay_have_exact_counts(setup):
    root=setup['root'];results=[]
    for i,fresh in enumerate((False,False,True)):
        path=root/f'.cache/output-{i}.json';value=job.run('equal',7,path,fresh=fresh)
        report,case=ui.load_result(path,root=root,request=dict(kind='index_backtest',index_rule='equal',index_mask=7,index_fresh=fresh))
        assert case==setup['case'] and report==value
        results.append((value['metrics']['executed_cases'],value['metrics']['reused_cases']))
    assert results==[(1,0),(0,1),(1,0)]
    assert len(setup['calls'])==2 and len(setup['trials'])==2
    assert all(t['status']=='completed' for t in setup['trials'])


def test_cache_hit_still_checks_source_bytes(setup):
    root=setup['root'];job.run('equal',7,root/'.cache/first.json')
    setup['source'].write_text('{}')
    with pytest.raises(ValueError,match='Frozen source changed'):job.run('equal',7,root/'.cache/second.json')
    assert len(setup['calls'])==1 and not (root/'.cache/second.json').exists()


def test_changed_source_during_run_cannot_publish_success(setup,monkeypatch):
    real=job.audit_index
    def altered(value,data):
        result=real(value,data);setup['source'].write_text('{}');return result
    monkeypatch.setattr(job,'audit_index',altered)
    with pytest.raises(ValueError,match='during replay'):job.run('equal',7,setup['root']/'.cache/result.json')
    assert setup['trials'][0]['status']=='failed'
    assert not (setup['root']/'.cache/result.json').exists()


def test_changed_account_and_network_call_are_blocked(setup,monkeypatch):
    def network():socket.create_connection(('example.com',443))
    monkeypatch.setattr(job,'inputs',network)
    with pytest.raises(Exception):job.run('equal',7,setup['root']/'.cache/network.json')
    assert setup['trials'][0]['status']=='failed'
    bad=deepcopy(setup['case']);bad['account']['daily'][-1]['nav']+=1
    with pytest.raises(ValueError,match='differs'):job.compare(bad,setup['case'])


@pytest.mark.parametrize('arm,mask',[('best',0),('equal',True),('equal',8),('equal',-1),('equal','7')])
def test_only_fixed_cases_are_accepted(arm,mask):
    with pytest.raises(ValueError):job.validate_choice(arm,mask)


@pytest.mark.parametrize('mutation',['summary','account','source','request','fresh','count','live','hash','period'])
def test_tampered_finished_result_is_rejected(setup,mutation):
    root=setup['root'];path=root/'.cache/run.json';value=job.run('equal',7,path)
    request=dict(kind='index_backtest',index_rule='equal',index_mask=7)
    expected_hash=None
    if mutation=='summary':value['summary']['total_return']+=1
    if mutation=='account':
        c=deepcopy(setup['case']);c['account']['daily'][-1]['nav']+=1
        value['account']=setup['save'](Path('.cache/wrong-account.json'),c)
    if mutation=='source':
        record=json.loads((root/value['source_identity']['path']).read_text());record['identity']['sources'].clear()
        record['identity_digest']=job.content_digest(record['identity'])
        value['source_identity']=setup['save'](Path('.cache/wrong-source.json'),record)
    if mutation=='request':request['index_mask']=0
    if mutation=='fresh':value['metrics'].update(executed_cases=0,reused_cases=1);request['index_fresh']=True
    if mutation=='count':value['metrics']['executed_cases']=True
    if mutation=='live':value['live_qualified']=True
    if mutation=='hash':expected_hash='0'*64
    if mutation=='period':value['start']='2010-01-01'
    job.write(path,value)
    with pytest.raises(ValueError):ui.load_result(path,root=root,request=request,expected_sha256=expected_hash)


def test_no_existing_account_or_result_is_overwritten(setup):
    root=setup['root'];path=root/'.cache/run.json'
    job.write(root/'.cache/run.account.json',{'keep':True})
    with pytest.raises(ValueError,match='new result'):job.run('equal',7,path)
    assert not path.exists() and not setup['calls']


def test_corrupt_cache_receipt_never_counts_as_reused_or_success(setup):
    root=setup['root'];job.run('equal',7,root/'.cache/first.json')
    receipt=next((root/'.cache/replays').glob('*/cases/equal_7/receipt.json'))
    receipt.write_text('{}')
    with pytest.warns(RuntimeWarning,match='Ignoring incomplete'):
        with pytest.raises(ValueError,match='conflicting cached'):
            job.run('equal',7,root/'.cache/second.json')
    assert setup['trials'][-1]['status']=='failed'
    assert not (root/'.cache/second.json').exists()


def test_worker_verifies_report_before_declaring_completed(setup,monkeypatch):
    from app import workbench_worker as worker
    from types import SimpleNamespace
    root=setup['root'];folder=root/'.cache/jobs';folder.mkdir()
    request=jobs.WorkRequest(kind='index_backtest',index_rule='equal',index_mask=7)
    record=dict(job_id='b'*32,request=request.model_dump(mode='json'),status='queued')
    saved=[]
    monkeypatch.setattr(worker,'ROOT',root);monkeypatch.setattr(worker,'JOBS_DIR',folder)
    monkeypatch.setattr(worker,'read_job',lambda _:deepcopy(record))
    monkeypatch.setattr(worker,'write_job',lambda value:saved.append(deepcopy(value)))
    def execute(command,**kwargs):
        job.run('equal',7,Path(command[command.index('--output')+1]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(worker.subprocess,'run',execute)
    worker.main(record['job_id']);last=saved[-1]
    assert last['status']=='completed' and last['research_only'] and last['live_qualified'] is False
    assert last['result_sha256']==job.sha(Path(last['result_path']))
    # A zero process exit alone must not promote a malformed completion file.
    record['job_id']='c'*32
    def malformed(command,**kwargs):
        job.write(Path(command[command.index('--output')+1]),{'completed':True})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(worker.subprocess,'run',malformed)
    worker.main(record['job_id'])
    assert saved[-1]['status']=='failed' and 'result_sha256' not in saved[-1]


def test_index_job_command_is_bounded_and_cannot_fetch(tmp_path):
    request=jobs.WorkRequest(kind='index_backtest',index_rule='trend220',index_mask=7,index_fresh=True)
    command=jobs.command_for(request,tmp_path/'result.json')
    assert command[1:]==['scripts/run_index_case.py','--arm','trend220','--mask','7','--output',str(tmp_path/'result.json'),'--fresh']
    for kwargs in ({'index_mask':True},{'index_mask':8},{'index_rule':'best'},{'fetch_news':True},{'fetch_flow':True}):
        with pytest.raises(ValueError):jobs.WorkRequest(kind='index_backtest',**kwargs)
