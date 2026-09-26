"""A bounded offline rerun of one already published continuous ETF account."""
from datetime import datetime,timezone
from pathlib import Path
import time
from app.file_lock import file_lock
from app.index_continuous_ui import load as publication,detail
from scripts.research_exit_scenarios import read,write,sha,summarize
from skills.backtest_case_cache import CaseStore,content_digest,file_identities
from skills.index_continuous_inputs import load as inputs
from skills.index_exposure_replay import IndexExposureReplay
from skills.index_exposure_audit import audit_index
from skills.trial_registry import append_trial_registry
from skills.verified_backtest_tool import offline_only

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.cache/index-case-backtest'
ARMS={'equal':0,'trend200':200,'trend180':180,'trend220':220}
COMPARED=('account','decisions','plans','pending','audit','summary')

def validate_choice(arm,mask):
    if arm not in ARMS or type(mask) is not int or mask not in range(8):
        raise ValueError('Choose one published ETF rule and stress scenario')

def compare(value,expected):
    if value.get('completed') is not True or any(value.get(k) is not False for k in ('live_qualified','unseen_validation')):
        raise ValueError('Incomplete or promoted ETF case')
    if value.get('config')!=expected['config'] or any(value.get(k)!=expected[k] for k in COMPARED):
        raise ValueError('ETF rerun differs from sealed full account or decisions')

def run(arm,mask,output,*,fresh=False):
    validate_choice(arm,mask);output=Path(output).resolve();started=time.perf_counter()
    case_path=output.with_name(output.stem+'.account.json')
    if not output.is_relative_to(ROOT/'.cache') or output.suffix!='.json':
        raise ValueError('Use a new result JSON under the local cache')
    with file_lock(CACHE/'.run.lock',timeout=0),offline_only():
        if output.exists() or case_path.exists():
            raise ValueError('Use a new result JSON under the local cache')
        print('[TIMER] source_validation start',flush=True)
        pub=publication(ROOT);name=f'{arm}_{mask}';expected,benchmark=detail(pub,name,ROOT)
        source=read(ROOT/pub['source_identity']['path'])
        if sha(ROOT/pub['source_identity']['path'])!=pub['source_identity']['sha256']:
            raise ValueError('Source identity changed')
        extra=[ROOT/'skills/index_case_job.py',ROOT/'scripts/run_index_case.py',ROOT/'skills/backtest_case_cache.py',
               ROOT/'docs/index_case_backtest.md',ROOT/pub['source_identity']['path']]
        for ref in (pub['cases'][name]['result'],pub['benchmarks']['combined' if mask&1 else 'control']['result']):
            source[ref['path']]=ref['sha256']
        source.update(file_identities(extra,ROOT))
        if file_identities([ROOT/p for p in source],ROOT)!=source:raise ValueError('Frozen source changed before replay')
        identity=dict(schema='index_case_identity_v1',sources=source,period=[pub['start'],pub['end']],initial_cash=pub['initial_cash'])
        store=CaseStore(CACHE/content_digest(identity),identity);config=expected['config']
        cached=None if fresh else store.load(name,config)
        validation_seconds=time.perf_counter()-started
        print('[TIMER] source_validation done',flush=True)
        if cached is not None:
            compare(cached,expected);value=cached;executed=0
        else:
            error=None;status='failed'
            try:
                print('[TIMER] case start',flush=True)
                data=inputs();engine=IndexExposureReplay(data,ARMS[arm],mask);account=engine.run()
                value=dict(completed=True,config=config,account=account,decisions=engine.decisions,plans=engine.plans,
                    pending=engine.pending,summary=summarize(account),live_qualified=False,unseen_validation=False)
                value['audit']=audit_index(value,data);compare(value,expected)
                if file_identities([ROOT/p for p in source],ROOT)!=source:raise ValueError('Frozen source changed during replay')
                store.save(name,config,value);status='completed';executed=1
                print('[TIMER] case done',flush=True)
            except Exception as exc:
                error=type(exc).__name__+': '+str(exc);raise
            finally:
                append_trial_registry(dict(timestamp=datetime.now(timezone.utc).isoformat(),source='index_case_job',
                    command='scripts/run_index_case.py',case=name,config=config,status=status,error=error,
                    result_path=str(output.relative_to(ROOT))))
        write(case_path,value)
        def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
        result=dict(schema='index_case_job_v1',completed=True,arm=arm,factor_mask=mask,start=pub['start'],end=pub['end'],
            initial_cash=pub['initial_cash'],live_qualified=False,unseen_validation=False,strict_data_ready=False,
            account=ref(case_path),benchmark=pub['benchmarks']['combined' if mask&1 else 'control']['result'],
            source_identity=ref(store.directory/'identity.json'),canonical_case=pub['cases'][name]['result'],
            summary=value['summary'],benchmark_summary=benchmark['summary'],full_account_and_decisions_reproduced=True,
            data_quality=pub['data_quality'],limitations=pub['limitations'],
            metrics=dict(network_calls=0,database_writes=0,executed_cases=executed,reused_cases=1-executed,
                source_validation_seconds=round(validation_seconds,3),elapsed_seconds=round(time.perf_counter()-started,3)))
        write(output,result);return result
