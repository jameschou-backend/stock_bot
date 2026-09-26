from pathlib import Path
import hashlib
import json

import pytest

from app.research_validation_ui import load_summary,comparison_rows,load_uncertainty,uncertainty_rows

ARMS={'valid1':'原當日有效','valid2':'多候補一天'}


def fixture(root):
    def save(path,data):
        path=root/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,sort_keys=True))
        return dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    cases={};files={}
    for arm in ARMS:
        for m in range(8):
            name=f'{arm}_{m}';ref=save(Path('.cache/run/cases')/(name+'.json'),{})
            files[f'cases/{name}.json']=ref['sha256']
            complete=name!='valid2_7'
            cases[name]=dict(completed=complete,result=ref,reason=None if complete else 'missing terms',
                summary=dict(total_return=1.,max_drawdown=-.2) if complete else None,
                metrics=dict(excess_return=.5,benchmark_return=.5) if complete else None)
    report=dict(cases=cases,all_completed=False)
    ref=save(Path('.cache/run/report.json'),report);files['report.json']=ref['sha256']
    first=save(Path('.cache/run/manifest.json'),dict(files_sha256=files))
    second=save(Path('.cache/recheck/manifest.json'),dict(files_sha256=files))
    proof=save(Path('.cache/proof.json'),dict(passed=True,compared_cases=16,all_completed=False,runs=[first,second]))
    value=dict(report,schema='candidate_queue_publication_v1',start='2022-01-03',end='2026-09-09',
        initial_cash=1_000_000,live_qualified=False,unseen_validation=False,adopted=False,
        offline_verification=proof,run_manifest=first)
    path=root/'artifacts/report.json';ref=save(Path('artifacts/report.json'),value)
    path.with_suffix('.sha256').write_text(ref['sha256'])
    return path,value


def test_summary_retains_blocked_cases_and_denominator(tmp_path):
    path,_=fixture(tmp_path);report=load_summary(path,'candidate_queue',ARMS,tmp_path)
    rows=comparison_rows(report,ARMS)
    assert rows[1]['全部壓力淨報酬']=='資料阻擋'
    assert rows[1]['跑贏0050的情境']=='7/8' and rows[1]['未完成情境']==1


@pytest.mark.parametrize('mutation',['return','live','omitted'])
def test_resigned_summary_cannot_override_replayed_results(tmp_path,mutation):
    path,value=fixture(tmp_path)
    if mutation=='return':value['cases']['valid1_0']['summary']['total_return']=999
    elif mutation=='live':value['live_qualified']=True
    else:del value['cases']['valid2_7']
    path.write_text(json.dumps(value));path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):load_summary(path,'candidate_queue',ARMS,tmp_path)


def test_changed_second_run_proof_is_rejected(tmp_path):
    path,_=fixture(tmp_path)
    (tmp_path/'.cache/recheck/manifest.json').write_text('{}')
    with pytest.raises(ValueError):load_summary(path,'candidate_queue',ARMS,tmp_path)


@pytest.mark.parametrize('mutation',[None,'different_run','live','missing_case'])
def test_statistics_are_bound_to_the_displayed_cases(tmp_path,mutation):
    _,research=fixture(tmp_path)
    cases={name:dict(available=row['completed'],scope='descriptive_current_account_not_selection_adjusted',
        live_qualified=False,dsr=dict(available=False),bootstrap=dict(excess_sharpe_observed=.1,
        excess_ci_low=-.2,excess_ci_high=.3)) for name,row in research['cases'].items()}
    manifest=research['run_manifest']
    stats=dict(schema='account_statistics_v1',live_qualified=mutation=='live',unseen_validation=False,
        selection_adjusted_statistics_verified=False,complete_trial_coverage_verified=False,
        source_sha256={manifest['path']:manifest['sha256']},
        studies={str(Path(manifest['path']).parent):dict(cases=cases)})
    if mutation=='different_run':stats['source_sha256'][manifest['path']]='0'*64
    if mutation=='missing_case':cases.pop('valid2_0')
    full=tmp_path/'.cache/stats.json';full.write_text(json.dumps(stats))
    publication=dict(schema='account_statistics_publication_v1',report=dict(
        path=str(full.relative_to(tmp_path)),sha256=hashlib.sha256(full.read_bytes()).hexdigest()))
    path=tmp_path/'artifacts/stats.json';path.write_text(json.dumps(publication))
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    if mutation:
        with pytest.raises(ValueError):load_uncertainty(path,research,tmp_path)
    else:
        loaded=load_uncertainty(path,research,tmp_path)
        rows=uncertainty_rows(loaded,ARMS)
        assert len(rows)==3 and rows[0]['95%區間']=='-0.20 ～ 0.30'
