from pathlib import Path
import hashlib
import json
import pytest
from app.index_continuous_ui import load,overview,PUBLICATION,ARMS

def fixture(root):
    def save(path,value):
        path=root/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        return dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    cases={};benchmarks={};files={};controls={}
    for name in [*[f'{a}_{m}' for a in ARMS for m in range(8)],'benchmark_control','benchmark_combined',
                 *[f'reference_{p}_{m}' for p in ('early','recent') for m in ('control','combined')]]:
        ref=save(Path('.cache/continuous/cases')/(name+'.json'),{})
        row=dict(completed=True,result=ref,summary=dict(total_return=.5))
        files['cases/'+name+'.json']=ref['sha256']
        if name.startswith('reference_'):controls[name]=dict(row,account_identical=True,return_difference=0.)
        elif name.startswith('benchmark_'):benchmarks[name.removeprefix('benchmark_')]=row
        else:cases[name]=row
    report=dict(cases=cases,benchmarks=benchmarks,validation=dict(arms={a:dict(benchmark_winning_stresses=0) for a in ARMS}),
                data_quality=dict(strict_data_ready=False),reference_controls=controls,all_completed=True)
    ref=save(Path('.cache/continuous/report.json'),report);files['report.json']=ref['sha256']
    limit=save(Path('.cache/continuous/limit-audit.json'),{})
    files['limit-audit.json']=limit['sha256']
    first=save(Path('.cache/continuous/manifest.json'),dict(files_sha256=files))
    second=save(Path('.cache/recheck/manifest.json'),dict(files_sha256=files))
    proof=save(Path('.cache/proof.json'),dict(passed=True,compared_cases=38,newly_executed_cases=76,runs=[first,second]))
    impact=save(Path('.cache/impact.json'),dict(controls={k:dict(new_result=v['result']) for k,v in controls.items()}))
    value=dict(report,schema='index_continuous_publication_v1',start='2016-01-04',end='2026-09-09',initial_cash=1000000,
        adopted=False,live_qualified=False,unseen_validation=False,strict_data_ready=False,run_manifest=first,offline_verification=proof,limit_reconciliation=limit,reference_impact=impact)
    def publish():
        ref=save(PUBLICATION,value);(root/PUBLICATION.with_suffix('.sha256')).write_text(ref['sha256'])
    publish();return value,publish

def test_continuous_summary_keeps_scope_and_rejected_primary(tmp_path):
    fixture(tmp_path);v=load(tmp_path);o=overview(tmp_path)
    assert len(v['cases'])==32 and len(v['benchmarks'])==2
    assert o['available'] and o['arms']['trend200']['winning_stresses']==0
    assert o['start']=='2016-01-04' and o['live_qualified'] is False

@pytest.mark.parametrize('change',['live','dates','validation','missing_case','second_run'])
def test_tampered_continuous_evidence_is_not_presented_as_success(tmp_path,change):
    v,publish=fixture(tmp_path)
    if change=='live':v['live_qualified']=True
    if change=='dates':v['start']='2022-01-03'
    if change=='validation':v['validation']['arms']['trend200']['benchmark_winning_stresses']=8
    if change=='missing_case':v['cases'].pop('trend200_7')
    if change=='second_run':(tmp_path/'.cache/recheck/manifest.json').write_text('{}')
    publish()
    with pytest.raises(ValueError):load(tmp_path)
    assert overview(tmp_path)['available'] is False
