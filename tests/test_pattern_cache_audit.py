from copy import deepcopy
import pytest
from skills.pattern_cache_audit import audit_cache_reuse


def fixture():
    arms={'control':('control',False),'pattern':('control',True)}
    files={};parent={'cases':{}};cases={};trials=[]
    for mask in range(8):
        ref={'path':f'old/control_{mask}.json','sha256':str(mask)}
        case={'completed':True,'config':{'factor_mask':mask,'technical_mode':'control'},'account':{'cash':100}}
        files[ref['path']]=case;parent['cases'][f'control_{mask}']={'result':ref}
        for arm in arms:
            fresh=arm=='pattern';name=f'{arm}_{mask}'
            c=deepcopy(case)
            if fresh:c['config']['pattern_filter']=True
            target={'path':f'new/{name}.json','sha256':str(mask)};files[target['path']]=c
            cases[name]={'completed':True,'config':deepcopy(c['config']),'result':target,
                'reused_case':None if fresh else ref,'cache_hit':not fresh}
            if fresh:trials.append({'case':name,'status':'completed','result_path':target['path']})
    return dict(cases=cases,trial_count=8,cache_hits=8),parent,trials,arms,files


def test_cached_accounts_are_not_counted_as_new_executions():
    report,parent,trials,arms,files=fixture()
    assert audit_cache_reuse(report,parent,trials,arms,lambda r:files[r['path']])==dict(
        exact_parent_accounts_reused=True,newly_executed_cases=8,cache_hits=8)


@pytest.mark.parametrize('change',['cash','count','duplicate_trial','wrong_parent','fake_reuse','wrong_status'])
def test_cache_or_trial_misrepresentation_is_rejected(change):
    report,parent,trials,arms,files=fixture()
    if change=='cash':files['new/control_0.json']['account']['cash']=999
    elif change=='count':report['trial_count']=16
    elif change=='duplicate_trial':trials[-1]=trials[0]
    elif change=='wrong_parent':report['cases']['control_0']['reused_case']=parent['cases']['control_1']['result']
    elif change=='fake_reuse':report['cases']['pattern_0']['cache_hit']=True
    else:trials[0]['status']='blocked'
    with pytest.raises(ValueError):audit_cache_reuse(report,parent,trials,arms,lambda r:files[r['path']])
