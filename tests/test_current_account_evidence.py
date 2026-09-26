import hashlib
import json

import pytest

from app.current_account_evidence import overview
from test_research_validation_ui import fixture


def publication(root,wrong_scope=False):
    _,value=fixture(root)
    for name,row in value['cases'].items():
        row['config']=dict(factor_mask=int(name.rsplit('_',1)[1]),benchmark=False,
                           board_only=True,position_count=4 if wrong_scope else 5)
    def save(path,value):
        path.write_text(json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    report=root/'.cache/run/report.json'
    digest=save(report,dict(cases=value['cases'],all_completed=False))
    manifest=root/'.cache/run/manifest.json';m=json.loads(manifest.read_text())
    m['files_sha256']['report.json']=digest
    value['run_manifest']['sha256']=save(manifest,m)
    proof=root/'.cache/proof.json';p=json.loads(proof.read_text());p['runs'][0]=value['run_manifest']
    value['offline_verification']['sha256']=save(proof,p)
    target=root/'artifacts/forward_simulation/candidate_queue_20260927.json';target.parent.mkdir()
    target.with_suffix('.sha256').write_text(save(target,value))
    return target


def test_current_evidence_preserves_denominator_and_missing_results(tmp_path):
    publication(tmp_path)
    value=overview(tmp_path);family=value['families']['candidate_queue']
    assert family['available'] and value['idle_capital']=='cash' and value['live_qualified'] is False
    arm=family['arms'][1]
    assert arm['completed_cases']==7 and arm['winning_stresses']==7 and arm['stress_count']==8
    assert arm['all_stresses']==dict(available=False,reason='missing terms')
    assert value['families']['support_risk']['reason']=='verified_publication_missing'


@pytest.mark.parametrize('changed',['publication','manifest','scope'])
def test_changed_or_incompatible_current_reports_cannot_fall_back_to_old_returns(tmp_path,changed):
    target=publication(tmp_path,wrong_scope=changed=='scope')
    if changed=='publication':target.write_text('{}')
    elif changed=='manifest':(tmp_path/'.cache/run/manifest.json').write_text('{}')
    value=overview(tmp_path)['families']['candidate_queue']
    assert value['available'] is False and value['reason']=='publication_verification_failed'
    assert 'arms' not in value and value['live_qualified'] is False


def test_missing_publications_are_explicit_and_never_live(tmp_path):
    value=overview(tmp_path)
    assert all(not f['available'] and not f['live_qualified'] for f in value['families'].values())


def test_http_evidence_prefers_verified_current_accounts_and_retains_legacy_context(tmp_path,monkeypatch):
    import asyncio
    import importlib
    import httpx
    from fastapi import FastAPI
    from app import workbench_service as service
    from app.workbench_api import router
    publication(tmp_path)
    monkeypatch.setattr(service,'ROOT',tmp_path)
    for name in ('capacity_research','regime_switch_research','diffusion_research',
                 'guidance_research','revenue_research','event_group_research'):
        monkeypatch.setattr(importlib.import_module('app.'+name),'overview',lambda:{'available':False})
    for name in ('rule_research_overview','flow_research_overview','theme_research_overview'):
        monkeypatch.setattr(service,name,lambda:{'available':True,'scope':'legacy_fixture'})
    app=FastAPI();app.include_router(router)
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app,client=('127.0.0.1',50000)),base_url='http://localhost') as client:
            return await client.get('/workbench/evidence')
    response=asyncio.run(request())
    assert response.status_code==200
    value=response.json()
    assert value['preferred_comparison']=='current_accounts' and value['live_qualified'] is False
    assert value['current_accounts']['families']['candidate_queue']['arms'][0]['normal']['total_return']==1.
    assert value['rule_research']['scope']=='legacy_fixture' and value['legacy_reports_note']
