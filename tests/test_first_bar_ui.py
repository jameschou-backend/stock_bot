import hashlib
import json
import pytest
from app.first_bar_ui import PUBLICATION, load, case_names, timing_rows


def put(root,name,payload):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(payload))
    return dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def seal(root,**changes):
    cases={n:dict(completed=False,reason='missing',result=put(root,'.cache/a/'+n+'.json',{})) for n in case_names()}
    r=dict(schema='first_bar_v1',completed=True,live_qualified=False,adopted=False,unseen_validation=False,
        historical_first_publication_verified=False,theme_filter_included=False,price_statistics_are_account_returns=False,
        all_accounts_completed=False,cases=cases,source_sha256={})
    r.update(changes);report=put(root,'.cache/a/report.json',r)
    files={n+'.json':c['result']['sha256'] for n,c in cases.items()}
    manifest=dict(source_sha256={},files_sha256=dict(files,**{'report.json':report['sha256']}))
    runs=[put(root,'.cache/'+d+'/manifest.json',manifest) for d in ('a','b')]
    pub=dict(schema='first_bar_publication_v1',report=report,reproducibility=dict(passed=True,runs=runs,files_sha256=files))
    p=put(root,PUBLICATION,pub);(root/PUBLICATION).with_suffix('.sha256').write_text(p['sha256'])
    return r


def test_unfinished_accounts_and_false_qualification_are_preserved(tmp_path):
    seal(tmp_path);r,p=load(tmp_path)
    assert not r['all_accounts_completed'] and len(r['cases'])==20
    for flag in ('live_qualified','historical_first_publication_verified','price_statistics_are_account_returns'):
        seal(tmp_path,**{flag:True})
        with pytest.raises(ValueError):load(tmp_path)


def test_unknown_price_statistics_are_not_formatted_as_zero():
    r=dict(statistics=[dict(lag=8,horizon=20,phase='replication',group='concentrated',events=2,paired_known=0,
        paired_first_mean=None,paired_wait_mean=None,paired_difference_mean=None,first_excess=None,false_start5=None,unknown=2)])
    assert timing_rows(r,8,20,'replication')[0]['第一根後平均漲幅']=='未知'


def test_output_or_scope_tampering_is_rejected(tmp_path):
    seal(tmp_path);(tmp_path/'.cache/a/report.json').write_text('{}')
    with pytest.raises(ValueError):load(tmp_path)
    seal(tmp_path,cases={})
    with pytest.raises(ValueError):load(tmp_path)
