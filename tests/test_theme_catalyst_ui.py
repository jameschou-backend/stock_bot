import hashlib
import json
import pytest
from app.theme_catalyst_ui import load, comparison_rows, case_names, PUBLICATION


def save(root,path,value):
    file=root/path;file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(value))
    return dict(path=path,sha256=hashlib.sha256(file.read_bytes()).hexdigest())


def fixture(root,**changes):
    cases={}
    for name in case_names():
        descriptor=save(root,'.cache/a/'+name+'.json',{})
        row=dict(completed=True,summary=dict(total_return=.2,max_drawdown=-.1),candidate_count=1,
            result=descriptor,average_cash_fraction=.5)
        if not name.startswith('benchmark'):row.update(benchmark_return=.2,excess_return=0.)
        cases[name]=row
    report=dict(schema='theme_catalyst_v1',completed=True,cases=cases,all_accounts_completed=True,
        live_qualified=False,adopted=False,unseen_validation=False,
        historical_first_publication_verified=False,valid_unbiased_strategy_evidence=False,source_sha256={})
    report.update(changes)
    desc=save(root,'.cache/a/report.json',report)
    files={name+'.json':r['result']['sha256'] for name,r in cases.items()}
    manifest=dict(files_sha256=dict(files,**{'report.json':desc['sha256']}),source_sha256={})
    runs=[save(root,f'.cache/{folder}/manifest.json',manifest) for folder in ('a','b')]
    pub=dict(schema='theme_catalyst_publication_v1',report=desc,
        reproducibility=dict(passed=True,runs=runs,files_sha256=files))
    ref=save(root,PUBLICATION,pub);(root/PUBLICATION).with_suffix('.sha256').write_text(ref['sha256'])
    return report,pub


def test_qualification_and_seal_are_enforced(tmp_path):
    report,_=fixture(tmp_path)
    assert load(tmp_path)==report
    fixture(tmp_path,live_qualified=True)
    with pytest.raises(ValueError):load(tmp_path)
    fixture(tmp_path,historical_first_publication_verified=True)
    with pytest.raises(ValueError):load(tmp_path)


def test_missing_account_cannot_be_displayed_as_zero_return():
    rows={}
    for name in case_names():
        rows[name]=dict(completed=False,candidate_count=3)
    table=comparison_rows({'cases':rows},0)
    assert len(table)==4 and all(r['一般淨報酬']=='資料不足，未完成' for r in table)


def test_changed_report_and_incomplete_comparison_rejected(tmp_path):
    report,pub=fixture(tmp_path)
    (tmp_path/pub['report']['path']).write_text('{}')
    with pytest.raises(ValueError):load(tmp_path)
    fixture(tmp_path,cases={})
    with pytest.raises(ValueError):load(tmp_path)
