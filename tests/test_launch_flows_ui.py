import hashlib
import json
import pytest
from app.launch_flows_ui import PUBLICATION,load,fmt


def put(root,name,data):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
    return dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def seal(root,**changes):
    r=dict(schema='launch_flows_v1',completed=True,live_qualified=False,adopted=False,unseen_validation=False,
        portfolio_returns_computed=False,historical_first_publication_verified=False,strategy_net_return=None,
        source_sha256={},artifacts={})
    r.update(changes);report=put(root,'.cache/a/report.json',r)
    manifest=dict(report_sha256=report['sha256'],source_sha256={},files={})
    runs=[put(root,f'.cache/{s}/manifest.json',manifest) for s in ('a','b')]
    d=put(root,PUBLICATION,dict(schema='launch_flows_publication_v1',report=report,
        reproducibility=dict(passed=True,runs=runs,csv_sha256={})))
    (root/PUBLICATION).with_suffix('.sha256').write_text(d['sha256'])


def test_publication_must_remain_descriptive(tmp_path):
    seal(tmp_path);assert load(tmp_path)['strategy_net_return'] is None
    seal(tmp_path,portfolio_returns_computed=True)
    with pytest.raises(ValueError):load(tmp_path)
    seal(tmp_path,live_qualified=True)
    with pytest.raises(ValueError):load(tmp_path)


def test_report_tampering_is_rejected(tmp_path):
    seal(tmp_path);(tmp_path/'.cache/a/report.json').write_text('{}')
    with pytest.raises(ValueError):load(tmp_path)


def test_unknown_zero_and_capped_streak_remain_distinct():
    assert fmt(None)=='未知' and fmt(0)=='0.0'.join(['+',''])
    assert fmt(20,'streak')=='至少20' and fmt(0,'bool')=='否'
