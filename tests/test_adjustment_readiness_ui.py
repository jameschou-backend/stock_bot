import hashlib
import json
import pytest
from streamlit.testing.v1 import AppTest
from app import adjustment_readiness_ui as ui


def test_audit_snapshot_rejects_modified_or_missing_report(tmp_path):
    report=tmp_path/'report.json'
    report.write_text(json.dumps(dict(live_qualified=False,database_mutations=0)))
    summary=dict(report_path='report.json',report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                 live_qualified=False,applied_to_production=False)
    path=tmp_path/'summary.json';path.write_text(json.dumps(summary))
    assert ui.load_summary(path,tmp_path)==summary
    report.write_text('{}')
    with pytest.raises(ValueError,match='已變更'):ui.load_summary(path,tmp_path)
    report.unlink()
    with pytest.raises(OSError):ui.load_summary(path,tmp_path)


def test_display_does_not_present_partial_audit_as_live_qualification(monkeypatch):
    monkeypatch.setattr(ui,'load_summary',lambda:dict(source_windows=302,shadow_stocks=2027,shadow_rows=4616245,
        candidate_count=458,candidate_unresolved=1,candidate_changed_above_0_1pp=0,cross_market_transitions=10,unresolved_post_end_companies=22))
    monkeypatch.setattr(ui,'load_completion',lambda:dict(
        price_repair=dict(reviewed_stocks=22,raw_rows=64,invalidated_labels=61),
        derived_rebuild=dict(rows=125940),
        market_identity=dict(current_rows=1987,current_ordinary_stocks=1945,ended_episodes=60,
                             signal_rows=458,missing_historical_starts=41),
        odd_lot_sample=dict(record_count=10)))
    at=AppTest.from_string('from app.adjustment_readiness_ui import render\nrender()').run()
    assert not at.exception
    assert any('457 筆可比較，1 筆待補' in item.value for item in at.markdown)
    assert any('未套用的研究草稿' in item.value for item in at.warning)
    assert any('64 筆錯誤行情已隔離' in item.value for item in at.success)
    assert any('41 筆掛牌起日待核實' in item.value for item in at.markdown)


def test_completion_requires_exact_repair_and_rebuild_evidence(tmp_path):
    registry=tmp_path/'registry.json';registry.write_text('{}')
    digest=hashlib.sha256(registry.read_bytes()).hexdigest()
    docs=dict(price_registry={},price_repair=dict(applied=True,registry_sha256=digest),
              derived_rebuild=dict(completed=True,registry_sha256=digest))
    sources={}
    for key,value in docs.items():
        path=tmp_path/(key+'.json');path.write_text(json.dumps(value))
        sources[key]=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path=tmp_path/'summary.json'
    path.write_text(json.dumps(dict(schema='historical_data_completion_v1',live_qualified=False,sources=sources)))
    assert ui.load_completion(path,tmp_path)['price_repair']['applied']
    (tmp_path/'derived_rebuild.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):ui.load_completion(path,tmp_path)
