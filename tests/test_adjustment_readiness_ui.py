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
    at=AppTest.from_string('from app.adjustment_readiness_ui import render\nrender()').run()
    assert not at.exception
    assert any('457 筆可比較，1 筆待補' in item.value for item in at.markdown)
    assert any('草稿尚未套用正式資料' in item.value for item in at.warning)
