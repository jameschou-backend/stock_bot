import hashlib
import json
import pytest
from streamlit.testing.v1 import AppTest
from app.contingent_execution_ui import load_report


def test_missing_inventory_shows_no_backtest_return(tmp_path):
    script='from pathlib import Path\nfrom app.contingent_execution_ui import render\nrender(Path('+repr(str(tmp_path/'missing.json'))+'))'
    app=AppTest.from_string(script).run()
    assert not app.exception and app.info and not app.dataframe


@pytest.mark.parametrize('change',[{'total_return':6.37},{'historical_replay_completed':True},{'live_qualified':True}])
def test_inventory_cannot_claim_historical_performance(tmp_path,change):
    report=dict(scope='legacy_path_dependency_inventory',audit_completed=True,
                historical_replay_completed=False,live_qualified=False,total_return=None,network_calls=0)
    report.update(change);path=tmp_path/'r.json';path.write_text(json.dumps(report))
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='完整逐筆'):load_report(path,tmp_path)


def test_changed_published_results_are_rejected(tmp_path):
    path=tmp_path/'r.json';path.write_text('{}');path.with_suffix('.sha256').write_text('old')
    with pytest.raises(ValueError,match='檔案已變動'):load_report(path,tmp_path)
