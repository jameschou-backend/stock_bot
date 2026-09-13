import hashlib
import json
import pytest
from streamlit.testing.v1 import AppTest
from app.contingent_replay_ui import load_report


def test_missing_replay_publication_has_no_claim(tmp_path):
    script='from pathlib import Path\nfrom app.contingent_replay_ui import render\nrender(Path('+repr(str(tmp_path/'missing.json'))+'))'
    app=AppTest.from_string(script).run()
    assert not app.exception and not app.dataframe


@pytest.mark.parametrize('change',[{'total_return':6.37},{'historical_replay_completed':True}, {'live_qualified':True}])
def test_demo_cannot_become_strategy_performance(tmp_path,change):
    report=dict(scope='contingent_adapter_demonstration',live_qualified=False,total_return=None,
        historical_replay_completed=False,missing_historical_odd=True,network_calls=0)
    report.update(change);path=tmp_path/'demo.json';path.write_text(json.dumps(report))
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='完整歷史'):load_report(path,tmp_path)
