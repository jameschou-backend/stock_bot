import json
import pytest
from streamlit.testing.v1 import AppTest
from app.reservation_bridge_ui import load_report


def test_missing_paired_study_does_not_show_returns(tmp_path):
    script = 'from pathlib import Path\nfrom app.reservation_bridge_ui import render\nrender(Path('+repr(str(tmp_path/'missing.json'))+'))'
    app = AppTest.from_string(script).run()
    assert not app.exception and not app.dataframe and app.info


@pytest.mark.parametrize('change', [{'strict_intraday_return':6.37}, {'strict_intraday_completed':True}, {'live_qualified':True}])
def test_daily_proxy_cannot_be_relabelled_as_intraday_qualified(tmp_path, change):
    report = dict(all_completed=True, cases={str(i):{} for i in range(8)},
        live_qualified=False, strict_intraday_completed=False, strict_intraday_return=None,
        offline=dict(identical=True, network_calls=0))
    report.update(change)
    p = tmp_path/'report.json'; p.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='逐筆驗證'):
        load_report(p, tmp_path)


def test_publication_rejects_stale_offline_verification(tmp_path):
    from scripts.export_reservation_bridge import export
    (tmp_path/'summary.json').write_text(json.dumps(dict(all_completed=True, cases={str(i):{} for i in range(8)})))
    (tmp_path/'offline.json').write_text(json.dumps(dict(identical=True, network_calls=0, manifest_sha256='old')))
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='current manifest'):
        export(tmp_path, tmp_path/'delivery.json')
